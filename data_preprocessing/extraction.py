import torch
from torchvision import models, transforms
import os
import numpy as np
from PIL import Image
import pandas as pd
import json
import re
import argparse
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tsv(file_path : str) -> pd.DataFrame:

    """
    Overview:
        Load tsv file into dataframe.
    Arguments:
        - file_path (:obj:`str`): Path to tsv file.
    Returns:
        - df (:obj:`pd.DataFrame`): Dataframe.    
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found") 
    if not file_path.endswith(".tsv"):
        raise ValueError("Path must be a .tsv file")
    
    df = pd.read_csv(file_path, sep="\t")
    
    return df


def load_csv(file_path : str) -> pd.DataFrame:

    """
    Overview:
        Load csv file into dataframe.
    Arguments:
        - file_path (:obj:`str`): Path to csv file.
    Returns:
        - df (:obj:`pd.DataFrame`): Dataframe.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found") 
    if not file_path.endswith(".csv"):
        raise ValueError("Path must be a .csv file")
    
    df = pd.read_csv(file_path)
    
    return df


def extract_feature(model : torch.nn.Module,
                    input_image : torch.Tensor,
    ) -> np.ndarray:
    
    """
    Overview:
        Extract feature from input image using model.
    Arguments:
        - model (:obj:`torch.nn.Module`): Model to extract feature.
        - input_image (:obj:`torch.Tensor`): Input image.
    Returns:
        - output_feature (:obj:`torch.Tensor`): Output feature.
    """

    output_feature = model(input_image)
    output_feature = output_feature.squeeze().cpu().detach().numpy()

    return output_feature


def get_feats(model: torch.nn.Module, norms, transformer, device):
    feats = []
    for norm in norms:
        norm = Image.fromarray(norm)
        norm = transformer(norm).unsqueeze(0).to(device)
        feat = model(norm).detach()
        feats.append(feat.squeeze())
        
    return feats


def generate_survival_dataframe(model : torch.nn.Module,
                                transformer : transforms.Compose,
                                folder_path : str, 
                                sample_data : dict,
    ) -> pd.DataFrame:

    """
    Overview:
        Generate survival dataframe.
    Arguments:
        - model (:obj:`torch.nn.Module`): Model to extract feature.
        - transformer (:obj:`transforms.Compose`): Data transformer.
        - folder_path (:obj:`str`): Path to normalized image folder.
        - samples_data (:obj:`dict`): Dictionary of samples data.
        - metadata (:obj:`list`): List of metadata.
    Returns:
        - df (:obj:`pd.DataFrame`): Dataframe.
    """
    
    # get all image names
    image_names = os.listdir(folder_path)
    image_names = [image_name for image_name in image_names if image_name.endswith(".jpg")]

    data = {}
    data["bcr_patient_barcode"] = sample_data["_PATIENT"]
    data["vital_status"] = sample_data["OS"]
    data["days"] = sample_data["OS.time"]

    features = []
    for image_name in tqdm(image_names, leave=True):
        image_path = os.path.join(folder_path, image_name)
        image = Image.open(image_path)
        image = transformer(image).unsqueeze(0).to(device)
        feature = extract_feature(model, image)
        features.append(feature)

    features = np.array(features)
    features = features.transpose()

    for i in range(features.shape[0]):
        data[str(i)] = features[i].tolist() # TODO: check

    return pd.DataFrame(data)


def survival_prediction_preprocessing(normalized_path : str, 
                                      COLD_tsv_path : str,
                                      READ_tsv_path : str, 
                                      metadata_json_path : list,
                                      save_path : str=None,
    ) -> None:
    
    """
    Overview:
        Preprocessing for survival prediction.
    Arguments:
        - normalized_path (:obj:`str`): Path to normalized image folders.
        - COLD_tsv_path (:obj:`str`): Path to COLD tsv file.
        - READ_tsv_path (:obj:`str`): Path to READ tsv file.
        - metadata_json_path (:obj:`list`): List of metadata json path.
        - save_path (:obj:`str`): Path to save dataframe.
    Returns:
        - None
    """

    os.makedirs(save_path, exist_ok=True)

    # load dataframe
    COLD_df = load_tsv(COLD_tsv_path)
    READ_df = load_tsv(READ_tsv_path)
    tsv_df = pd.concat([COLD_df, READ_df])
    tsv_df = tsv_df.reset_index(drop=True)
    samples_data = tsv_df.set_index("sample").to_dict(orient='index')

    # load metadata
    with open(metadata_json_path, 'r') as fp:
        metadata = json.load(fp)
    
    # load model
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) # load pretrained model
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1]) # remove last layer
    resnet50.to(device)

    # load data transformer
    transformer = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get all folders under normalized_path
    folder_names = os.listdir(normalized_path)
    sample_pattern = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{3}"
    for i, folder_name in enumerate(folder_names):
        match = re.search(sample_pattern, metadata[i]["submitter_id"])
        if match:
            sample_id = match.group()
        else:
            raise ValueError("Sample ID not found")
        sample_data = samples_data[sample_id]

        folder_path = os.path.join(normalized_path, folder_name)
        df = generate_survival_dataframe(resnet50, transformer, folder_path, sample_data)
        
        if save_path is not None:
            df.to_pickle(os.path.join(save_path, folder_name + ".pkl"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--normalized_path', type=str)
    parser.add_argument('--COLD_tsv_path', type=str)
    parser.add_argument('--READ_tsv_path', type=str)
    parser.add_argument('--metadata_json_path', type=str)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    survival_prediction_preprocessing(args.normalized_path,
                                      args.COLD_tsv_path,
                                      args.READ_tsv_path,
                                      args.metadata_json_path,
                                      args.save_path) 
