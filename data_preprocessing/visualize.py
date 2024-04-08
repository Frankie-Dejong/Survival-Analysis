import os
from PIL import Image
import numpy as np

def vis_normed_res(normed_path: str, vis_path: str):
    normed_results = np.load(normed_path)
    os.makedirs(vis_path, exist_ok=True)
    for i, img in enumerate(normed_results):
        Image.fromarray(img).save(os.path.join(vis_path, str(i) + ".jpg")) 

if __name__ == "__main__":

    normed_path = "/media/kevin/DATA2/cancer_prediction_frb/dataset/TCGA/normed/TCGA-A6-2671-01A.npy"
    vis_path = "./vis_results"

    vis_normed_res(normed_path, vis_path)

