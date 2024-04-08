import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import warnings
import multiprocessing as mp
# warnings.filterwarnings("ignore")


def normalize_color(file_path : str, 
                    save_path : str='./normalized'):
    '''

    Input:
        file_path: file path of images
        save_path: path to save images
    Output:
        Inorm: normalized image
    '''

    if not os.path.exists(file_path):
        exit("Error: File path doesn't exist")

    print(f"Normalizing color in {file_path} ...")
    os.makedirs(save_path, exist_ok=True)

    img_names = os.listdir(file_path)
    # for mac os
    if '.DS_Store' in img_names:
        img_names.remove('.DS_Store')
    for img_name in tqdm(img_names):
        img = np.array(Image.open(os.path.join(file_path, img_name)))
        normalize_staining(img, img_name, save_path=save_path)


def normalize_color_folders(file_path : str,
                            save_path : str):
    '''

    Input:
        file_name: file path list
        save_path: save path list
    Output:
        Inorm: normalized image
    '''
    if not os.path.exists(file_path):
        exit("Error: File path doesn't exist")

    os.makedirs(save_path, exist_ok=True)

    folder_names = os.listdir(file_path)
    for folder_name in folder_names:
        normalize_color(os.path.join(file_path, folder_name), os.path.join(save_path, folder_name))


def get_norms(tiles, io, alpha, beta, filter_size):
    norms = []
    for tile in tiles:
        tile = np.array(tile)
        norm = normalize_staining(tile, "", None, io, alpha, beta, filter_size)
        if norm is not None:
            norms.append(norm)

    return norms


def normalize_staining(img : np.ndarray,
                       name : str, 
                       save_path : str=None, 
                       io : int=240, 
                       alpha : int=1, 
                       beta : float=0.15,
                       filter_size : int=2**18):
   
    ''' Normalize staining appearence of H&E stained images
        
    Input:
        img: RGB input image
        io: (optional) transmitted light intensity
        
    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image
    '''
             
    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])
        
    maxCRef = np.array([1.9705, 1.0308])
    
    # define height and width of image
    h, w, c = img.shape
    
    # reshape image
    img = img.reshape((-1, 3))
    
    # calculate optical density
    # OD = -np.log((img.astype(np.float64) + 1) / io)
    OD = -np.log((img.astype(np.float64) + 1) / io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # special case if too few pixels are left
    if ODhat.shape[0] < filter_size:
        return None

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
    # eigvals, eigvecs = np.linalg.eig(np.cov(ODhat.T))

    #eigvecs *= -1
    
    #project on the plane spanned by the eigenvectors corresponding to the two 
    # largest eigenvalues    
    That = ODhat.dot(eigvecs[:,1:3])
    
    phi = np.arctan2(That[:,1],That[:,0])
    
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    
    v_min = eigvecs[:,1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:,1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)
    
    # a heuristic to make the vector corresponding to hematoxylin first and the 
    # one corresponding to eosin second
    if v_min[0] > v_max[0]:
        HE = np.array((v_min[:,0], v_max[:,0])).T
    else:
        HE = np.array((v_max[:,0], v_min[:,0])).T
    
    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T
    
    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    
    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0,:], 99), np.percentile(C[1,:],99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])
    
    # recreate the image using reference mixing matrix
    Inorm = np.multiply(io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)  
    
    # unmix hematoxylin and eosin
    H = np.multiply(io, np.exp(np.expand_dims(-HERef[:,0], axis=1).dot(np.expand_dims(C2[0,:], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)
    
    E = np.multiply(io, np.exp(np.expand_dims(-HERef[:,1], axis=1).dot(np.expand_dims(C2[1,:], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)
    
    if save_path is not None:
        Image.fromarray(Inorm).save(os.path.join(save_path, name))  

    return Inorm


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./')
    parser.add_argument('--output_path', type=str, default='./normalized')
    parser.add_argument('--folders', type=bool, default=False)

    args = parser.parse_args()

    if args.folders:
        normalize_color_folders(args.input_path, args.output_path)
    else:
        normalize_color(args.input_path, args.output_path)
