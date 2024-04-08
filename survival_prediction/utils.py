import torch
import pandas as pd
import os
from torch.distributions import Weibull, Normal
import torch.nn.functional as F


def map_weibull_param(a, b):
    alpha = torch.exp(a).view(-1, 1)
    beta = torch.nn.functional.softplus(b).view(-1, 1)
    return alpha, beta


def map_norm_param(ab):
    a = ab[0].view(-1, 1)
    b = torch.exp(ab[1]).view(-1, 1)
    return torch.concat((a, b), dim=1).squeeze()


def norm_surv_function(t, mu, sigma):
    norm =  Normal(mu, sigma)
    return norm.cdf(t)

def norm_log_likelihood(y_true, ab_pred):
    events = y_true[1]
    durations = y_true[0]
    mu = ab_pred[0]
    sigma = ab_pred[1]
    
    survival_prob = norm_surv_function(durations, mu, sigma)
    return F.binary_cross_entropy(survival_prob, events)
    

def weibull_survival_function(t, scale, shape):
    weibull_dist = Weibull(scale, shape)
    return weibull_dist.cdf(t).unsqueeze(0)

def weibull_log_likelihood(y_true, alpha, beta):
    events = y_true[1]
    durations = y_true[0]
    scale = alpha
    shape = beta
    survival_prob = weibull_survival_function(durations, scale, shape)
    return F.mse_loss(survival_prob, events)



def weibull_loglik_discrete(y_true, ab_pred):
    y_ = y_true[0]
    u_ = y_true[1]
    a_ = ab_pred[0]
    b_ = ab_pred[1]
    hazard0 = torch.pow((y_ + 1e-35) / a_, b_)
    hazard1 = torch.pow((y_ + 1) / a_, b_)
    return -1 * torch.mean(u_ * torch.log(torch.exp(hazard1 - hazard0) - 1.0) - hazard1)


def weibull_loglik_continuous(y_true, ab_pred):
    y_ = y_true[:, 0]
    u_ = y_true[:, 1]
    a_ = ab_pred[:, 0]
    b_ = ab_pred[:, 1]  # death / live
    ya = (y_ + 1e-35) / a_
    return -1 * torch.mean(u_ * (torch.log(b_) + b_ * torch.log(ya)) - torch.pow(ya, b_))
 

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
