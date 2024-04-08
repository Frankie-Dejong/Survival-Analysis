import torch
import yaml
import numpy as np
from utils import map_weibull_param
from Models import CancerPredictModel
from CancerDataset import CancerDataset
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter, WeibullFitter
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

import sys
sys.path.append('./code/')

cfg = yaml.safe_load(open("configuration/tcga_cfg.yml", "r"))
device = cfg["device"]
test_set = CancerDataset(feat_path=cfg['feature_path'], anno_dir=cfg["anno_path"], batch_size=1, loader='numpy', phase='eval')
state_dict = torch.load('./ckpt.pt')  # TODO: change this
model = CancerPredictModel()
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

real_list = []
pred_list = []
vs_list = []

stat_data = []

for bags, real_days, vital_status, ids in test_set:
    for i in range(test_set.batch_size):
        bag = torch.Tensor(bags[i]).to(device)
        real_day = torch.Tensor(real_days[i]).item()
        vs = vital_status[i].item()
        alpha, beta = model(bag)

        per = weibull_min.ppf(0.5, alpha.detach().cpu(), beta.detach().cpu())
        
        real_list.append(real_day)
        pred_list.append(per)
        vs_list.append(vs)
        
        stat_data.append({'alpha': alpha.detach().cpu().item(), 
                          'beta': beta.detach().cpu().item(), 
                          'sample id': ids[i], 
                          'observe day': real_day * test_set.rms , 
                          'vital status': vs, 
                          'pred death day': per.item() * test_set.rms})
        
import json
with open('./stat_res.json', mode='w') as f:
    json.dump(stat_data, f, indent=4)

c_index = concordance_index(real_list, pred_list, vs_list)
print(f"c-index: {c_index}")

median_day = np.median(pred_list)
std_day = np.std(pred_list)
mean_real = np.median(real_list)
std_real = np.std(real_list)
print(f"pred median day: {median_day}")
print(f'pre std: {std_day}')
print(f'gt median day: {mean_real}')
print(f'gt std: {std_real}')

T_big=[]
T_small=[]
E_big=[]
E_small=[]

for pred, real, vs in zip(pred_list, real_list, vs_list):
    if pred > median_day:
        T_big.append(real)
        E_big.append(vs)
    else:
        T_small.append(real)
        E_small.append(vs)

logrank_results = logrank_test(T_big, T_small, event_observed_A=E_big, event_observed_B=E_small)
print(f"Log-rank: \n{logrank_results}")

kmf = KaplanMeierFitter()
# kmf = WeibullFitter()
ax = plt.subplot(111)

kmf.fit(T_small, event_observed=E_small, label="Shorter-term survivors")
kmf.plot(show_censors=True,ax=ax)

kmf.fit(T_big, event_observed=E_big, label="Longer-term survivors")
kmf.plot(show_censors=True,ax=ax)

# save the T, E
plt.savefig('kmf.png')
