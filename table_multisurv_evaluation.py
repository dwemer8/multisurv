# %% [markdown]
# <a id='Top'></a>
# 
# # MultiSurv results<a class='tocSkip'></a>
# 
# Evaluation metric results for MultiSurv.

# %%
# %load_ext autoreload
# %autoreload 2

# %load_ext watermark

import sys
import os

import ipywidgets as widgets
import numpy as np
import pandas as pd
import torch

# Make modules in "src" dir visible
project_dir = os.path.split(os.getcwd())[0]
# if project_dir not in sys.path:
#     sys.path.append(os.path.join(project_dir, 'src'))
sys.path.append(os.path.join(project_dir, 'src'))

import dataset
from model import Model
import utils

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#DataLoader" data-toc-modified-id="DataLoader-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><code>DataLoader</code></a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Load-weights" data-toc-modified-id="Load-weights-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Load weights</a></span></li></ul></li><li><span><a href="#Evaluate" data-toc-modified-id="Evaluate-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Evaluate</a></span><ul class="toc-item"><li><span><a href="#Write-to-results-table" data-toc-modified-id="Write-to-results-table-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Write to results table</a></span></li><li><span><a href="#Check-results-on-all-datasets" data-toc-modified-id="Check-results-on-all-datasets-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Check results on all datasets</a></span></li></ul></li></ul></div>

# %%
DATA = '/mnt/data/d.kornilov/TCGA/processed_GBM_LGG'
MODELS = '/home/d.kornilov/work/multisurv/outputs/models_gbm_lgg'
LABELS_FILE = '/home/d.kornilov/work/multisurv/data/labels_gbm_lgg.tsv'
# MODEL = 'clinical_lr0.005_epoch71_concord0.83.pth'
# MODEL = 'clinical_mRNA_lr0.005_epoch17_concord0.85.pth'
MODEL = 'mRNA_lr0.005_epoch11_concord0.91.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# # `DataLoader`

# %%
modalities = widgets.SelectMultiple(
    options=['clinical', 'mRNA', 'DNAm', 'miRNA', 'CNV', 'wsi'],
    index=[1],
    rows=6,
    description='Input data',
    disabled=False
)
# display(modalities)

# %%
# 20-cancer type subset (to compare to Cheerla and Gevaert 2019)

# cancers = ['BLCA', 'BRCA', 'CESC', 'COAD', 'READ', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML',
#            'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

# labels = pd.read_csv('../data/labels.tsv', sep='\t')
# print(labels.head(3))

# # List of patients to exclude: patients with cancers that are not in the subset
# exclude_cancers = list(labels.loc[~labels['project_id'].isin(cancers), 'submitter_id'])
# print(len(exclude_cancers))

# %%
dataloaders = utils.get_dataloaders(
    data_location=DATA,
    labels_file=LABELS_FILE,
    modalities=modalities.value,
    wsi_patch_size=299,
    n_wsi_patches=5,
#     exclude_patients=exclude_cancers,
    num_workers=8,
    drop_last=False 
)

for split, dataloader in dataloaders.items():
    print(f"{split} dataloader: {len(dataloader)}")

# %% [markdown]
# # Model

# %%
# prediction_intervals = torch.arange(0., 365 * 21, 365)
# prediction_intervals = torch.arange(0., 365 * 11, 365)
# prediction_intervals = torch.arange(0., 365 * 10.1, 365 / 2)
# prediction_intervals = torch.arange(0., 365 * 6, 365)
# prediction_intervals = torch.arange(0., 365 * 5.1, 365 / 2)

# %%
# labels = [(t, e) for t, e in dataloaders['train'].dataset.label_map.values()]
# durations = [t for t, _ in labels]
# events = [e for _, e in labels]

# prediction_intervals = utils.discretize_time_by_duration_quantiles(durations, events, 20)
# prediction_intervals = torch.from_numpy(prediction_intervals)

# %%
multisurv = Model(dataloaders=dataloaders,
#                   output_intervals=prediction_intervals,
                  device=device)

# %%
print('Output intervals (in years):')
print(multisurv.output_intervals / 365)

# %% [markdown]
# ## Load weights

# %%
# !ls -1 /home/d.kornilov/work/multisurv/outputs/models

# %%
# !ls -1 /mnt/dataA/multisurv_models/wsi*

# %%
# weights =  'clinical_mRNA_lr0.005_discretized_by_duration_quantiles_epoch50_acc0.79.pth'
# weights =  'clinical_mRNA_lr0.005_20_1year_intervals_epoch35_acc0.80.pth'
# weights =  'clinical_mRNA_lr0.005_10_1year_intervals_epoch37_acc0.80.pth'
# weights =  'clinical_mRNA_lr0.005_20_half-year_intervals_epoch47_acc0.80.pth'
# weights =  'clinical_mRNA_lr0.005_5_1year_intervals_epoch39_acc0.80.pth'
# weights =  'clinical_mRNA_lr0.005_10_half-year_intervals_epoch35_acc0.79.pth'

# multisurv.load_weights(os.path.join(MODELS, weights))

# %%
# 20 cancers
# weights = 'clinical_DNAm_lr0.005_20_cancers_epoch39_concord0.79.pth'
# multisurv.load_weights(os.path.join(MODELS, weights))

# %%
# Best model
multisurv.load_weights(os.path.join(MODELS, MODEL))

# %%
for modality in modalities.value:
    print(modality)

# %% [markdown]
# # Evaluate

# %%
# def get_interval_midpoints(intervals):    
#     return intervals[1:] - np.diff(intervals) / 2

# %%
# %%time

# Using custom output intervals

# performance = utils.Evaluation(model=multisurv, dataset=dataloaders['test'].dataset, device=device)

# prediction_time_points = get_interval_midpoints(prediction_intervals)
# performance.run_bootstrap(time_points=prediction_time_points)
# print()

# %%
# %%time

# Using MultiSurv's default output intervals
performance = utils.Evaluation(model=multisurv, dataset=dataloaders['test'].dataset,
                               device=device)
performance.run_bootstrap()
print()

# %%
data_modalities = ' + '.join(modalities.value) if len(modalities.value) > 1 else modalities.value[0]
print(f'>> {data_modalities} <<')
print()
performance.show_results()

# %%
print("empirical")
data_modalities = ' + '.join(modalities.value) if len(modalities.value) > 1 else modalities.value[0]
print(f'>> {data_modalities} <<')
print()
performance.show_results(method='empirical')

# %% [markdown]
# ## Write to results table

# # %%
# results = utils.ResultTable()

# # %%
# data_modalities = ' + '.join(modalities.value) if len(modalities.value) > 1 else modalities.value[0]

# results.write_result_dict(result_dict=performance.format_results(),
#                           algorithm='MultiSurv',
#                           data_modality=data_modalities)
# results.table

# # %% [markdown]
# # ## Check results on all datasets

# # %%
# %%time

# print('~' * 23)
# print('     RESULT CHECK')
# print('~' * 23)
# check_results = {'train': None, 'val': None, 'test': None}

# for group in check_results.keys():
#     print(f'~ {group} ~')
#     performance = utils.Evaluation(model=multisurv, dataset=dataloaders[group].dataset, device=device)
#     performance.compute_metrics()
#     performance.show_results()
#     print()

# # %% [markdown]
# # # Watermark<a class='tocSkip'></a>

# # %%
# %watermark --iversions
# %watermark -v
# print()
# %watermark -u -n

# # %% [markdown]
# # [Top of the page](#Top)


