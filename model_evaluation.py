# %% [markdown]
# <a id='Top'></a>
# 
# # Multisurv model training<a class='tocSkip'></a>
# 
# Train MultiSurv models with different combinations of input data modalities.

# %%
# %load_ext autoreload
# %autoreload 2

# %load_ext watermark

import sys
import os

import ipywidgets as widgets
import pandas as pd
import torch

if torch.cuda.is_available():
    print('>>> PyTorch detected CUDA <<<')

# Make modules in "src" dir visible
# if os.getcwd() not in sys.path:
#     sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src'))

import utils
from model import Model
import wandb

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#DataLoader" data-toc-modified-id="DataLoader-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><code>DataLoader</code></a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Different-intervals" data-toc-modified-id="Different-intervals-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Different intervals</a></span><ul class="toc-item"><li><span><a href="#Equidistant-times" data-toc-modified-id="Equidistant-times-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Equidistant times</a></span></li><li><span><a href="#By-duration-quantiles" data-toc-modified-id="By-duration-quantiles-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>By duration quantiles</a></span></li></ul></li><li><span><a href="#Pick-learning-rate" data-toc-modified-id="Pick-learning-rate-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pick learning rate</a></span></li><li><span><a href="#Fit" data-toc-modified-id="Fit-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Fit</a></span><ul class="toc-item"><li><span><a href="#Save-model-weights" data-toc-modified-id="Save-model-weights-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Save model weights</a></span></li></ul></li><li><span><a href="#Check-validation-metrics" data-toc-modified-id="Check-validation-metrics-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Check validation metrics</a></span></li></ul></li></ul></div>

# %%
# DATA = utils.INPUT_DATA_DIR
# MODELS = utils.TRAINED_MODEL_DIR
DATA = '/mnt/data/d.kornilov/TCGA/processed_mtcp'
MODELS = '/home/d.kornilov/work/multisurv/outputs/models_mtcp'
LABELS_FILE = '/home/d.kornilov/work/multisurv/data/labels_mtcp_intersection.tsv'
LOG_DIR = '.training_logs_mtcp'
CLINICAL_DATASET = 'data/clinical_data_mtcp_preprocessed.tsv'
RUN_NAME_PREFIX = 'eval_intersection_'
WEIGHTS_PREFIX = 'clinical_mRNA_lr0.005'
N_FOLDS = 5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
print(device)

# %%
data_modalities = widgets.SelectMultiple(
    options=['clinical', 'mRNA', 'DNAm', 'miRNA', 'CNV', 'wsi'],
    index=[0, 1],
    rows=6,
    description='Input data',
    disabled=False
)
print(data_modalities)

picked_lr = 5e-3
fit_args = {
    'lr': picked_lr,
    'num_epochs': 75,
    'info_freq': 1,
#     'info_freq': None,
#     'lr_factor': 0.25,
#     'scheduler_patience': 5,
    'lr_factor': 0.5,
    'scheduler_patience': 10,
}

weights = []
for name in os.listdir(MODELS):
    if name.startswith(WEIGHTS_PREFIX):
        weights.append(os.path.join(MODELS, name))
weights = sorted(weights, key=lambda x: int(x.split("_")[5]))
print(weights)
assert len(weights) == N_FOLDS, f'Number of weights {len(weights)} is not equal to number of folds {N_FOLDS}'

wandb.init(
    name=f"{RUN_NAME_PREFIX}{'_'.join(data_modalities.value)}",
    config=fit_args,
    entity="dmitriykornilov_team",
    project="MultiSurv"
)

all_val_metrics = []
all_test_metrics = []

for fold in range(N_FOLDS):
    print(f"Fold {fold}")

    dataloaders = utils.get_dataloaders(data_location=DATA,
                                        labels_file=LABELS_FILE,
                                        modalities=data_modalities.value,
                                        wsi_patch_size=299,
                                        n_wsi_patches=5,
    #                                     batch_size=20,
    #                                     batch_size=64,
    #                                     batch_size=32,
    #                                     exclude_patients=exclude_cancers,
                                        num_workers=8,
                                        drop_last=False,
                                        fold=fold,
                                        clinical_categorical_num=8,
                                        clinical_dataset_file=CLINICAL_DATASET
                                    )

    for split, dataloader in dataloaders.items():
        print(f"{split} dataloader: {len(dataloader)}")


    # %%
    cuts = torch.tensor([ 0.        ,  0.92617159,  1.85234319,  2.77851478,  3.70468637,
        4.63085797,  5.55702956,  6.48320115,  7.40937275,  8.33554434,
        9.26171593, 10.18788753, 11.11405912, 12.04023071, 12.96640231,
       13.8925739 , 14.81874549, 15.74491709, 16.67108868, 17.59726027], dtype=torch.float32) #from mtcp
    cuts *= 365
    multisurv = Model(dataloaders=dataloaders,
    #                   fusion_method='attention',
    #                   output_intervals=interval_cuts,
                    output_intervals=cuts, 
                    device=device)
    print('Output intervals (in years):')
    print(multisurv.output_intervals / 365)
    print(multisurv.model)
    multisurv.load_weights(weights[fold])

    # %%
    dataloaders = utils.get_dataloaders(data_location=DATA,
                                        labels_file=LABELS_FILE,
                                        modalities=data_modalities.value,
                                        wsi_patch_size=299,
                                        n_wsi_patches=5,
    #                                     exclude_patients=exclude_cancers,
                                        return_patient_id=True,
                                        fold=fold,
                                        clinical_categorical_num=8,
                                        clinical_dataset_file=CLINICAL_DATASET
                                    )

    # %%
    for phase in ["val", "test"]:
        performance = utils.Evaluation(
            model=multisurv, dataset=dataloaders[phase].dataset,
            device=device)
        performance.compute_metrics()
        performance.show_results()

        metrics = {
            "c-index": performance.c_index,
            "ctd": performance.c_index_td,
            "ibs": performance.ibs,
            "inbll": performance.inbll
        }

        if phase == "val":
            all_val_metrics.append(metrics)
        elif phase == "test":
            all_test_metrics.append(metrics)

        wandb.summary[f"{phase}.fold_{fold}"] = metrics

    # %%
    # wandb.finish()

wandb.summary["val"] = utils.agg_fold_metrics(all_val_metrics)
wandb.summary["test"] = utils.agg_fold_metrics(all_test_metrics)
wandb.finish()


