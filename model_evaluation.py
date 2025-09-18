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
import numpy as np
import torch

if torch.cuda.is_available():
    print('>>> PyTorch detected CUDA <<<')

# Make modules in "src" dir visible
# if os.getcwd() not in sys.path:
#     sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'src'))

import utils
from model import Model
# import wandb
from torch.utils.tensorboard import SummaryWriter

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#DataLoader" data-toc-modified-id="DataLoader-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><code>DataLoader</code></a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Different-intervals" data-toc-modified-id="Different-intervals-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Different intervals</a></span><ul class="toc-item"><li><span><a href="#Equidistant-times" data-toc-modified-id="Equidistant-times-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Equidistant times</a></span></li><li><span><a href="#By-duration-quantiles" data-toc-modified-id="By-duration-quantiles-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>By duration quantiles</a></span></li></ul></li><li><span><a href="#Pick-learning-rate" data-toc-modified-id="Pick-learning-rate-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pick learning rate</a></span></li><li><span><a href="#Fit" data-toc-modified-id="Fit-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Fit</a></span><ul class="toc-item"><li><span><a href="#Save-model-weights" data-toc-modified-id="Save-model-weights-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Save model weights</a></span></li></ul></li><li><span><a href="#Check-validation-metrics" data-toc-modified-id="Check-validation-metrics-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Check validation metrics</a></span></li></ul></li></ul></div>

# %%
#/home/d.kornilov/work/multisurv/outputs/models_mtcp_intersection_all_GBM,LGG/clinical_mRNA_lr0.005_fold_4_epoch43_concord0.90.pth
CANCER_TYPE = 'GBM,LGG'
DATA = f'/mnt/data/d.kornilov/TCGA/processed_mtcp_intersection_all/GBM_LGG' #{CANCER_TYPE}' #'/mnt/data/d.kornilov/TCGA/processed_mtcp'
MODELS = f'/home/d.kornilov/work/multisurv/outputs/models_mtcp_intersection_all_{CANCER_TYPE}'
LABELS_FILE = f'/home/d.kornilov/work/multisurv/data/labels_mtcp_intersection_all_{CANCER_TYPE}.tsv' #'/home/d.kornilov/work/multisurv/data/labels_mtcp.tsv'
LOG_DIR = f'.training_logs_mtcp_intersection_all_{CANCER_TYPE}'
CLINICAL_DATASET = f'/home/d.kornilov/work/multisurv/data/clinical_data_mtcp_intersection_all_preprocessed_{CANCER_TYPE}.tsv' #data/clinical_data_mtcp_preprocessed.tsv'

NUM_OF_CATEGORICAL_CLINICAL_FEATURES = 9 #VERY IMPORTANT!!!
N_FOLDS = 5
NUM_EPOCHS = 75
CLINICAL_EMBEDDINGS_DIM = [
    (2, 1), (2, 1), (6, 3), (2, 1), (3, 2), (3, 2), (3, 2), (3, 2), (1, 1) #gbm, lgg VERY IMPORTANT!!!
    # (1, 1), (1, 1), (8, 4), (2, 1), (2, 1), (2, 1), (3, 2), (3, 2), (1, 1) #ucec
    # (1, 1), (2, 1), (6, 3), (3, 2), (2, 1), (3, 2), (3, 2), (3, 2), (10, 5) # luad
    # (1, 1), (2, 1), (6, 3), (2, 1), (2, 1), (2, 1), (3, 2), (3, 2), (5, 3) # blca
    # (1, 1), (2, 1), (7, 4), (3, 2), (3, 2), (2, 1), (3, 2), (3, 2), (13, 7) #brca
]

RUN_NAME_PREFIX = 'eval_'
WEIGHTS_PREFIX = 'clinical_mRNA_lr0.005'


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
weights = sorted(weights, key=lambda x: int(os.path.split(x)[1].split("_")[4])) #sort by epoch number
print(weights)
assert len(weights) == N_FOLDS, f'Number of weights {len(weights)} is not equal to number of folds {N_FOLDS}'

# wandb.init(
#     name=f"{RUN_NAME_PREFIX}{'_'.join(data_modalities.value)}",
#     config=fit_args,
#     entity="dmitriykornilov_team",
#     project="MultiSurv"
# )

all_val_metrics = []
all_test_metrics = []

writer = SummaryWriter(log_dir = os.path.join(
    LOG_DIR, 
    "_".join(data_modalities.value) + f"_lr{picked_lr}" + "_summary"
))

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
                                        clinical_categorical_num=NUM_OF_CATEGORICAL_CLINICAL_FEATURES,
                                        clinical_dataset_file=CLINICAL_DATASET,
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
                    device=device,
                    clinical_embedding_dims = CLINICAL_EMBEDDINGS_DIM)
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
                                        clinical_categorical_num=NUM_OF_CATEGORICAL_CLINICAL_FEATURES,
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

        # wandb.summary[f"{phase}.fold_{fold}"] = metrics
        for metric, value in metrics.items():
            if isinstance(value, list):
                value = np.mean(value)
            writer.add_scalar(f"{phase}/fold_{fold}/{metric}", value, 0)

    # %%
    # wandb.finish()

# wandb.summary["val"] = utils.agg_fold_metrics(all_val_metrics)
# wandb.summary["test"] = utils.agg_fold_metrics(all_test_metrics)
# wandb.finish()

# Replace wandb.summary["val"] = utils.agg_fold_metrics(all_val_metrics)
val_metrics_summary = utils.agg_fold_metrics(all_val_metrics)
for metric_name, stats in val_metrics_summary.items():
    for stat_name, value in stats.items():
        writer.add_scalar(f"val/{metric_name}/{stat_name}", value, 0)

# Replace wandb.summary["test"] = utils.agg_fold_metrics(all_test_metrics)
test_metrics_summary = utils.agg_fold_metrics(all_test_metrics)
for metric_name, stats in test_metrics_summary.items():
    for stat_name, value in stats.items():
        writer.add_scalar(f"test/{metric_name}/{stat_name}", value, 0)

writer.close()

