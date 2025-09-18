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
# DATA = utils.INPUT_DATA_DIR
# MODELS = utils.TRAINED_MODEL_DIR
DATA = '/mnt/data/d.kornilov/TCGA/processed_mtcp_intersection_all/UCEC' #'/mnt/data/d.kornilov/TCGA/processed_mtcp'
MODELS = '/home/d.kornilov/work/multisurv/outputs/models_mtcp_intersection_all_UCEC'
LABELS_FILE = '/home/d.kornilov/work/multisurv/data/labels_mtcp_intersection_all_UCEC.tsv' #'/home/d.kornilov/work/multisurv/data/labels_mtcp.tsv'
LOG_DIR = '.training_logs_mtcp_intersection_all_UCEC'
CLINICAL_DATASET = '/home/d.kornilov/work/multisurv/data/clinical_data_mtcp_intersection_all_preprocessed_UCEC.tsv' #data/clinical_data_mtcp_preprocessed.tsv'
NUM_OF_CATEGORICAL_CLINICAL_FEATURES = 9 #VERY IMPORTANT!!!
N_FOLDS = 1 #5
NUM_EPOCHS = 2#75
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
print(device)

# %% [markdown]
# # `DataLoader`

# %%
data_modalities = widgets.SelectMultiple(
    options=['clinical', 'mRNA', 'DNAm', 'miRNA', 'CNV', 'wsi'],
    index=[0, 1],
    rows=6,
    description='Input data',
    disabled=False
)
# display(data_modalities)
print(data_modalities)

# %%
#-----------------------------------------------------------------------------#
#                             20-CANCER SUBSET                                #
#                 (to compare to Cheerla and Gevaert 2019)                    #
#-----------------------------------------------------------------------------#

# cancers = ['BLCA', 'BRCA', 'CESC', 'COAD', 'READ',
#            'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML',
#            'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV',
#            'PAAD', 'PRAD', 'SKCM', 'STAD', 'THCA', 'UCEC']

# labels = pd.read_csv('data/labels.tsv', sep='\t')
# print(labels.head(3))

# # List of patients to exclude: patients with cancers that are not in the subset
# exclude_cancers = list(labels.loc[~labels['project_id'].isin(cancers), 'submitter_id'])
# len(exclude_cancers)

# %%

picked_lr = 5e-3
fit_args = {
    'lr': picked_lr,
    'num_epochs': NUM_EPOCHS,
    'info_freq': 1,
#     'info_freq': None,
#     'lr_factor': 0.25,
#     'scheduler_patience': 5,
    'lr_factor': 0.5,
    'scheduler_patience': 10,
}

# wandb.init(
#     name=f"{'_'.join(data_modalities.value)}",
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
                                        clinical_categorical_num=NUM_OF_CATEGORICAL_CLINICAL_FEATURES, #VERY IMPORTANT!!!
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
                    device=device,
                    clinical_embedding_dims = [
                        # (2, 1), (2, 1), (6, 3), (2, 1), (3, 2), (3, 2), (3, 2), (3, 2), (1, 1) #gbm, lgg VERY IMPORTANT!!!
                        (1, 1), (1, 1), (8, 4), (2, 1), (2, 1), (2, 1), (3, 2), (3, 2), (1, 1) #ucec
                    ])

    # %%
    print('Output intervals (in years):')
    print(multisurv.output_intervals / 365)

    # %%
    print(multisurv.model_blocks)

    # %%
    print('Trainable blocks:')
    layer = None

    for name, child in multisurv.model.named_children():
        for name_2, params in child.named_parameters():
            if name is not layer:
                print(f'   {name}: {params.requires_grad}')
            layer = name

    # %%
    print(multisurv.model)

    # %% [markdown]
    # ## Pick learning rate

    # %%
    # %%time

    # multisurv.test_lr_range()
    # print()

    # # %%
    # multisurv.plot_lr_range(trim=1)

    # %% [markdown]
    # ## Fit

    # %%

    run_tag = utils.compose_run_tag(model=multisurv, lr=picked_lr,
                                    dataloaders=dataloaders,
                                    log_dir=LOG_DIR,
                                    suffix=f'_fold_{fold}')

    fit_args.update({
        'log_dir': os.path.join(LOG_DIR, run_tag),
        "run_tag": run_tag,
        "fold": fold
    })

    # wandb.init(
    #     name=f"{run_tag}_{multisurv.fusion_method}_{'_'.join(multisurv.data_modalities)}_n_intervals_{len(multisurv.output_intervals)}_period_{multisurv.output_intervals[1] - multisurv.output_intervals[0]}",
    #     config={
    #         **fit_args,
    #         "self.fusion_method": multisurv.fusion_method,
    #         "output_intervals": multisurv.output_intervals,
    #         "modalities": multisurv.data_modalities
    #     },
    #     entity="dmitriykornilov_team",
    #     project="MultiSurv"
    # )

    multisurv.fit(**fit_args)

    # %% [markdown]
    # ### Save model weights
    # 
    # If desired.

    # %%
    print(multisurv.best_model_weights.keys())

    # %%
    print(multisurv.best_concord_values)
    best_epoch = sorted(multisurv.best_concord_values.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    print("Best epoch:", best_epoch)

    # %%
    print(multisurv.current_concord)

    # %%
    multisurv.save_weights(saved_epoch=best_epoch, prefix=run_tag, weight_dir=MODELS)

    # %% [markdown]
    # ## Check validation metrics

    # %%
    dataloaders = utils.get_dataloaders(data_location=DATA,
                                        labels_file=LABELS_FILE,
                                        modalities=data_modalities.value,
                                        wsi_patch_size=299,
                                        n_wsi_patches=5,
    #                                     exclude_patients=exclude_cancers,
                                        return_patient_id=True,
                                        fold=fold,
                                        clinical_categorical_num=NUM_OF_CATEGORICAL_CLINICAL_FEATURES, #VERY IMPORTANT!!!
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

