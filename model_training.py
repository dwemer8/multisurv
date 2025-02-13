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

# %% [markdown]
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#DataLoader" data-toc-modified-id="DataLoader-1"><span class="toc-item-num">1&nbsp;&nbsp;</span><code>DataLoader</code></a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#Different-intervals" data-toc-modified-id="Different-intervals-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Different intervals</a></span><ul class="toc-item"><li><span><a href="#Equidistant-times" data-toc-modified-id="Equidistant-times-2.1.1"><span class="toc-item-num">2.1.1&nbsp;&nbsp;</span>Equidistant times</a></span></li><li><span><a href="#By-duration-quantiles" data-toc-modified-id="By-duration-quantiles-2.1.2"><span class="toc-item-num">2.1.2&nbsp;&nbsp;</span>By duration quantiles</a></span></li></ul></li><li><span><a href="#Pick-learning-rate" data-toc-modified-id="Pick-learning-rate-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Pick learning rate</a></span></li><li><span><a href="#Fit" data-toc-modified-id="Fit-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Fit</a></span><ul class="toc-item"><li><span><a href="#Save-model-weights" data-toc-modified-id="Save-model-weights-2.3.1"><span class="toc-item-num">2.3.1&nbsp;&nbsp;</span>Save model weights</a></span></li></ul></li><li><span><a href="#Check-validation-metrics" data-toc-modified-id="Check-validation-metrics-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Check validation metrics</a></span></li></ul></li></ul></div>

# %%
# DATA = utils.INPUT_DATA_DIR
# MODELS = utils.TRAINED_MODEL_DIR
DATA = '/mnt/data/d.kornilov/TCGA/processed_GBM_LGG'
MODELS = '/home/d.kornilov/work/multisurv/outputs/models_gbm_lgg'
LABELS_FILE = '/home/d.kornilov/work/multisurv/data/labels_gbm_lgg.tsv'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
print(device)

# %% [markdown]
# # `DataLoader`

# %%
data_modalities = widgets.SelectMultiple(
    options=['clinical', 'mRNA', 'DNAm', 'miRNA', 'CNV', 'wsi'],
    index=[1],
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
                                    drop_last=False
                                   )

for split, dataloader in dataloaders.items():
    print(f"{split} dataloader: {len(dataloader)}")

# %% [markdown]
# # Model

# %% [markdown]
# ## Different intervals
# 
# If trying out different time interval outputs.

# %% [markdown]
# ### Equidistant times

# %%
# interval_cuts = torch.arange(0., 365 * 5.1, 365 / 2)

# %% [markdown]
# ### By duration quantiles

# %%
# labels = [(t, e) for t, e in dataloaders['train'].dataset.label_map.values()]
# durations = [t for t, _ in labels]
# events = [e for _, e in labels]

# interval_cuts = utils.discretize_time_by_duration_quantiles(durations, events, 20)
# interval_cuts = torch.from_numpy(interval_cuts)

# %%
#-----------------------------------------------------------------------------#
#                       PRE-TRAINED UNIMODAL MODELS                           #
#-----------------------------------------------------------------------------#

# unimodal_weigths = {'clinical': 'clinical_lr0.005_epoch49_acc0.78.pth',
#                     'mRNA': 'mRNA_lr0.005_epoch54_acc0.76.pth',
#                     'DNAm': 'DNAm_lr0.005_epoch57_acc0.77.pth',
#                     'miRNA': None,
#                     'CNV': None,
#                     'wsi': None,}

# unimodal_weigths = {k: os.path.join(MODELS, v) if v is not None else None
#                     for k, v in unimodal_weigths.items()}

# multisurv = Model(dataloaders=dataloaders,
#                   unimodal_state_files=unimodal_weigths,
#                   freeze_up_to='aggregator',
#                   device=device)

# %%
#-----------------------------------------------------------------------------#
#                              AUXILIARY LOSS                                 #
#-----------------------------------------------------------------------------#

# cosine_embedding_margin = 1e-5
# auxiliary_criterion = torch.nn.CosineEmbeddingLoss(margin=cosine_embedding_margin)

# multisurv = Model(dataloaders=dataloaders,
#                   auxiliary_criterion=auxiliary_criterion,
#                   device=device)

# %%
multisurv = Model(dataloaders=dataloaders,
#                   fusion_method='attention',
#                   output_intervals=interval_cuts,
                  device=device)

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
picked_lr = 5e-3

run_tag = utils.compose_run_tag(model=multisurv, lr=picked_lr,
                                dataloaders=dataloaders,
                                log_dir='.training_logs_gbm_lgg/',
                                suffix='')

# %%
fit_args = {
    'lr': picked_lr,
    'num_epochs': 75,
    'info_freq': 1,
#     'info_freq': None,
#     'lr_factor': 0.25,
#     'scheduler_patience': 5,
    'lr_factor': 0.5,
    'scheduler_patience': 10,
    'log_dir': os.path.join('.training_logs_gbm_lgg/', run_tag),
}

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
                                   )

# %%
performance = utils.Evaluation(
    model=multisurv, dataset=dataloaders['val'].dataset,
    device=device)
performance.compute_metrics()
performance.show_results()

# %% [markdown]
# # Watermark <a class='tocSkip'></a>

# %%
# %watermark --iversions
# %watermark -v
# print()
# %watermark -u -n

# %% [markdown]
# [Top of the page](#Top)


