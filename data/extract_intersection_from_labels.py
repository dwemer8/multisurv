import pandas as pd

DATASET_LOCATION = '../DRIM/data/files/full_dataset.csv'
LABELS_LOCATION  = 'data/labels_mtcp.tsv'

dataset = pd.read_csv(DATASET_LOCATION)
labels = pd.read_csv(LABELS_LOCATION, sep="\t")

mask = ~dataset['RNA'].isna() & ~dataset['DNAm'].isna() & ~dataset['MRI'].isna() & ~dataset['WSI'].isna()
dataset_intersection = dataset[mask]
# print(len(dataset_intersection))

labels_intersection = pd.merge(
    how="inner",
    on="submitter_id",
    left=labels,
    right=dataset_intersection[["submitter_id"]]
)

# print(len(labels_intersection), len(labels_intersection[labels_intersection['group'] == 'test']))
labels_intersection.to_csv("data/labels_mtcp_intersection.tsv", sep="\t", index=False)