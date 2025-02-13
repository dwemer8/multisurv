#!/usr/bin/env python

"""
    Compute variance
    ~~~~~~~~~~~~~~~~
    Compute gene value variance across all samples and save to file.
"""

import time
import os
from io import StringIO

import click
import requests
import json
import pandas as pd

from tqdm import tqdm

@click.command()
@click.option('-i', '--input_file_dir', default='/mnt/data/m.zubrikhina/TCGA/GBMLGG/raw_RNA',
              type=click.Path(exists=True),
              help='Directory containing input files.')
@click.option('-s', '--chunk_size', default=10000, type=int,
              help='Chunk size (number of lines, i.e. genes) to process at a' +
              ' time. Default: 10000')
@click.option('-o', '--output_file', default='/mnt/data/m.zubrikhina/TCGA/GBMLGG/raw_RNA_variance.tsv', type=click.Path(),
              help='Path to output file. Default: None')
@click.option('-t', '--n_samples', default=None, type=int,
              help='Number of samples to run (allows running a subset for' +
              ' testing). Default: None')
@click.option('-i', '--labels', default='/home/d.kornilov/work/multisurv/data/labels_gbm_lgg.tsv',
              type=click.Path(exists=True),
              help='Labels .tsv')
@click.option('-i', '--mrna_files_path', default='/home/d.kornilov/work/multisurv/data/mRNA_files.csv',
              type=click.Path(exists=True),
              help='mRNA_files_path .csv')

@click.version_option(version='0.0.1', prog_name='Compute gene variance')
def main(input_file_dir, chunk_size, output_file, n_samples, labels, mrna_files_path):
    """Run variance calculation pipeline."""
    start = time.time()
    print_header()

    print('Downloading metadata from GDC database...')

    
    if os.path.exists(mrna_files_path):
        print(f"mRNA_files was loaded from {mrna_files_path}")
        mRNA_files = pd.read_csv(mrna_files_path)
    else:
        print(f"mRNA_files was requested")
        mRNA_files = request_file_info()

    mRNA_files = mRNA_files[
        mRNA_files['cases.0.project.project_id'].str.startswith('TCGA')]
    # mRNA_files = mRNA_files[
    #     mRNA_files['file_name'].str.endswith('FPKM-UQ.txt.gz')]
    mRNA_files = mRNA_files[
        mRNA_files['file_name'].str.endswith('rna_seq.augmented_star_gene_counts.tsv')]
    # mRNA_files = mRNA_files[
    #     mRNA_files['cases.0.samples.0.sample_type'] == 'Primary Tumor']

    # When there is more than one file for a single patient just keep the first 
    # (this is assuming they are just replicates and all similar)
    mRNA_files = mRNA_files[~mRNA_files.duplicated(
        subset=['cases.0.submitter_id'], keep='first')]

    file_map = make_patient_file_map(
        mRNA_files, base_dir=input_file_dir)
    
    labels = pd.read_csv(labels, sep='\t')
    file_map = {k: file_map[k] for k in file_map if k in list(labels['submitter_id'])}
    print(f"Only {len(file_map)} files with labels are left")
    
    file_map = leave_only_existing(file_map)
    print(f"Only existing {len(file_map)} files are left")

    # Subset
    if n_samples is not None:
        print('Keeping only subset of {n_samples} samples...')
        file_map = {key: file_map[key] for i, key in enumerate(file_map)
                    if i < n_samples}

    eg_file = file_map[list(file_map.keys())[0]]
    total_n_lines = len(read_star_counts_tsv(eg_file))
    print(f"Total n lines: {total_n_lines}")
    # total_n_lines = len(list(pd.read_csv(
    #     eg_file, sep='\t', header=None, index_col=0, names=['count']).index))

    print('Process gene chunks:')
    variance_table = pd.DataFrame()

    for bin in [range(i, i + chunk_size)
                for i in range(0, total_n_lines, chunk_size)]:
        print('>>>', bin)
        chunks = load_data_chunk(file_map, total_n_lines, list(bin))
        chunks = merge_dfs(chunks)

        if variance_table.empty:
            variance_table = get_var(chunks)
        else:
            variance_table = pd.concat([variance_table, get_var(chunks)])

    # Remove rows with ambiguous reads, not aligned, etc., included by HTSeq
    # (they start with '__')
    variance_table = variance_table[~variance_table.index.str.startswith('__')]
    variance_table.shape

    #-------------------------------------------------------------------------#
    # Save to file
    print()
    print('Saving result to file:')
    print(f'"{output_file}"')
    variance_table.to_csv(output_file, sep='\t', index=True)

    print_footer(start)

def leave_only_existing(file_map):
    file_map_exists = {}

    for patient, file in file_map.items():
        # file_dir = os.sep.join(file.split(os.sep)[:-1])
        # print(file_dir, end="")
        if os.path.exists(file):
            file_map_exists[patient] = file
            # print(" exists")
        else:
            # print(" not exists")
            pass

    return file_map_exists

def print_header():
    print()
    print(' ' * 15, '*' * 25)
    print(' ' * 15, '  COMPUTE GENE VARIANCE')
    print(' ' * 15, '*' * 25)
    print()

def print_footer(start):
    hrs, mins, secs = elapsed_time(start)
    print()
    print(' ' * 18, f'Completed in {hrs}hrs {mins}m {secs}s')
    print(' ' * 15, '*' * 25)

def elapsed_time(start):
    """Compute time since provided start time.

    Parameters
    ----------
    start: float
        Output of time.time().
    Returns
    -------
    Elapsed hours, minutes and seconds (as tuple of int).
    """
    time_elapsed = time.time() - start
    hrs = time_elapsed // 3600
    secs = time_elapsed % 3600
    mins = secs // 60
    secs = secs % 60

    return int(hrs), int(mins), int(secs)

def request_file_info():
    fields = [
        "file_name",
        "cases.submitter_id",
        "cases.samples.sample_type",
        "cases.project.project_id",
        "cases.project.primary_site",
        ]

    fields = ",".join(fields)

    files_endpt = "https://api.gdc.cancer.gov/files"

    filters = {
        "op": "and",
        "content":[
            {
            "op": "in",
            "content":{
                "field": "files.experimental_strategy",
                "value": ['RNA-Seq']
                }
            }
        ]
    }

    params = {
        "filters": filters,
        "fields": fields,
        "format": "TSV",
        "size": "400000"
        }

    response = requests.post(
        files_endpt,
        headers={"Content-Type": "application/json"},
        json=params)

    return pd.read_csv(StringIO(response.content.decode("utf-8")), sep="\t")

def make_patient_file_map(df, base_dir):
    return {row['cases.0.submitter_id']: os.path.join(
        base_dir, row.id, row.file_name)
            for _, row in df.iterrows()}

def read_star_counts_tsv(path):
    df = pd.read_csv(path, sep='\t', header=1)
    df.drop(axis="index", labels=[0, 1, 2, 3], inplace=True) #not data
    df.reset_index(drop=True, inplace=True)
    return df

def load_data_chunk(patient_file_map, total_lines, chunk_lines):
    n = len(patient_file_map)

    rows_to_skip = [x for x in list(range(0, total_lines))
                    if not x in chunk_lines]

    dfs = []
    for patient in tqdm(patient_file_map):
        # print('\r' + f'   Load tables: {str(i + 1)}/{n}')
        # df = pd.read_csv(patient_file_map[patient], sep='\t', header=None,
                        #  index_col=0, names=['FPKM-UQ'], skiprows=rows_to_skip)

        df = read_star_counts_tsv(patient_file_map[patient])
        df.drop(axis="index", labels=rows_to_skip, inplace=True)
        df.set_index('gene_id', inplace=True)
        df = df[['fpkm_uq_unstranded']]
        df.rename({'fpkm_uq_unstranded': patient}, axis='columns', inplace=True)
        df.columns = [patient]
        dfs.append(df)

    print()

    return dfs

def merge_dfs(table_list):
    n = len(table_list)

    final_table = pd.DataFrame()

    for i, table in enumerate(table_list):
        print('\r' + f'   Merge tables: {str(i + 1)}/{n}', end='')
        if final_table.empty:
            final_table = table
        else:
            final_table = final_table.join(table)

    print()

    return final_table

def get_var(table):
    print('   Compute count variance...')

    return table.var(axis=1)


if __name__ == '__main__':
    main()
