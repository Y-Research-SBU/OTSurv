"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import json
import logging
import os
import sys
from os.path import join as j_

import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.file_utils import save_pkl
from utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, get_current_time, extract_patching_info, setup_logging)
from wsi_datasets import WSI_OTSurv_Dataset

from .engine import train


def collate_fn_list(batch):
    return batch  # return list directly, no padding

def build_datasets(csv_splits, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    
    for k in csv_splits.keys():  # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSI_OTSurv_Dataset(df=df['histo'], **dataset_kwargs)

        # use custom collate_fn (enable when data is not of equal length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=dataset_kwargs.get('shuffle', False),
                                num_workers=num_workers,
                                collate_fn=collate_fn_list)  # key modification

        dataset_splits[k] = dataloader
        logging.info(f'split: {k}, n: {len(dataset)}')

        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()

    return dataset_splits

def main(args):
    # Setup logging
    log_file = j_(args.results_dir, 'training.log')
    setup_logging(log_file)
    
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    censorship_col = args.target_col.split('_')[0] + '_censorship'
    
    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False
                      )

    all_results, all_dumps = {}, {}

    seed_torch(args.seed, args.device)
    csv_splits = read_splits(args)
    logging.info('successfully read splits for: ' + str(list(csv_splits.keys())))
    dataset_splits = build_datasets(csv_splits, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args)

    # Save results
    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=16)

### misc ###
parser.add_argument('--print_every', default=1,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=1, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=10,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

# model args ###
parser.add_argument('--model_type', default='otsurv', help='model type')

# Prototype related
parser.add_argument('--in_dim', default=1024, type=int,
                    help='dim of input features')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
parser.add_argument('--loss_fn', type=str, default='cox', choices=['nll', 'cox', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0.5,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='BLCA_survival')
parser.add_argument('--target_col', type=str, default='dss_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source') # /your/data/path/feats_h5
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')  # e.g. src/splits/survival/TCGA_BLCA_overall_survival_k=0
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')

# device args ###
parser.add_argument('--device', type=str, default='cuda:0',
                    help='device to run the model on')

args = parser.parse_args()

if __name__ == "__main__":

    logging.info(f'task: {args.task}')
    args.split_dir = j_('splits', args.split_dir)
    logging.info(f'split_dir: {args.split_dir}')
    split_num = args.split_dir.split('/')[2].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source: 
        ### assert data source exists + extract feature name ###
        logging.info(f'data source: {src}')
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        ### parse patching info ###
        feat_name = os.path.basename(src)
        mag, patch_size = extract_patching_info(os.path.dirname(src))
        if (mag < 0 or patch_size < 0):
            raise ValueError(f"invalid patching info parsed for {src}")
        check_params_same.append([feat_name, mag, patch_size])

        #### parse model name ####
        parsed = parse_model_name(feat_name) 
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})
    
    try:
        check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
        assert check_params_same.drop(['feats_name'],axis=1).drop_duplicates().shape[0] == 1
        logging.info("All data sources have the same feature extraction parameters.")
    except:
        logging.info("Data sources do not share the same feature extraction parameters. Exiting...")
        sys.exit()
        
    ### Updated parsed mdoel parameters in args.Namespace ###
    for key, val in parsed.items():
        setattr(args, key, val)
    
    ### setup results dir ###
    if args.exp_code is None:
        exp_code = f"{args.split_name_clean}::{args.model_type}::{feat_name}"
    else:
        pass
    
    args.results_dir = j_(args.results_dir, 
                          args.task, 
                          f'k={args.split_k}', 
                          str(exp_code), 
                          f"Time::{get_current_time()}")

    os.makedirs(args.results_dir, exist_ok=True)

    logging.info("\n################### Settings ###################")
    for key, val in vars(args).items():
        logging.info("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    logging.info("FINISHED!\n\n\n")