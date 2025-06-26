import logging
import os
from os.path import join as j_

import numpy as np
import torch

try:
    from sksurv.metrics import concordance_index_censored
except ImportError:
    print('scikit-survival not installed. Exiting...')
    raise

from mil_models import create_survival_model
from utils.losses import NLLSurvLoss, CoxLoss, SurvRankingLoss
from utils.utils import (EarlyStopping, save_checkpoint, AverageMeter, safe_list_to,
                         get_optim, print_network, get_lr_scheduler)

def train(datasets, args):
    """
    Train for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    assert args.es_metric == 'loss' or args.es_metric == 'c_index'
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    logging.info('Init Model...')

    model = create_survival_model(args)
    model.to(torch.device(args.device))
    
    print_network(model)

    logging.info('Init optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, datasets['train'])

    if args.early_stopping:
        logging.info('Setup EarlyStopping...')
        early_stopper = EarlyStopping(save_dir=args.results_dir,
                                      patience=args.es_patience,
                                      min_stop_epoch=args.es_min_epochs,
                                      better='min' if args.es_metric == 'loss' else 'max',
                                      verbose=True)
    else:
        logging.info('No EarlyStopping...')
        early_stopper = None
    
    #####################
    # The training loop #
    #####################
    for epoch in range(args.max_epochs):
        step_log = {'epoch': epoch, 'samples_seen': (epoch + 1) * len(datasets['train'].dataset)}

        ### Train Loop
        logging.info(f'{"#" * 10} TRAIN Epoch: {epoch} {"#" * 10}')
        train_results = train_loop_survival(model, datasets['train'], optimizer, lr_scheduler, loss_fn,
                                            print_every=args.print_every, accum_steps=args.accum_steps, epoch=epoch)

        ### Validation Loop (Optional)
        if 'val' in datasets.keys():
            logging.info(f'{"#" * 11} VAL Epoch: {epoch} {"#" * 11}')
            val_results, _ = validate_survival(model, datasets['val'], loss_fn,
                                                   print_every=args.print_every, verbose=True)

            ### Check Early Stopping (Optional)
            if epoch > 10 and early_stopper is not None:
                if args.es_metric == 'loss':
                    score = val_results['loss']
                elif args.es_metric == 'c_index':
                    score = val_results['c_index']
                else:
                    raise NotImplementedError
                save_ckpt_kwargs = dict(config=vars(args),
                                        epoch=epoch,
                                        model=model,
                                        score=score,
                                        fname=f's_checkpoint.pth')
                stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
                if stop:
                    break
        logging.info(f'{"#" * (22 + len(f"TRAIN Epoch: {epoch}"))}\n')

    ### End of epoch: Load in the best model (or save the latest model with not early stopping)
    if args.early_stopping:
        model.load_state_dict(torch.load(j_(args.results_dir, f"s_checkpoint.pth"))['model'])
    else:
        torch.save(model.state_dict(), j_(args.results_dir, f"s_checkpoint.pth"))

    ### End of epoch: Evaluate on val and test set
    results, dumps = {}, {}
    for k, loader in datasets.items():
        logging.info(f'End of training. Evaluating on Split {k.upper()}...:')
        if args.model_type == "otsurv_visual":
            return_attn = True 
        else:
            return_attn = False
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        
    return results, dumps


def test(datasets, args):
    """
    Test for a single fold for suvival
    """
    
    writer_dir = args.results_dir
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)
    
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
    elif args.loss_fn == 'rank':
        loss_fn = SurvRankingLoss()

    args.feat_dim = args.in_dim # Patch feature dimension
    logging.info('Init Model...')

    model = create_survival_model(args)
    model.to(torch.device(args.device))
    print_network(model)
    model.load_state_dict(torch.load(j_(args.checkpoint_path))['model'])

    results, dumps = {}, {}
    for k, loader in datasets.items():
        logging.info(f'End of training. Evaluating on Split {k.upper()}...:')
        if args.model_type == "otsurv_visual":
            return_attn = True 
        else:
            return_attn = False
        results[k], dumps[k] = validate_survival(model, loader, loss_fn, print_every=args.print_every,
                                                     dump_results=True, return_attn=return_attn, verbose=False)

        if k == 'train':
            _ = results.pop('train')  # Train results by default are not saved in the summary, but train dumps are
        
    return results, dumps

## SURVIVAL
def train_loop_survival(model, loader, optimizer, lr_scheduler, loss_fn=None, 
                        print_every=50, accum_steps=32, epoch=0):
    
    model.train()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    iterations_per_epoch = len(loader)
    iter_in_epoch = 0    
    for batch_idx, batch in enumerate(loader):
        device = next(model.parameters()).device
        data = [torch.Tensor(batch[i]['img']).to(device) for i in range(len(batch))]
        label = torch.Tensor([batch[i]['label'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        event_time = torch.Tensor([batch[i]['survival_time'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        censorship = torch.Tensor([batch[i]['censorship'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        iterations = epoch * iterations_per_epoch + iter_in_epoch
        data += [iterations_per_epoch]
        data += [iterations]
        out, log_dict = model(data, label=label, censorship=censorship, loss_fn=loss_fn)
        data = data[:-2]

        if out['loss'] is None:
            continue

        # Get loss + backprop
        loss = out['loss']
        loss = loss / accum_steps
        loss.backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # End of iteration survival-specific metrics to calculate / log
        all_risk_scores.append(out['risk'].detach().cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))

        bag_size_meter.update(np.mean([data[i].shape[0] for i in range(len(data))]), n=len(data))

        if ((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            logging.info(msg)
        
        iter_in_epoch += 1

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    results['iterations_per_epoch'] = iterations_per_epoch
    results['iterations'] = iterations

    msg = [f"{k}: {v:.3f}" for k, v in results.items()]
    logging.info("\t".join(msg))

    del all_risk_scores, all_censorships, all_event_times
    torch.cuda.empty_cache()
    
    return results


@torch.no_grad()
def validate_survival(model, loader,
                      loss_fn=None,
                      print_every=50,
                      dump_results=False,
                      recompute_loss_at_end=True,
                      return_attn=False,
                      verbose=1,
                      train_results={'iterations_per_epoch':10, 'iterations':1000}):
    model.eval()
    meters = {'bag_size': AverageMeter()}
    bag_size_meter = meters['bag_size']
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_path_attn = []

    for batch_idx, batch in enumerate(loader):
        device = next(model.parameters()).device
        data = [torch.Tensor(batch[i]['img']).to(device) for i in range(len(batch))]
        label = torch.Tensor([batch[i]['label'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        event_time = torch.Tensor([batch[i]['survival_time'] for i in range(len(batch))]).to(device).unsqueeze(-1)
        censorship = torch.Tensor([batch[i]['censorship'] for i in range(len(batch))]).to(device).unsqueeze(-1)

        data += [train_results['iterations_per_epoch']]
        data += [train_results['iterations']]
        out, log_dict = model(data, label=label, censorship=censorship, loss_fn=loss_fn, return_attn=return_attn)
        data = data[:-2]

        if return_attn:
            all_path_attn.append(out['path_attn'].detach().cpu().numpy())
        # End of iteration survival-specific metrics to calculate / log
        # bag_size_meter.update(data.size(1), n=len(data))
        bag_size_meter.update(np.mean([data[i].shape[0] for i in range(len(data))]), n=len(data))
    
        for key, val in log_dict.items():
            if key not in meters:
                meters[key] = AverageMeter()
            meters[key].update(val, n=len(data))
        all_risk_scores.append(out['risk'].cpu().numpy())
        all_censorships.append(censorship.cpu().numpy())
        all_event_times.append(event_time.cpu().numpy())

        if verbose and (((batch_idx + 1) % print_every == 0) or (batch_idx == len(loader) - 1)):
            msg = [f"avg_{k}: {meter.avg:.4f}" for k, meter in meters.items()]
            msg = f"batch {batch_idx}\t" + "\t".join(msg)
            logging.info(msg)

    # End of epoch survival-specific metrics to calculate / log
    all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
    all_censorships = np.concatenate(all_censorships).squeeze(1)
    all_event_times = np.concatenate(all_event_times).squeeze(1)
    if return_attn:
        all_path_attn = np.vstack(all_path_attn)

    c_index = concordance_index_censored(
        (1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    results = {k: meter.avg for k, meter in meters.items()}
    results.update({'c_index': c_index})

    if recompute_loss_at_end and isinstance(loss_fn, CoxLoss):
        surv_loss_dict = loss_fn(logits=torch.tensor(all_risk_scores).unsqueeze(1),
                                 times=torch.tensor(all_event_times).unsqueeze(1),
                                 censorships=torch.tensor(all_censorships).unsqueeze(1))
        results['surv_loss'] = surv_loss_dict['loss'].item()
        results.update({k: v.item() for k, v in surv_loss_dict.items() if isinstance(v, torch.Tensor)})

    if verbose:
        msg = [f"{k}: {v:.3f}" for k, v in results.items()]
        logging.info("\t".join(msg))

    dumps = {}
    if dump_results:
        dumps['all_risk_scores'] = all_risk_scores
        dumps['all_censorships'] = all_censorships
        dumps['all_event_times'] = all_event_times
        dumps['sample_ids'] = np.array(
            loader.dataset.idx2sample_df['sample_id'])
        if return_attn:
            dumps['all_path_attn'] = all_path_attn
    
    del all_risk_scores, all_censorships, all_event_times
    torch.cuda.empty_cache()

    return results, dumps