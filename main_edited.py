# ------------------------------------------------------------------------
# Main script to commence baseline experiments on WEAR dataset
# ------------------------------------------------------------------------
# Author: Marius Bock
# Email: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import datetime
import json
import os
from pprint import pprint
import sys
import time


import pandas as pd
import numpy as np
import neptune
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from inertial_baseline.train import run_inertial_network
from utils.torch_utils import fix_random_seed
from utils.os_utils import Logger, load_config
import matplotlib.pyplot as plt
from camera_baseline.actionformer.main import run_actionformer
from camera_baseline.tridet.main import run_tridet

'''
def calculate_softmax_probs(predictions):
    predictions = np.reshape(predictions, (predictions.shape[0], 1))  # Reshape to a 2D array
    exp_preds = np.exp(predictions)  # Exponentiate the predictions
    softmax_probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)  # Calculate softmax probabilities
    return softmax_probs

def extract_labels_from_data(data):
    labels = []
    for subject_data in data.values():
        annotations = subject_data.get('annotations', [])
        for annotation in annotations:
            label = annotation.get('label')
            if label:
                labels.append(label)
    return labels
'''
def main(args):
    if args.neptune:
        run = neptune.init_run(
        project="chetanjain0339/latefusion",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjOGFiOWFlNi1jYjM5LTQxMGYtYTdjNS0wMTIzYWY0ZmNjOGUifQ=="
            
        )
    else:
        run = None 

    config = load_config(args.config)
    config['init_rand_seed'] = args.seed
    config['devices'] = [args.gpu]

    #ts = datetime.datetime.fromtimestamp(int(time.time()))
    ts = datetime.datetime.fromtimestamp(int(time.time()))
    formatted_ts = ts.strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('logs', config['name'] , str(formatted_ts))
    #log_dir = os.path.join('logs', config['name'], str(ts))
    sys.stdout = Logger(os.path.join(log_dir, 'log.txt'))

    '''
    with open(args.data_json) as f:
        data = json.load(f)

    config['labels'] = extract_labels_from_data(data)
    config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))

    '''
    # save the current cfg
    with open(os.path.join(log_dir, 'cfg.txt'), 'w') as fid:
        pprint(config, stream=fid)
        fid.flush()
    
    if args.neptune:
        run['eval_type'] = args.eval_type
        run['config_name'] = args.config
        run['config'].upload(os.path.join(log_dir, 'cfg.txt'))

    rng_generator = fix_random_seed(config['init_rand_seed'], include_cuda=True)    

    all_v_pred = np.array([])
    tridet_pred = np.array([])
    actionformer_pred = np.array([])
    all_v_gt = np.array([])
    tridet_v_gt = np.array([])
    actionformer_v_gt = np.array([])
    camera_t_pred = np.array([])
    inertial_t_pred = np.array([])
    camera_t_gt = np.array([])
    inertial_t_gt = np.array([])
    all_v_mAP = np.empty((0, len(config['dataset']['tiou_thresholds'])))
    '''    
    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]
        
        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        if config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        if config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_preds, v_gt, scores = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        tridet_probs = calculate_softmax_probs(scores)

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print("Tridet Softmax Probabilities", tridet_probs)

        print('\n'.join([block1, block2, block3, block4, block5]))
        
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)


        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')
    
    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))
    

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    
    tridet_pred = np.append(tridet_pred,all_v_pred)
    tridet_v_gt = np.append(tridet_v_gt, all_v_gt)
    print(tridet_pred)
    print("ALL FINISHED 1")


    
    config = load_config('./configs/60_frames_30_stride/actionformer_combined.yaml')

    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]
        
        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        #if config['name'] == 'tridet':
        t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        #if config['name'] == 'tridet':
           # t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        actionformer_probs = calculate_softmax_probs(v_preds)

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print("Actionformer Softmax Probabilities", actionformer_probs)

        print('\n'.join([block1, block2, block3, block4, block5]))
        
        
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)


        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')
    
    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))
    

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    actionformer_pred = np.append(actionformer_pred, all_v_pred)
    actionformer_v_gt = np.append(actionformer_v_gt, all_v_gt)
    print(actionformer_pred)
    print("ALL FINISHED 2")

    '''
    '''
    
    config = load_config('./configs/60_frames_30_stride/actionformer_camera.yaml')
    print("ACTIONFORMER_CAMERA")

    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
        print("Prediction Probabilities:")
        print(prediction_probs)
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    print("ALL FINISHED 1")
    '''
    '''
    config = load_config('./configs/60_frames_30_stride/tridet_combined.yaml')
    print("TRIDET_COMBINED")

    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    print(len(all_v_gt))
    print("ALL FINISHED 2")
    '''
    
    config = load_config('./configs/60_frames_30_stride/tridet_combined.yaml')
    print("TRIDET_CAMERA")

    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_preds, v_gt, prediction_probs = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run, i)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            block1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            block1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        block2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        block3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        block4 = ''
        block4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        block5 = ''
        block5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        block5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        block5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        block5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([block1, block2, block3, block4, block5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')

    print(all_v_gt, all_v_pred)

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    block1 = '\nFINAL AVERAGED RESULTS:'
    block2 = ''
    block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    block2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    block2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    block2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    block2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([block1, block2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    camera_t_gt = np.append(camera_t_gt, all_v_gt)
    camera_t_pred = np.append(camera_t_pred, all_v_pred)
    print("ALL FINISHED 3")
    
    '''
    
    config = load_config('./configs/60_frames_30_stride/tridet_inertial.yaml')
    print("TRIDET_INERTIAL")

    for i, anno_split in enumerate(config['anno_json']):
        with open(anno_split) as f:
            file = json.load(f)
        anno_file = file['database']
        if (not config['dataset']['include_null']) and (config['dataset']['has_null']):
            config['labels'] = ['null'] + list(file['label_dict'])
        else:
            config['labels'] = list(file['label_dict'])
        config['label_dict'] = dict(zip(config['labels'], list(range(len(config['labels'])))))
        train_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Training']
        val_sbjs = [x for x in anno_file if anno_file[x]['subset'] == 'Validation']

        print('Split {} / {}'.format(i + 1, len(config['anno_json'])))
        if args.eval_type == 'split':
            name = 'split_' + str(i)
        elif args.eval_type == 'loso':
            name = 'sbj_' + str(i)
        config['dataset']['json_anno'] = anno_split
        if config['name'] == 'tadtr':
            config['dataset']['json_info'] = config['info_json'][i]

        if config['name'] == 'deepconvlstm' or config['name'] == 'attendanddiscriminate':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_inertial_network(train_sbjs, val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'actionformer':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_actionformer(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        elif config['name'] == 'tridet':
            t_losses, v_losses, v_mAP, v_preds, v_gt = run_tridet(val_sbjs, config, log_dir, args.ckpt_freq, args.resume, rng_generator, run)
        
        # raw results
        conf_mat = confusion_matrix(v_gt, v_preds, normalize='true', labels=range(len(config['labels'])))
        v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
        v_prec = precision_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_rec = recall_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))
        v_f1 = f1_score(v_gt, v_preds, average=None, zero_division=1, labels=range(len(config['labels'])))

        # print to terminal
        if args.eval_type == 'split':
            blocka1 = '\nFINAL RESULTS SPLIT {}'.format(i + 1)
        elif args.eval_type == 'loso':
            blocka1 = '\nFINAL RESULTS SUBJECT {}'.format(i)
        blocka2 = 'TRAINING:\tavg. loss {:.2f}'.format(np.nanmean(t_losses))
        blocka3 = 'VALIDATION:\tavg. loss {:.2f}'.format(np.nanmean(v_losses))
        blocka4 = ''
        blocka4  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(v_mAP) * 100)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], v_mAP):
            block4 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(tiou_mAP*100)
        blocka5 = ''
        blocka5  += '\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
        blocka5  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
        blocka5  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
        blocka5  += ' F1 {:>4.2f} (%)\n'.format(np.nanmean(v_f1) * 100)

        print('\n'.join([blocka1, blocka2, blocka3, blocka4, blocka5]))
                                
        all_v_mAP = np.append(all_v_mAP, v_mAP[None, :], axis=0)
        all_v_gt = np.append(all_v_gt, v_gt)
        all_v_pred = np.append(all_v_pred, v_preds)

        # save raw confusion matrix
        _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
        conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels'])
        conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
        ax.set_title('Confusion Matrix ' + name + ' (raw)')
        plt.savefig(os.path.join(log_dir, name + '_raw.png'))
        plt.close()
        if run is not None:
            run['conf_matrices'].append(_, name=name + '_raw')
    
    print(len(all_v_gt), len(all_v_pred))

    # final raw results across all splits
    conf_mat = confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    # print final results to terminal
    blocka1 = '\nFINAL AVERAGED RESULTS:'
    blocka2 = ''
    blocka2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
        block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    blocka2  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(v_acc) * 100)
    blocka2  += ' Prec {:>4.2f} (%)'.format(np.nanmean(v_prec) * 100)
    blocka2  += ' Rec {:>4.2f} (%)'.format(np.nanmean(v_rec) * 100)
    blocka2  += ' F1 {:>4.2f} (%)'.format(np.nanmean(v_f1) * 100)
    
    print('\n'.join([blocka1, blocka2]))

    # save final raw confusion matrix
    _, ax = plt.subplots(figsize=(15, 15), layout="constrained")
    ax.set_title('Confusion Matrix Total (raw)')
    conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=config['labels']) 
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    plt.savefig(os.path.join(log_dir, 'all_raw.png'))
    plt.close()
    if run is not None:
        run['conf_matrices'].append(File(os.path.join(log_dir, 'all_raw.png')), name='all')

    # submit final values to neptune 
    if run is not None:
        run['final_avg_mAP'] = np.nanmean(all_v_mAP)
        for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
            run['final_mAP@' + str(tiou)] = (np.nanmean(tiou_mAP))
        run['final_accuracy'] = np.nanmean(v_acc)
        run['final_precision'] = (np.nanmean(v_prec))
        run['final_recall'] = (np.nanmean(v_rec))
        run['final_f1'] = (np.nanmean(v_f1))

    inertial_t_gt = np.append(inertial_t_gt, all_v_gt)
    inertial_t_pred = np.append(inertial_t_pred, all_v_pred)
    print("ALL FINISHED 4")
    '''

    all_v_gt = np.concatenate((camera_t_gt, inertial_t_gt, all_v_gt))
    print(len(all_v_gt))

    '''
    def fusion_method(camera_t_pred, inertial_t_pred, batch_size=1000):
        camera_t_pred = np.asarray(camera_t_pred)
        inertial_t_pred = np.asarray(inertial_t_pred)
        #tridet_pred = np.reshape(tridet_pred, actionformer_pred.shape)
        #fused_preds = (tridet_pred[:, np.newaxis] + actionformer_pred) / 2
        fused_preds = []
        for i in range(0, len(camera_t_pred), batch_size):
            camera_t_batch = camera_t_pred[i:i+batch_size]
            inertial_t_batch = inertial_t_pred[i:i+batch_size]

            if camera_t_batch.shape[0] < inertial_t_batch.shape[0]:
                camera_t_batch = np.resize(camera_t_batch, inertial_t_batch.shape)
            elif inertial_t_batch.shape[0] < camera_t_batch.shape[0]:
                inertial_t_batch = np.resize(inertial_t_batch, camera_t_batch.shape)

            fused_batch = (camera_t_batch + inertial_t_batch) / 2
            fused_preds.append(fused_batch)

        fused_preds = np.concatenate(fused_preds)
        return fused_preds
    
    fused_preds = fusion_method(camera_t_pred, inertial_t_pred)
    #print(fused_preds, len(fused_preds))

    
    def calculate_accuracy(camera_t_gt, inertial_t_gt, fused_preds):
        #assert len(tridet_v_gt) == len(fused_preds), "Number of ground truth labels and predicted labels must be the same."
    
        num_samples = len(camera_t_gt)
        num_correct = 0
    
        for i in range(num_samples):
            if camera_t_gt[i] == fused_preds[i]:
                num_correct += 1
    
        camera_t_accuracy = num_correct / num_samples
    
        num_correct = 0
    
        for i in range(num_samples):
            if inertial_t_gt[i] == fused_preds[i]:
                num_correct += 1
    
        inertial_t_accuracy = num_correct / num_samples
    
        fused_accuracy = (camera_t_accuracy + inertial_t_accuracy) / 2
    
        return fused_accuracy
    
    fused_accuracy = calculate_accuracy(camera_t_gt, inertial_t_gt, fused_preds)
    print(fused_accuracy)

    def fusion_ground_truth(camera_t_gt, inertial_t_gt, batch_size=1000):
        camera_t_gt = np.asarray(camera_t_gt)
        inertial_t_gt = np.asarray(inertial_t_gt)

        fused_labels = []
        for i in range(0, len(camera_t_gt), batch_size):
            camera_batch = camera_t_gt[i:i+batch_size]
            inertial_batch = inertial_t_gt[i:i+batch_size]

            if camera_batch.shape[0] < inertial_batch.shape[0]:
                camera_batch = np.resize(camera_batch, inertial_batch.shape)
            elif inertial_batch.shape[0] < camera_batch.shape[0]:
                inertial_batch = np.resize(inertial_batch, camera_batch.shape)

            fused_batch = (camera_batch + inertial_batch) / 2
            fused_labels.append(fused_batch)

        fused_labels = np.concatenate(fused_labels)
        return fused_labels
    

    fused_labels = fusion_ground_truth(camera_t_gt, inertial_t_gt)
    #print(fused_labels, len(fused_labels))

    fused_preds_discrete = np.round(fused_preds).astype(int)
    fused_labels_discrete = np.round(fused_labels).astype(int)

    conf_mat = confusion_matrix(fused_labels_discrete, fused_preds_discrete, normalize='true', labels=range(len(config['labels'])))
    acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    prec = precision_score(fused_labels_discrete, fused_preds_discrete, average=None, zero_division=1, labels=range(len(config['labels'])))
    rec = recall_score(fused_labels_discrete, fused_preds_discrete, average=None, zero_division=1, labels=range(len(config['labels'])))
    f1 = f1_score(fused_labels_discrete, fused_preds_discrete, average=None, zero_division=1, labels=range(len(config['labels'])))
    
    '''
    # print final results to terminal
    #block11 = '\nFINAL FUSED RESULTS:'
    #block12 = ''
    #block2  += '\n\t\tAvg. mAP {:>4.2f} (%) '.format(np.nanmean(all_v_mAP) * 100)
    #for tiou, tiou_mAP in zip(config['dataset']['tiou_thresholds'], all_v_mAP.T):
    #    block2 += 'mAP@' + str(tiou) +  ' {:>4.2f} (%) '.format(np.nanmean(tiou_mAP)*100)
    #block12  += '\n\t\tAcc {:>4.2f} (%)'.format(np.nanmean(acc) * 100)
    #block12  += ' Prec {:>4.2f} (%)'.format(np.nanmean(prec) * 100)
    #block12  += ' Rec {:>4.2f} (%)'.format(np.nanmean(rec) * 100)
    #block12  += ' F1 {:>4.2f} (%)'.format(np.nanmean(f1) * 100)

    #print('\n'.join([block11, block12]))


    #confusion_mat = confusion_matrix(all_v_gt, fused_preds)
    #print(confusion_matrix)
    '''
    # Adjust the lengths of all_v_gt and fused_preds
    min_length = min(len(all_v_gt), len(fused_preds))
    all_v_gt = all_v_gt[:min_length]
    fused_preds = fused_preds[:min_length]

    conf_mat =  confusion_matrix(all_v_gt, all_v_pred, normalize='true', labels=range(len(config['labels'])))
    v_acc = conf_mat.diagonal()/conf_mat.sum(axis=1)
    v_prec = precision_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_rec = recall_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))
    v_f1 = f1_score(all_v_gt, all_v_pred, average=None, zero_division=1, labels=range(len(config['labels'])))

    '''

    

    '''
    def fusion_method(tridet_pred, actionformer_pred, batch_size=1000):
        tridet_pred = np.asarray(tridet_pred)
        actionformer_pred = np.asarray(actionformer_pred)
        #tridet_pred = np.reshape(tridet_pred, actionformer_pred.shape)
        #fused_preds = (tridet_pred[:, np.newaxis] + actionformer_pred) / 2
        fused_preds = []
        for i in range(0, len(tridet_pred), batch_size):
            tridet_batch = tridet_pred[i:i+batch_size]
            actionformer_batch = actionformer_pred[i:i+batch_size]

            if tridet_batch.shape[0] < actionformer_batch.shape[0]:
                tridet_batch = np.resize(tridet_batch, actionformer_batch.shape)
            elif actionformer_batch.shape[0] < tridet_batch.shape[0]:
                actionformer_batch = np.resize(actionformer_batch, tridet_batch.shape)

            fused_batch = (tridet_batch + actionformer_batch) / 2
            fused_preds.append(fused_batch)

        fused_preds = np.concatenate(fused_preds)
        return fused_preds
        


    '''
    '''
    def calculate_accuracy(all_v_gt, fused_preds):
        print("Length of fused_preds:", len(fused_preds))
        print("Length of all_v_gt:", len(all_v_gt))
        assert len(all_v_gt) == len(fused_preds), "Number of ground truth labels and predicted labels must be the same."
        num_correct = sum(all_v_gt == fused_preds)
        accuracy = num_correct / len(all_v_gt)
        return accuracy
    '''

    '''

    def calculate_accuracy(tridet_v_gt, actionformer_v_gt, fused_preds):
        #assert len(tridet_v_gt) == len(fused_preds), "Number of ground truth labels and predicted labels must be the same."
    
        num_samples = len(tridet_v_gt)
        num_correct = 0
    
        for i in range(num_samples):
            if tridet_v_gt[i] == fused_preds[i]:
                num_correct += 1
    
        tridet_accuracy = num_correct / num_samples
    
        num_correct = 0
    
        for i in range(num_samples):
            if actionformer_v_gt[i] == fused_preds[i]:
                num_correct += 1
    
        actionformer_accuracy = num_correct / num_samples
    
        fused_accuracy = (tridet_accuracy + actionformer_accuracy) / 2
    
        return fused_accuracy
    
    fused_preds = fusion_method(tridet_pred, actionformer_pred)

    fused_accuracy = calculate_accuracy(tridet_v_gt, actionformer_v_gt, fused_preds)

    print("Accuracy - Fused Model: ", fused_accuracy)



'''
    


    '''
    def fuse_softmax_probs(tridet_probs, actionformer_probs):
        fused_probs = (actionformer_probs + tridet_probs) / 2  # Example: Simple average fusion
        return fused_probs

    fused_probs = fuse_softmax_probs(tridet_probs, actionformer_probs)

    labels = [0, 1, 0, 1, 1, ...]  # Replace with your actual labels

    def evaluate_model_score(softmax_probs, ground_truth_labels):
        num_samples = softmax_probs.shape[0]

    # Convert softmax probabilities to predicted labels
        predicted_labels = np.argmax(softmax_probs, axis=1)

    # Calculate accuracy
        accuracy = np.sum(predicted_labels == ground_truth_labels) / num_samples

    # Other evaluation metrics can be calculated here as well, depending on your requirements

        return accuracy

    # Step 6: Evaluate the final model score
    accuracy = evaluate_model_score(fused_probs, labels)

    print("Final Model Score:")
    print("Accuracy:", accuracy)
    '''

    '''
    with open(anno_split) as f:
        file = json.load(f)
    anno_file = file['database']
    ground_truth_labels = []
    for activity_instance in anno_file.values():
        label = activity_instance['annotations'][0]['label']
        segment = activity_instance['annotations'][0]['segment']
        segment_frames = activity_instance['annotations'][0]['segment (frames)']
        label_id = activity_instance['annotations'][0]['label_id']
        # Store the ground truth label information for later use
        ground_truth_labels.append((label, segment, segment_frames, label_id))

    fused_probs = (actionformer_probs + tridet_probs)/2

    fused_predictions = np.argmax(fused_probs, axis=1)

    accuracy = np.sum(fused_predictions == ground_truth_labels) / len(ground_truth_labels)

    print("Accuracy:", accuracy)

    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/60_frames_30_stride/deepconvlstm_short.yaml')
    parser.add_argument('--eval_type', default='split')
    parser.add_argument('--neptune', default=True, type=bool) 
    parser.add_argument('--seed', default=42, type=int)       
    parser.add_argument('--ckpt-freq', default=-1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)


