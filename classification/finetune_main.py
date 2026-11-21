"""Fine-tune CBraMod to classify 1-s epochs into the six movements, for one leave-one-subject-out fold.

Adapted from finetune_main.py of CBraMod (https://github.com/wjq-learning/CBraMod, MIT licence).
run_loso.sh runs the 20 folds.
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml

from datasets import motortask_dataset
from finetune_trainer import Trainer
from models import model_for_motortask


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1', 'yes'):
        return True
    elif v.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v!r}')


def save_config(params):
    config_path = os.path.join(params.out, 'config.yaml')
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(vars(params), f, default_flow_style=False, allow_unicode=True)
    print(f"Configuration saved to {config_path}")


def save_architecture(model, out):
    arch_path = os.path.join(out, 'model_architecture.txt')
    with open(arch_path, 'w', encoding='utf-8') as f:
        f.write(str(model))
    print(f"Model architecture saved to {arch_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--cuda', type=int, default=1, help='CUDA device index')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of the backbone')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW or SGD')
    parser.add_argument('--clip_value', type=float, default=1, help='maximum gradient norm (0: no clipping)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout of the classification head')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing of the cross-entropy loss')
    parser.add_argument('--multi_lr', type=str2bool, default=True,
                        help='the classification head learns at 5 x lr')
    parser.add_argument('--frozen', action='store_true', help='freeze the backbone')
    parser.add_argument('--use_pretrained_weights', type=str2bool, default=True,
                        help='start from the pretrained CBraMod weights')
    parser.add_argument('--foundation_dir', type=str, default='pretrained_weights/pretrained_weights.pth',
                        help='file with the pretrained CBraMod weights')

    parser.add_argument('--datasets_dir', type=str, required=True, help='folder of epochs written by make_epochs.py')
    parser.add_argument('--num_of_classes', type=int, default=6, help='number of classes')
    parser.add_argument('--channel_size', type=int, default=20, help='number of EEG channels')
    parser.add_argument('--window_size', type=int, default=1, help='number of 1-s patches per epoch')
    parser.add_argument('--test_subject', type=str, required=True, help='test participant, e.g. sub-01')
    parser.add_argument('--val_subject', type=str, required=True, help='validation participant, e.g. sub-02')
    parser.add_argument('--fold_idx', type=int, default=0, help='fold number used in the result file names')
    parser.add_argument('--out', type=str, required=True,
                        help='new or empty folder for the model, logs and test predictions of the fold')

    params = parser.parse_args()
    if os.path.exists(params.out) and (not os.path.isdir(params.out) or os.listdir(params.out)):
        raise SystemExit(f'{params.out} exists and is not an empty folder')
    os.makedirs(params.out, exist_ok=True)
    print(params)
    save_config(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    data_loader = motortask_dataset.LoadDataset(params).get_data_loader()
    model = model_for_motortask.Model(params)
    save_architecture(model, params.out)
    t = Trainer(params, data_loader, model)
    t.train_for_multiclass()
    print('Done')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
