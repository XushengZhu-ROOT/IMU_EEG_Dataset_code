"""Fine-tuning loop; keeps the best validation epoch.

Adapted from https://github.com/wjq-learning/CBraMod (MIT licence).
"""
import json
import os
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from finetune_evaluator import Evaluator

RESULT_PREFIX = 'motortask_cbramod'


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])

        self.model = model.cuda()
        self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()

        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
                param.requires_grad = not params.frozen
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:  # the classification head learns at 5 x lr
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)

        self.training_start_time = None
        self.training_logs = []

    def train_for_multiclass(self):
        self.training_start_time = time.time()
        best_metrics = {
            'val_acc': 0,
            'val_bacc': -1,  # the first epoch is always recorded
            'val_kappa': 0,
            'val_f1': 0,
            'val_cm': None,
            'test_acc': 0,
            'test_bacc': 0,
            'test_kappa': 0,
            'test_f1': 0,
            'test_cm': None,
            'epoch': 0
        }

        for epoch in range(self.params.epochs):
            self.model.train()
            epoch_start_time = time.time()

            losses = []
            for x, y, _ in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()

                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())

                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)

                self.optimizer.step()
                self.optimizer_scheduler.step()

            with torch.no_grad():
                val_acc, val_bacc, val_kappa, val_f1, val_cm = self.val_eval.get_metrics_for_multiclass(self.model)
                test_acc, test_bacc, test_kappa, test_f1, test_cm = self.test_eval.get_metrics_for_multiclass(self.model)

                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.training_logs.append({
                    'epoch': epoch + 1,
                    'train_loss': float(np.mean(losses)),
                    'val_acc': float(val_acc),
                    'val_bacc': float(val_bacc),
                    'val_kappa': float(val_kappa),
                    'val_f1': float(val_f1),
                    'test_acc': float(test_acc),
                    'test_bacc': float(test_bacc),
                    'test_kappa': float(test_kappa),
                    'test_f1': float(test_f1),
                    'learning_rate': float(current_lr),
                    'epoch_time_seconds': float(epoch_time),
                    'epoch_time_minutes': float(epoch_time / 60),
                })

                print(f"Epoch {epoch + 1}: Training Loss: {np.mean(losses):.5f}")
                print(f"  Val  - acc: {val_acc:.5f}, bacc: {val_bacc:.5f}, kappa: {val_kappa:.5f}, f1: {val_f1:.5f}")
                print(f"  Test - acc: {test_acc:.5f}, bacc: {test_bacc:.5f}, kappa: {test_kappa:.5f}, f1: {test_f1:.5f}")
                print(f"  LR: {current_lr:.5f}, Time: {epoch_time/60:.2f} mins")
                print("Val CM:")
                print(val_cm)
                print("Test CM:")
                print(test_cm)

                if val_bacc > best_metrics['val_bacc']:
                    print(">>> Val BACC increasing... updating best model!")
                    best_metrics.update({
                        'val_acc': val_acc,
                        'val_bacc': val_bacc,
                        'val_kappa': val_kappa,
                        'val_f1': val_f1,
                        'val_cm': val_cm,
                        'test_acc': test_acc,
                        'test_bacc': test_bacc,
                        'test_kappa': test_kappa,
                        'test_f1': test_f1,
                        'test_cm': test_cm,
                        'epoch': epoch + 1
                    })
                    self.best_model_states = {k: v.clone() for k, v in self.model.state_dict().items()}

        print("Training completed")
        print(f"Best model from Epoch {best_metrics['epoch']} (selected by Val BACC)")
        print(f"Best Val  - acc: {best_metrics['val_acc']:.5f}, bacc: {best_metrics['val_bacc']:.5f}, "
              f"kappa: {best_metrics['val_kappa']:.5f}, f1: {best_metrics['val_f1']:.5f}")
        print(f"Best Test - acc: {best_metrics['test_acc']:.5f}, bacc: {best_metrics['test_bacc']:.5f}, "
              f"kappa: {best_metrics['test_kappa']:.5f}, f1: {best_metrics['test_f1']:.5f}")

        final_results = {
            'best_epoch': best_metrics['epoch'],
            'val_acc': float(best_metrics['val_acc']),
            'val_bacc': float(best_metrics['val_bacc']),
            'val_kappa': float(best_metrics['val_kappa']),
            'val_f1': float(best_metrics['val_f1']),
            'val_cm': best_metrics['val_cm'].tolist(),
            'test_acc': float(best_metrics['test_acc']),
            'test_bacc': float(best_metrics['test_bacc']),
            'test_kappa': float(best_metrics['test_kappa']),
            'test_f1': float(best_metrics['test_f1']),
            'test_cm': best_metrics['test_cm'].tolist(),
        }

        model_path = os.path.join(
            self.params.out,
            f"best_model_epoch{best_metrics['epoch']}_valBacc{best_metrics['val_bacc']:.5f}_testBacc{best_metrics['test_bacc']:.5f}.pth"
        )
        torch.save(self.best_model_states, model_path)
        print(f"Best model saved to {model_path}")

        self.save_training_logs(final_results)
        self.save_fold_results(best_metrics)

    def save_training_logs(self, final_results):
        log_path = os.path.join(self.params.out, 'training_logs.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=4)
        print(f"Training logs saved to {log_path}")

        elapsed = time.time() - self.training_start_time
        summary = {
            'training_info': {
                'total_epochs': self.params.epochs,
                'total_training_time_seconds': elapsed,
                'average_time_per_epoch_seconds': elapsed / self.params.epochs,
                'optimizer': self.params.optimizer,
                'learning_rate': self.params.lr,
                'batch_size': self.params.batch_size,
            },
            'final_results': final_results,
        }
        summary_path = os.path.join(self.params.out, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Training summary saved to {summary_path}")

    def save_fold_results(self, best_metrics):
        """Save the test predictions (.npz) and fold settings (.json)."""
        self.model.load_state_dict(self.best_model_states)
        fold_idx = self.params.fold_idx
        base = os.path.join(self.params.out, f"{RESULT_PREFIX}_fold{fold_idx:02d}")

        predictions = self.test_eval.predict(self.model)
        np.savez(base + '.npz', **predictions)
        print(f"Saved test predictions to {base}.npz ({len(predictions['epoch_id'])} epochs)")

        meta = {
            'fold': fold_idx,
            'test_subject': self.params.test_subject,
            'val_subject': self.params.val_subject,
            'balanced_accuracy': float(best_metrics['test_bacc']),
            'best_epoch': best_metrics['epoch'],
            'hyperparams': {
                'lr': self.params.lr,
                'weight_decay': self.params.weight_decay,
                'batch_size': self.params.batch_size,
                'epochs': self.params.epochs,
                'optimizer': self.params.optimizer,
                'seed': self.params.seed,
                'channel_size': self.params.channel_size,
                'window_size': self.params.window_size,
                'frozen': bool(self.params.frozen),
            },
            'train_time_sec': time.time() - self.training_start_time,
        }
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"Saved fold settings to {base}.json")
