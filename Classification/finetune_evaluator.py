import re

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, cohen_kappa_score, \
    recall_score
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def get_metrics_for_multiclass(self, model):
        model.eval()

        truths = []
        preds = []
        for batch in tqdm(self.data_loader, mininterval=1):
            # 兼容处理：batch 可能是 (x,y) / (x,y,epoch_ids) / (x,y,epoch_ids,sample_ids)；
            # 这里不需要 epoch_ids/sample_ids，只取前两项
            x, y = batch[0], batch[1]

            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            pred_y = torch.max(pred, dim=-1)[1]

            truths += y.cpu().squeeze().numpy().tolist()
            preds += pred_y.cpu().squeeze().numpy().tolist()

        truths = np.array(truths)
        preds = np.array(preds)
        acc = accuracy_score(truths, preds)
        bacc = balanced_accuracy_score(truths, preds)
        # [保留] weighted F1，用于与旧实验日志对齐
        f1_weighted = f1_score(truths, preds, average='weighted')
        # ===== [R1 新增] Macro F1 + Per-class Recall（审稿要求的扩展指标）=====
        f1_macro = f1_score(truths, preds, average='macro')
        per_class_recall = recall_score(truths, preds, average=None, zero_division=0)
        kappa = cohen_kappa_score(truths, preds)
        cm = confusion_matrix(truths, preds)

        # 默认仍返回 weighted F1，保持旧 random_epoch 实验日志可对比；
        # subject_independent 时额外打印审稿指标，并返回 macro F1。
        split_mode = getattr(self.params, 'split_mode', 'random_epoch')
        if split_mode == 'subject_independent':
            print("[R1 Expanded Test/Val Metrics]")
            print(f"  Balanced Accuracy: {bacc:.5f}")
            print(f"  Macro F1:          {f1_macro:.5f}")
            print(f"  Weighted F1:       {f1_weighted:.5f}")
            print(f"  Per-class Recall:  {np.array2string(per_class_recall, precision=5, separator=', ')}")
            print("  Confusion Matrix:")
            print(cm)
            f1 = f1_macro
        else:
            f1 = f1_weighted
        return acc, bacc, kappa, f1, cm

    def save_test_predictions_for_mcnemar(self, model, save_path):
        """在最佳 val bacc 时调用，保存 test 逐样本预测（与 get_metrics 使用相同 data_loader）"""
        import os
        model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1):
                if len(batch) >= 3:
                    x, y, epoch_ids = batch[0], batch[1], batch[2]
                else:
                    break  # 需要 epoch_ids
                x = x.cuda()
                pred = model(x)
                pred_y = torch.max(pred, dim=-1)[1]
                for i, eid in enumerate(epoch_ids):
                    p = int(pred_y[i].cpu().item())
                    t = int(y[i].cpu().item())
                    c = 1 if p == t else 0
                    results.append((eid, p, t, c))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('epoch_id\tpred\ttrue\tcorrect\n')
            for eid, p, t, c in results:
                f.write(f'{eid}\t{p}\t{t}\t{c}\n')
        print(f"Saved test predictions for McNemar to {save_path}")

    def save_fold_predictions_npz(self, model, task, model_name, fold_idx, save_dir):
        """
        [R2 新增] 对 self.data_loader（调用方传入 test loader）跑一遍推理，
        按 sample_id 排序后保存为 {task}_{model_name}_fold{fold_idx:02d}.npz，
        字段：sample_id / y_true / y_pred / y_prob(softmax，全部类别) / subject_id。
        目的：所有下游指标事后都能从这个 npz 重新算，不需要重跑训练。

        要求 dataset 的 collate 返回 (x, y, epoch_ids, sample_ids) 四元组
        （见 datasets/motortask_dataset.py 的 compute_sample_id）；
        任何异常（batch 里没有 sample_id、结果为空、写文件失败、写完读不回来）
        都直接抛异常退出，不静默跳过——保存失败必须让调用方知道。
        """
        import os
        model.eval()
        sample_ids, y_true, y_pred, y_prob, subject_ids = [], [], [], [], []
        subj_re = re.compile(r'^S(\d+)_')
        with torch.no_grad():
            for batch in tqdm(self.data_loader, mininterval=1, desc="save_fold_predictions_npz"):
                if len(batch) < 4:
                    raise RuntimeError(
                        f"save_fold_predictions_npz requires (x, y, epoch_ids, sample_ids) batches "
                        f"(got {len(batch)} elements) — dataset collate must return sample_id."
                    )
                x, y, epoch_ids, batch_sample_ids = batch[0], batch[1], batch[2], batch[3]
                x = x.cuda()
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                for i, sid in enumerate(batch_sample_ids):
                    m = subj_re.match(sid)
                    if not m:
                        raise ValueError(f"Cannot parse subject_id from sample_id: {sid!r}")
                    sample_ids.append(sid)
                    y_true.append(int(y[i].item()))
                    y_pred.append(int(preds[i].cpu().item()))
                    y_prob.append(probs[i].cpu().numpy())
                    subject_ids.append(int(m.group(1)))

        if len(sample_ids) == 0:
            raise RuntimeError("save_fold_predictions_npz: no samples collected, refusing to save an empty file.")

        sample_ids_arr = np.array(sample_ids)
        order = np.argsort(sample_ids_arr)
        sample_ids_arr = sample_ids_arr[order]
        y_true_arr = np.array(y_true, dtype=np.int64)[order]
        y_pred_arr = np.array(y_pred, dtype=np.int64)[order]
        y_prob_arr = np.array(y_prob, dtype=np.float32)[order]
        subject_id_arr = np.array(subject_ids, dtype=np.int64)[order]

        os.makedirs(save_dir, exist_ok=True)
        npz_path = os.path.join(save_dir, f"{task}_{model_name}_fold{fold_idx:02d}.npz")
        np.savez(
            npz_path,
            sample_id=sample_ids_arr,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_prob=y_prob_arr,
            subject_id=subject_id_arr,
        )
        if not os.path.exists(npz_path):
            raise RuntimeError(f"save_fold_predictions_npz: failed to write {npz_path}")
        # 保存完立刻回读校验，保存失败/损坏要当场报错，而不是留到事后分析才发现
        check = np.load(npz_path)
        for key in ("sample_id", "y_true", "y_pred", "y_prob", "subject_id"):
            if key not in check:
                raise RuntimeError(f"save_fold_predictions_npz: {npz_path} missing key '{key}' after save")
            if len(check[key]) != len(sample_ids_arr):
                raise RuntimeError(f"save_fold_predictions_npz: {npz_path} key '{key}' length mismatch after save")

        print(f"Saved fold predictions npz to {npz_path} ({len(sample_ids_arr)} samples)")
        return npz_path, sample_ids_arr, y_true_arr, y_pred_arr, y_prob_arr, subject_id_arr
