#!/bin/bash
# Fine-tunes CBraMod in 20 leave-one-subject-out folds.
# Set the paths below, then run: bash classification/run_loso.sh
# Each fold writes to $OUT/cbramod_loso/foldNN_test-sub-XX; finished folds are skipped.

EPOCHS=/path/to/epochs                        # folder written by make_epochs.py
WEIGHTS=/path/to/pretrained_weights.pth       # pretrained CBraMod weights
OUT=/path/to/output
GPU=0
PYTHON=python

for p in "$EPOCHS" "$WEIGHTS" "$OUT"; do
    case "$p" in
        /path/to/*) echo "Set EPOCHS, WEIGHTS and OUT at the top of $0" >&2; exit 1 ;;
    esac
done
HERE=$(cd "$(dirname "$0")" && pwd)

# test participant, validation participant
FOLDS=(
    "sub-01 sub-02"
    "sub-02 sub-03"
    "sub-03 sub-04"
    "sub-04 sub-05"
    "sub-05 sub-06"
    "sub-06 sub-07"
    "sub-07 sub-08"
    "sub-08 sub-09"
    "sub-09 sub-10"
    "sub-10 sub-11"
    "sub-11 sub-13"
    "sub-12 sub-01"
    "sub-13 sub-14"
    "sub-14 sub-15"
    "sub-15 sub-16"
    "sub-16 sub-17"
    "sub-17 sub-18"
    "sub-18 sub-19"
    "sub-19 sub-20"
    "sub-20 sub-12"
)

for i in "${!FOLDS[@]}"; do
    read -r test_subject val_subject <<< "${FOLDS[$i]}"
    fold=$(printf "fold%02d" "$i")
    fold_dir="$OUT/cbramod_loso/${fold}_test-${test_subject}"

    if [ -f "$fold_dir/motortask_cbramod_${fold}.json" ]; then
        echo "${fold} (test ${test_subject}, val ${val_subject}) done already, skipped"
        continue
    fi
    echo "LOSO ${fold} ($((i + 1))/${#FOLDS[@]}): test ${test_subject}, val ${val_subject}"

    "$PYTHON" "$HERE/finetune_main.py" \
        --seed 0 \
        --epochs 50 \
        --datasets_dir "$EPOCHS" \
        --optimizer AdamW \
        --num_of_classes 6 \
        --channel_size 20 \
        --window_size 1 \
        --use_pretrained_weights True \
        --foundation_dir "$WEIGHTS" \
        --cuda "$GPU" \
        --batch_size 256 \
        --lr 0.0005 \
        --weight_decay 0.002 \
        --val_subject "$val_subject" \
        --test_subject "$test_subject" \
        --fold_idx "$i" \
        --out "$fold_dir"
done

echo "Test predictions of all folds: $OUT/cbramod_loso/fold*/motortask_cbramod_fold*.npz"
