# Movement classification

The pretrained EEG model CBraMod is fine-tuned to classify 1-s EEG epochs of the movement periods into the six movements (labels 0-5: walkstraight, walkcurved, headshake, headnod, pickup, stairs). Leave-one-subject-out cross-validation runs 20 folds; in each fold one participant is the test set, another participant the validation set and the other 18 the training set. Each fold trains for 50 epochs, and the weights of the epoch with the highest balanced accuracy on the validation participant give the test predictions.

## Input

The input is cleaned EEG of the movement segments. These files are not part of the dataset. They are made from the raw EEG as follows:

1. The continuous recording is high-pass filtered at 3 Hz and low-pass filtered at 50 Hz (EEGLAB `pop_eegfiltnew`) and then cut into segments; where a pause is cut out of a segment, a boundary event marks the cut.
2. Each segment is resampled to 250 Hz (`pop_resample`), then `preprocess_eeg_forASR.m` re-references it to the common average and removes line noise with CleanLine (60 and then 120 Hz).
3. `apply_asr(EEG, 10)` runs artifact subspace reconstruction (clean_rawdata) with cutoff k = 10 and a drift high-pass of 0.25-0.75 Hz, keeping every channel and sample, followed by a 40 Hz low-pass filter. The resampling and this filter run separately on the two sides of a boundary.
4. Recordings with the 30-channel headset go through these steps with all 30 channels; then the 20 channels of the 20-channel montage are kept.

Each movement segment is saved with `save(file, 'EEG')` and named after its segment in the dataset, e.g. `sub-01_task-walkstraight_acq-slowmove.mat`.

The MATLAB functions need EEGLAB with the CleanLine and clean_rawdata plugins. The Python code needs torch, scikit-learn, einops, tqdm, pyyaml, mne, scipy and numpy; training needs a CUDA GPU.

## Run

1. Epochs (0.3-75 Hz band-pass, resampling to 200 Hz, non-overlapping 1-s epochs; the data loader divides them by 100):

       python classification/make_epochs.py --segments /path/to/clean_segments --out /path/to/epochs

2. Pretrained weights: download `pretrained_weights.pth` from the CBraMod model page, https://huggingface.co/weighting666/CBraMod.
3. Folds: set `EPOCHS`, `WEIGHTS` and `OUT` at the top of `run_loso.sh`, which lists the 20 (test, validation) pairs, and run

       bash classification/run_loso.sh

Each fold folder holds the selected model (`best_model_*.pth`), `config.yaml`, `training_logs.json`, `training_summary.json` and the test predictions: `motortask_cbramod_foldNN.npz` (epoch_id, y_true, y_pred, y_prob, subject_id) and `motortask_cbramod_foldNN.json` (participants, selected epoch, balanced accuracy, settings). In `training_summary.json`, `test_bacc` is the balanced accuracy and `test_f1` the macro-F1 over the classes present in the test participant.

## Origin

The training code (`finetune_*.py`, `datasets/`, `models/`) and `run_loso.sh` are the six-movement part of `cbramod_finetune` in [EEG-Review-Paper-Repo](https://github.com/XushengZhu-ROOT/EEG-Review-Paper-Repo) (commit c0a6ae6), with participant labels as in the dataset; `make_epochs.py` holds the epoching that prepared its input. The training code is adapted from [CBraMod](https://github.com/wjq-learning/CBraMod) (MIT licence, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); `models/cbramod.py` and `models/criss_cross_transformer.py` are unchanged copies. `preprocess_eeg_forASR.m` and `apply_asr.m` were written by Ella (HsinYu) Shih.
