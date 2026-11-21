# Movement classification

The pretrained EEG model CBraMod is fine-tuned to classify 1-s EEG epochs of the movement periods into the six movements (labels 0-5: walkstraight, walkcurved, headshake, headnod, pickup, stairs). Leave-one-subject-out cross-validation runs 20 folds (test, validation and 18 training participants); each fold trains for 50 epochs and keeps the epoch with the best validation balanced accuracy.

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

Each fold writes the selected model, its training logs and the test predictions (`motortask_cbramod_foldNN.npz`).
