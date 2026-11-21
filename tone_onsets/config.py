"""Parameters of the tone onset estimate (units: s, Hz)."""

FS = 500                        # nominal EEG sampling rate
CHANNELS = ('Fz', 'Cz')         # c is estimated from their mean

# zero-phase Butterworth band-pass of each gap-free segment
FILTER_ORDER = 2
BAND = (0.3, 20.0)

# EEG sample clock (eeg_clock.py)
SEGMENT_GAP = 2.0               # an arrival gap longer than this starts a new segment
ENVELOPE_WINDOW = 1.0           # the earliest arrival in each window traces the sample clock
ENVELOPE_KEEP = 60              # percentile of the window minima kept by the trimmed line fit
OFF_CLOCK_TOL = 0.020           # off-clock: the earliest arrival is more than this above the line
OFF_CLOCK_SPAN = 11             # ... for at least this many consecutive windows
COUNTER_MOD = 128               # the CGX packet counter runs 0..127, +1 per sample

# epoch around the LSL audio marker; a trial's status refers to this window
MARKER_EPOCH = (-0.6, 0.9)

# c = marker time - sound onset, from the N1 to standard tones at rest
N1_LATENCY = 0.100              # assumed latency of the auditory N1 peak after sound onset
N1_SEARCH = (-0.25, 0.15)       # where the N1 peak is looked for, relative to the marker
TRIM = 0.2                      # subject average: 20% trimmed mean over trials
PEAK_CI_HALF = 0.05             # bootstrap N1 search window: peak +- this
BOOTSTRAP = 2000
SEED = 1

# audio set-up of each recording laptop (LSL host name)
LAPTOPS = {'LAPTOP-KS96G76L': 'KS96', 'LAPTOP-0D968NO9': '0D96', 'Janmouking-Lenovo': 'Lenovo'}
# Sub02 ran on KS96 with a different audio set-up and gets its own c
SEPARATE_SUBJECTS = {'Sub02': 'KS96-Sub02'}
SETUPS = ('KS96', '0D96', 'Lenovo', 'KS96-Sub02')
