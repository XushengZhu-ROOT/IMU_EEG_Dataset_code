"""Parameters of the tone onset estimate (units: s, Hz, uV)."""

FS = 500                        # nominal sampling rate of the CGX Quick-20r / Quick-32r headsets
CHANNELS = ('Fz', 'Cz')         # c is set from the mean of Fz and Cz as recorded (online reference A1)

# continuous filtering: Butterworth band-pass, run forward and backward (zero phase) on each gap-free segment
FILTER_ORDER = 2
BAND = (0.3, 20.0)

# EEG sample clock (eeg_clock.py)
SEGMENT_GAP = 2.0               # an arrival gap longer than this starts a new segment (longest in the data: 1.1 s)
ENVELOPE_WINDOW = 1.0           # the earliest arrival in each window traces the sample clock
ENVELOPE_KEEP = 60              # percentile of the window minima kept by the trimmed line fit
OFF_CLOCK_TOL = 0.020           # samples are off-clock where the earliest arrival stays > 20 ms above the line
OFF_CLOCK_SPAN = 11             # ... in every window of an 11-window (11 s) neighbourhood
COUNTER_MOD = 128               # the CGX packet counter runs 0..127, +1 per sample

# epochs locked to the LSL audio marker. A trial's status refers to this window, which contains -200..800 ms around
# the corrected onset for every set-up.
MARKER_EPOCH = (-0.6, 0.9)

# c = marker time - sound onset, from the N1 to standard tones at rest
N1_LATENCY = 0.100              # assumed latency of the auditory N1 peak after sound onset
N1_SEARCH = (-0.25, 0.15)       # where the N1 peak is looked for, relative to the marker
TRIM = 0.2                      # subject average = 20% trimmed mean over trials at each time point
PEAK_CI_HALF = 0.05             # bootstrap CIs of the N1 latency search +-50 ms around the peak
BOOTSTRAP = 2000
SEED = 1

# Audio set-ups. audio.py ran on the recording laptop (LSL host name) and played the tones through its speakers.
LAPTOPS = {'LAPTOP-KS96G76L': 'KS96', 'LAPTOP-0D968NO9': '0D96', 'Janmouking-Lenovo': 'Lenovo'}
# Sub02, the first session, ran on KS96 with a different audio set-up, so it gets its own c.
SEPARATE_SUBJECTS = {'Sub02': 'KS96-Sub02'}
SETUPS = ('KS96', '0D96', 'Lenovo', 'KS96-Sub02')
