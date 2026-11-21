"""Read one raw XDF recording."""
import os

import numpy as np
import pyxdf


def load(path, channels):
    """EEG channels, packet counter and raw arrival times; audio markers; keyboard events."""
    streams, _ = pyxdf.load_xdf(path, dejitter_timestamps=False)
    eeg = next(s for s in streams if s['info']['type'][0] == 'EEG')
    named = {s['info']['name'][0]: s for s in streams}
    labels = [c['label'][0] for c in eeg['info']['desc'][0]['channels'][0]['channel']]
    data = eeg['time_series']

    # marker text: "<tone>,<trial>,<stimulus id>,<time of the play command>"
    audio = named['AudioMarkers']
    fields = [v[0].split(',') for v in audio['time_series']]
    kb = named.get('Keyboard', {'time_stamps': np.zeros(0), 'time_series': []})
    keys = [v[0].rsplit(' ', 1) for v in kb['time_series']]
    return dict(
        file=os.path.basename(path),
        host=eeg['info']['hostname'][0],
        eeg_ts=np.asarray(eeg['time_stamps'], dtype=np.float64),
        eeg=np.array([data[:, labels.index(c)] for c in channels], dtype=np.float64),
        counter=np.asarray(data[:, labels.index('Packet Counter')]).astype(np.int64),
        marker_ts=np.asarray(audio['time_stamps'], dtype=np.float64),
        tone=np.array([f[0] for f in fields]),
        trial=np.array([int(f[1]) for f in fields]),
        key_ts=np.asarray(kb['time_stamps'], dtype=np.float64),
        key=[k[0] for k in keys],
        action=[k[1] for k in keys],
    )
