#!/usr/bin/env python
"""Auditory oddball stimulus program used during the recordings.

Plays each tone, then sends its LSL marker (stream AudioMarkers). Writes tone_waveforms.png and
oddball_experiment_log.csv to the working directory.
"""

import numpy as np
import time
import random
import sounddevice as sd
import pylsl
import matplotlib.pyplot as plt
import pandas as pd

SAMPLE_RATE = 44100  # Hz
STANDARD_FREQ = 1000  # Hz
TARGET_FREQ = 3000    # Hz
DURATION = 0.3       # tone duration (s)
RISE_FALL_TIME = 0.005  # ramp duration (s)
MIN_ISI = 1.5        # shortest wait after a tone (s)
MAX_ISI = 2         # longest wait after a tone (s)
TOTAL_TRIALS = 30000    # length of the trial list
TARGET_PROB = 0.4     # share of targets in the list
VOLUME = 1          # 0-1
MIN_STANDARDS_BETWEEN_TARGETS = 2  # standards after each target


def create_lsl_outlet():
    """LSL marker stream for the tone markers."""
    info = pylsl.StreamInfo(
        name='AudioMarkers',
        type='Markers',
        channel_count=1,
        nominal_srate=0,  # irregular rate
        channel_format='string',
        source_id='auditory_oddball'
    )

    info.desc().append_child_value("manufacturer", "Custom Script")
    channels = info.desc().append_child("channels")
    channels.append_child("channel") \
        .append_child_value("label", "Stimulus Marker") \
        .append_child_value("type", "Trigger")

    outlet = pylsl.StreamOutlet(info)
    print("LSL marker stream created.")
    return outlet


def generate_tone(frequency, duration=DURATION, rise_fall=RISE_FALL_TIME, fs=SAMPLE_RATE):
    """Sine tone with Hanning ramps at onset and offset."""
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)

    rise_fall_samples = int(rise_fall * fs)
    window = np.ones(len(tone))
    window[:rise_fall_samples] = 0.5 * (1 - np.cos(np.pi * np.arange(rise_fall_samples) / rise_fall_samples))
    window[-rise_fall_samples:] = 0.5 * (1 - np.cos(np.pi * (rise_fall_samples - np.arange(rise_fall_samples) - 1) / rise_fall_samples))

    tone = tone * window * VOLUME

    return tone


def plot_tones():
    """Waveforms of the standard and target tones, saved to tone_waveforms.png."""
    standard = generate_tone(STANDARD_FREQ)
    target = generate_tone(TARGET_FREQ)

    t = np.linspace(0, DURATION, len(standard), False)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, standard)
    plt.title(f'Standard tone: {STANDARD_FREQ} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(t, target)
    plt.title(f'Target tone: {TARGET_FREQ} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig('tone_waveforms.png')
    plt.close()
    print("Tone waveforms saved to 'tone_waveforms.png'")


def generate_stimulus_sequence(total_trials, target_prob, min_standards_between_targets):
    """Trial list and the wait after each tone; at least min_standards_between_targets standards follow each target."""
    n_targets = int(total_trials * target_prob)
    n_standards = total_trials - n_targets

    sequence = []
    standards_left = n_standards
    targets_left = n_targets
    consecutive_standards = 0

    while standards_left + targets_left > 0:
        if consecutive_standards < min_standards_between_targets and targets_left < total_trials:
            sequence.append('standard')
            standards_left -= 1
            consecutive_standards += 1
        else:
            current_target_prob = targets_left / (standards_left + targets_left)

            if random.random() < current_target_prob:
                sequence.append('target')
                targets_left -= 1
                consecutive_standards = 0
            else:
                sequence.append('standard')
                standards_left -= 1
                consecutive_standards += 1

    actual_target_count = sequence.count('target')
    print(f"Targets: {actual_target_count} (planned: {n_targets})")
    print(f"Standards: {sequence.count('standard')} (planned: {n_standards})")

    isis = np.random.uniform(MIN_ISI, MAX_ISI, len(sequence))

    return sequence, isis


def play_tone(tone_array):
    """Play the tone and wait until playback has finished."""
    sd.play(tone_array, SAMPLE_RATE)
    sd.wait()


def run_experiment(lsl_outlet):
    """Run the oddball task: play each tone, send its marker, wait."""
    standard_tone = generate_tone(STANDARD_FREQ)
    target_tone = generate_tone(TARGET_FREQ)

    sequence, isis = generate_stimulus_sequence(
        TOTAL_TRIALS,
        TARGET_PROB,
        MIN_STANDARDS_BETWEEN_TARGETS
    )

    results = []

    print("\nThe experiment is about to start.")
    print(f"Trials: {TOTAL_TRIALS}")
    print(f"Expected duration: {sum(isis) + DURATION * TOTAL_TRIALS:.1f} s")
    print("Press Enter to start...")
    input()

    for i, (stim_type, isi) in enumerate(zip(sequence, isis)):
        if stim_type == 'standard':
            tone = standard_tone
            marker_value = "standard"
            stim_id = 0
        else:
            tone = target_tone
            marker_value = "target"
            stim_id = 1

        onset_time = time.time()  # time of the play command

        play_tone(tone)

        # marker text: tone, trial number, stimulus id, time of the play command
        marker_info = f"{marker_value},{i+1},{stim_id},{onset_time:.6f}"
        lsl_outlet.push_sample([marker_info])

        results.append({
            'trial': i+1,
            'type': stim_type,
            'id': stim_id,
            'onset_time': onset_time,
            'isi': isi
        })

        if (i+1) % 20 == 0:
            print(f"Done: {i+1}/{TOTAL_TRIALS} trials")

        time.sleep(isi)

    results_df = pd.DataFrame(results)
    results_df.to_csv('oddball_experiment_log.csv', index=False)
    print("\nExperiment finished.")
    print("Trial log saved to 'oddball_experiment_log.csv'")

    print("\nSummary:")
    print(f"Standards played: {results_df[results_df['type']=='standard'].shape[0]}")
    print(f"Targets played: {results_df[results_df['type']=='target'].shape[0]}")
    print(f"Target share: {results_df[results_df['type']=='target'].shape[0] / TOTAL_TRIALS:.2f}")
    print(f"Duration: {results_df['onset_time'].max() - results_df['onset_time'].min():.1f} s")


if __name__ == "__main__":
    try:
        print("Starting...")

        plot_tones()

        lsl_outlet = create_lsl_outlet()

        devices = sd.query_devices()
        print("\nAudio devices:")
        print(devices)

        default_output = sd.default.device[1]
        print(f"\nDefault output device: {default_output}")

        run_experiment(lsl_outlet)

    except KeyboardInterrupt:
        print("\nInterrupted from the keyboard.")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nStopping audio playback.")
        sd.stop()
