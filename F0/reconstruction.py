#!/usr/bin/env python
"""Reconstruct audio data from predicted control parameters.

To reconstruct audio from F0 values stored in ``pred.npy``, and test data in
``test_dir``, and to save the result in ``recon_dir``:

    python reconstruction.py pred.npy test_dir recon_dir

When processing large batches of audio, it may be desireable to parallelize
the computation, which may be done by specifying the number of parallel
processes to employ via the ``--n_jobs`` flag:

    python varying.py data_dir --n-files 10 --n-jobs -1

"""
import argparse
import numpy as np
import soundfile as sf
import os
import sys
import time
from datetime import timedelta
from joblib import delayed, parallel_backend, Parallel
from matplotlib import pylab as plt
from os import listdir
from pathlib import Path
from pynkTrombone.voc import Voc


sr = 44100


def _get_waveform_targets(train_dir, input_file):
    """Load inputs from .wav files. Load labels from .npy files"""
    wav_path = train_dir / input_file
    filename = input_file.split('.')[0]
    targets_path = Path(train_dir, filename + '.npy')
    waveform, sr = sf.read(wav_path)
    targets = np.load(targets_path)
    return waveform, targets


def get_waveform_targets(train_dir, train_input_files, n_jobs):
    """Returns all inputs/labels."""
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        f = delayed(_get_waveform_targets)
        res = Parallel(
            )(f(train_dir, input_file) for input_file in train_input_files)
    _, targets = zip(*res)
    targets = np.concatenate(targets, axis=0).astype(np.float32)
    return targets


def update(voc, params):
    voc.tenseness = params[0]
    voc.tongue_shape(params[2], params[1])
    voc.frequency = params[3]
    return voc


def reconstruct_audio(i, pred_params, test_input_files, test_labels,
                      recon_dir):
    voc = Voc(sr)
    audio = []
    for j in range(pred_params.shape[1]):
        params = pred_params[i, j]
        voc = update(voc, params)
        out = voc.play_chunk()
        audio.append(out)
    audio = np.array(audio).reshape(-1)
    test_input_file = test_input_files[i]
    filename = test_input_file.split('.')[0]
    wav_path = recon_dir/(filename + '_recon.wav')
    pred_plot_path = recon_dir/(filename + '_prediction.png')

    N = np.linspace(0, 5, test_labels.shape[1])  # sr / 512 * 5
    plt.plot(N, test_labels[i, :, -1], label='original')
    plt.plot(N, pred_params[i, :, -1], label='predicted')
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')
    plt.legend(loc='best')
    plt.savefig(pred_plot_path)
    plt.clf()
    sf.write(wav_path, audio, sr)


def main():
    parser = argparse.ArgumentParser(
        description='Audio data reconstruction.', add_help=True)
    parser.add_argument(
        'pred_file', type=Path,
        help='path to test data prediction file.')
    parser.add_argument('test_dir', type=Path, help='path to test data.')
    parser.add_argument(
        'recon_dir', type=Path, help='path to reconstructed data.')
    parser.add_argument(
        '--n-jobs', dest='n_jobs', nargs=None, default=-1, type=int,
        metavar='JOBS', help='number of parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    os.makedirs(args.recon_dir, exist_ok=True)

    pred = np.load(args.pred_file).reshape(-1, 431, 1)
    pred = np.clip(pred, 60, 404)
    test_input_files = [file for file in listdir(
        args.test_dir) if file.endswith('.wav')]
    test_labels = get_waveform_targets(
        args.test_dir, test_input_files, -1)
    test_labels = test_labels.reshape(-1, 431, 4)
    fixed_params = test_labels[:, :, :-1]
    pred_params = np.concatenate([fixed_params, pred], axis=-1)
    # (N, 431, 4)
    # N test files, 431 buffer frames, 4 control parameters (no velum)

    with parallel_backend('multiprocessing', n_jobs=args.n_jobs):
        f = delayed(reconstruct_audio)
        Parallel()(f(
            i, pred_params, test_input_files, test_labels, args.recon_dir)
                for i in range(pred_params.shape[0]))


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    print(f'Total running time: {str(duration).split(".")[0]}')
