#!/usr/bin/env python
"""Generate synthetic data, varying fundamental frequency of the glottal wave.

To store the synthesized data in the directory ``data_dir``:

    python varying.py data_dir

By default one .wav file with be generated. It is possible to change the
number of generated files via the `--n-files`` flag:

    python varying.py data_dir --n-files 10

When processing large batches of audio, it may be desireable to parallelize
the computation, which may be done by specifying the number of parallel
processes to employ via the ``--n_jobs`` flag:

    python varying.py data_dir --n-files 10 --n-jobs -1

"""
import argparse
import numpy as np
import random
import os
import soundfile as sf
import string
import sys
import time
from datetime import timedelta
from joblib import delayed, parallel_backend, Parallel
from math import pi, sin
from matplotlib import pylab as plt
from pathlib import Path
from pynkTrombone.voc import Voc


sr: float = 44100


def initialize(voc):
    alpha = [random.uniform(-pi/2, pi/2) for _ in range(3)]
    _t = sin(alpha[0]) * 0.5 + 0.5
    voc.tenseness = _t
    diam = sin(alpha[1]) * 1.75 + 1.75
    idx = sin(alpha[2]) * 8.5 + 20.5
    voc.tongue_shape(idx, diam)
    return voc, _t, diam, idx


def plotting(out, plot_path):
    plt.plot(np.linspace(0, 5, len(out)), out)
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')
    plt.savefig(plot_path)
    plt.clf()


def play_update(parent_dir):
    voc = Voc(sr)
    voc, _t, diam, idx = initialize(voc)
    x = random.uniform(- pi / 2, pi / 2)
    beta = random.uniform(0.02, 0.1)
    freq = sin(x * beta) * 172 + 232
    freq_list = [freq]
    voc.frequency = freq
    out = voc.play_chunk()
    while out.shape[0] < sr * 5:
        x += 1
        freq = sin(x * beta) * 172 + 232
        freq_list.append(freq)
        voc.frequency = freq
        out = np.concatenate([out, voc.play_chunk()])

    # Create control parameter matrix
    frequencies = np.asarray(freq_list).reshape(-1, 1)
    params = np.array([[_t, diam, idx]])
    params = np.repeat(params, len(freq_list), axis=0)
    params = np.concatenate((params, frequencies), axis=1)

    # Prepare paths to save variables
    sys_random = random.SystemRandom()
    formats = string.digits + string.ascii_letters
    filename = ''.join(sys_random.choice(formats) for i in range(8))
    wav_path = parent_dir/(filename + '.wav')
    npy_path = parent_dir/(filename + '.npy')

    # Save
    np.save(npy_path, params)
    sf.write(wav_path, out, sr)
    input_plot_path = parent_dir/(filename + '_input.png')
    plotting(freq_list, input_plot_path)


def main():
    parser = argparse.ArgumentParser(
        description='Generate audio data from pynkTrombone by varying F0',
        add_help=True)
    parser.add_argument(
        'parent_dir', type=Path, help='path to synthesized audio data')
    parser.add_argument(
        '--n-files', default=1, type=int,
        help='number of generated wav files (default: %(default)s)')
    parser.add_argument(
        '--n-jobs', dest='n_jobs', nargs=None, default=-1, type=int,
        metavar='JOBS', help='number of parallel jobs (default: %(default)s)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    os.makedirs(args.parent_dir, exist_ok=True)

    with parallel_backend('multiprocessing', n_jobs=args.n_jobs):
        f = delayed(play_update)
        Parallel()(f(args.parent_dir) for _ in range(args.n_files))


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    print(f'Total running time: {str(duration).split(".")[0]}')
