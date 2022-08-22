#!/usr/bin/env python
"""A Linear Regression Model to do articulatory inversion.

To train on data in ``train_dir``, test on data in ``test_dir``, and save
trained model and predictions in``result_dir``:

    python inversion.py train_dir test_dir result_dir

By default we use 40-D MFCC features as the representation of training data.
It is possible to concatenate neighboring frames as contexts to MFCC features.
This can be done via ``--context-size`` flag. For instance,

    python inversion.py train_dir test_dir result_dir --context-size 5

will concatenate an 11-frame context (5-1-5), yielding a final feature
dimension of 440.

When processing large batches of audio, it may be desireable to parallelize
the computation, which may be done by specifying the number of parallel
processes to employ via the ``--n_jobs`` flag:

    python varying.py data_dir --n-files 10 --n-jobs -1

"""
import argparse
import numpy as np
import os
import pickle
import sys
import time
import torchaudio
from datetime import timedelta
from joblib import delayed, parallel_backend, Parallel
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def _get_feats_targets(train_dir, input_file, context_size):
    """Load inputs from .wav files. Load labels from .npy files"""
    wav_path = train_dir / input_file
    filename = input_file.split('.')[0]
    targets_path = Path(train_dir, filename + '.npy')
    waveform, sample_rate = torchaudio.load(wav_path)
    feats = torchaudio.transforms.MFCC(sample_rate=sample_rate, melkwargs={
        "n_fft": 1024, "n_mels": 40, "hop_length": 512})(waveform)
    targets = np.load(targets_path)[:, -1]
    feats = feats.cpu().detach().numpy().squeeze().T
    feats = feats[:len(targets)]
    feats = add_context(feats, context_size)
    return feats, targets


def get_feats_targets(train_dir, train_input_files, context_size, n_jobs):
    """Returns all inputs/labels."""
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        f = delayed(_get_feats_targets)
        res = Parallel()(f(train_dir, input_file, context_size)
                         for input_file in train_input_files)
    feats, targets = zip(*res)
    feats = np.concatenate(feats, axis=0).astype(np.float32)
    targets = np.concatenate(targets, axis=0).astype(np.float32)
    print('feats shape:', feats.shape)
    print('targets shape:', targets.shape)
    return feats, targets


def add_context(feats, win_size):
    """Append context to each frame.

    Parameters
    ----------
    feats : ndarray, (n_frames, feat_dim)
        Features.

    win_size : int
        Number of frames on either side to append.

    Returns
    -------
    ndarray, (n_frames, feat_dim*(win_size*2 + 1))
        Features with context added.
    """
    if win_size <= 0:
        return feats
    feats = np.pad(feats, [[win_size, win_size], [0, 0]], mode='edge')
    inds = np.arange(-win_size, win_size+1)
    feats = np.concatenate(
        [np.roll(feats, ind, axis=0) for ind in inds], axis=1)
    feats = feats[win_size:-win_size, :]
    return feats


def set_random_seed(seed):
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description='Run regressors for inversion.', add_help=True)
    parser.add_argument('train_dir', type=Path, help='path to training data')
    parser.add_argument('test_dir', type=Path, help='path to test data')
    parser.add_argument(
        'result_dir', type=Path,
        help='path to save trained model and predictions')
    parser.add_argument(
        '--context-size', default=0, type=int,
        help='context window size in frames')
    parser.add_argument(
        '--n-jobs', dest='n_jobs', nargs=None, default=-1, type=int,
        metavar='JOBS', help='number of parallel jobs (default: %(default)s)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    set_random_seed(666)
    os.makedirs(args.result_dir, exist_ok=True)

    train_input_files = [file for file in os.listdir(
        args.train_dir) if file.endswith('.wav')]
    test_input_files = [file for file in os.listdir(
        args.test_dir) if file.endswith('.wav')]
    train_inputs, train_labels = get_feats_targets(
        args.train_dir, train_input_files, args.context_size, args.n_jobs)
    test_inputs, test_labels = get_feats_targets(
        args.test_dir, test_input_files, args.context_size, args.n_jobs)

    reg = LinearRegression().fit(train_inputs, train_labels)

    model_path = args.result_dir/'model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(reg, f)
    reg = pickle.load(open(model_path, 'rb'))
    train_preds = reg.predict(train_inputs)
    train_preds = np.clip(train_preds, 60, 404)
    print('training RMSE:', mean_squared_error(
        train_labels, train_preds, squared=False))
    test_preds = reg.predict(test_inputs)
    preds_path = args.result_dir/'preds.npy'
    np.save(preds_path, test_preds)
    test_preds = np.clip(test_preds, 60, 404)
    print('test RMSE:', mean_squared_error(
        test_labels, test_preds, squared=False))


if __name__ == '__main__':
    start_time = time.monotonic()
    main()
    end_time = time.monotonic()
    duration = timedelta(seconds=end_time-start_time)
    print(f'Total running time: {str(duration).split(".")[0]}')
