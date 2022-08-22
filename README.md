# InferringPitch

## Install Dependencies
* Python version &ge; 3.6
* Download [PynkTrombone](https://github.com/dkadish/pynktrombone), and add the directory path to `sys.path`, so that we can import `pynkTrombone` module from anywhere:

```
$ export PYTHONPATH='path/to/pynktrombone'
```
* If you are _not_ using GPU with CUDA 11.3, you might want to change `torchaudio==0.12.1+cu113` in `requirements.txt` to your own CUDA version before installation.
* Install other required packages:

```
$ pip install -r requirements.txt
```

## Code Organization
* `varying_*.py` synthesizes audio data using varying F0 in the Pink Trombone, and stores varied F0 values.
* `example_audio/` contains examples of synthesized audio.
* `inversion.py` does articulatory inversion and predicts the fundamental frequency of glottal waves in the Pink Trombone.
* `reconstruction.py` reconstructs audio data from predicted F0, and also plots the original and predicted waveforms.

## References
This is the code repository for the paper [Inferring Pitch from Coarse Spectral Features](https://arxiv.org/pdf/2204.04579.pdf). If you use the code in your work, please cite:

```
@article{ma2022inferring,
  title={Inferring Pitch from Coarse Spectral Features},
  author={Ma, Danni and Ryant, Neville and Liberman, Mark},
  journal={arXiv preprint arXiv:2204.04579},
  year={2022}
}
```
