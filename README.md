# Image Reconstruction using Fourier Features: A PyTorch Implementation

This project provides a [Pytorch](https://pytorch.org) implementation of an image reconstruction algorithm, as presented in
the paper: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)


[Original JAX-based project](https://github.com/tancik/fourier-feature-networks).

## Project description
This project includes a framework for conducting image reconstruction experiments. It enables Storage of training metrics and caching of a video showcasing the reconstruction quality throughout the training process

Also, a [Streamlit](https://streamlit.io/) application is provided to visualize and monitor the reconstruction quality in real-time during training.

![Demonstration](https://drive.google.com/uc?export=view&id=1I-GEFByjpEZ4djkiCA6E2oCVL3ZEoihF)
## Quickstart

Create virtual environment and install required packages:
```bash
python3 - m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download an image:
```bash
bash download_target_img.sh
```

Run a training process together with the demonstration app:

```bash
bash run_demonstraton.sh -c config.yaml
```

For further details see the files' docstrings.