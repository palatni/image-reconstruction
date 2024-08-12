# Image Reconstruction

This is a toy project with a [Pytorch](https://pytorch.org) implementation of image reconstruction alogorithm suuggested in the following paper: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)

[Original project based on JAX](https://github.com/tancik/fourier-feature-networks).

## Project description
A framework for image reconstraction experiments has been implemented which allows to store metrics and cache a video of the reconstuction quality during training.

Also, a [Streamlit](https://streamlit.io/)-based demonstration has been implemented that allows to monitor the reconstruction quality during the training

![Demonstration](https://drive.google.com/uc?export=view&id=1ywqvPVTrUK1LpWlOBH_YshvRmZDh06ut)
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