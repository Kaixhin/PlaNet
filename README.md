PlaNet
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

PlaNet: A Deep Planning Network for Reinforcement Learning. Supports symbolic/visual observation spaces. Supports some Gym environments (including classic control/non-MuJoCo environments, so DeepMind Control Suite/MuJoCo are optional dependencies).

Run with `python.main.py`. For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)).

Requirements
------------

- Python 3
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)
- [Gym](https://gym.openai.com/)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [PyTorch](http://pytorch.org/)
- [seaborn](https://seaborn.pydata.org/)

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate rainbow` to activate the environment. 

Links
-----

- [Introducing PlaNet: A Deep Planning Network for Reinforcement Learning](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/pdf/1811.04551.pdf)
- [google-research/planet](https://github.com/google-research/planet)
