PlaNet
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

PlaNet: A Deep Planning Network for Reinforcement Learning [[1]](#references). Supports symbolic/visual observation spaces. Supports some Gym environments (including classic control/non-MuJoCo environments, so DeepMind Control Suite/MuJoCo are optional dependencies). Hyperparameters have been taken from the original work and are tuned for DeepMind Control Suite, so would need tuning for any other domains (such as the Gym environments).

Run with `python.main.py`. For best performance with DeepMind Control Suite, try setting environment variable `MUJOCO_GL=egl` (see instructions and details [here](https://github.com/deepmind/dm_control#rendering)).


Results and pretrained models can be found in the [releases](https://github.com/Kaixhin/PlaNet/releases).

Requirements
------------

- Python 3
- [DeepMind Control Suite](https://github.com/deepmind/dm_control) (optional)
- [Gym](https://gym.openai.com/)
- [OpenCV Python](https://pypi.python.org/pypi/opencv-python)
- [Plotly](https://plot.ly/)
- [PyTorch](http://pytorch.org/)

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate planet` to activate the environment. 

Links
-----

- [Introducing PlaNet: A Deep Planning Network for Reinforcement Learning](https://ai.googleblog.com/2019/02/introducing-planet-deep-planning.html)
- [google-research/planet](https://github.com/google-research/planet)

Acknowledgements
----------------

- [@danijar](https://github.com/danijar) for [google-research/planet](https://github.com/google-research/planet) and [help reproducing results](https://github.com/google-research/planet/issues/28)
- [@sg2](https://github.com/sg2) for [running experiments](https://github.com/Kaixhin/PlaNet/issues/9)

References
----------

[1] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  
