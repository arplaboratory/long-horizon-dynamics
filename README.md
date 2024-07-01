# Learning Long-Horizon Predictions for Quadrotor Dynamics
This github repository contains the codebase accompanying the paper **Learning Long-Horizon Predictions for Quadrotor Dynamics** by [Pratyaksh Prabhav Rao](https://scholar.google.com/citations?user=_Vy11KoAAAAJ&hl=en&oi=sra), [Alessandro Saviolo](https://scholar.google.com/citations?user=HaOQ8AoAAAAJ&hl=en), [Tommaso Castiglione Ferrari](), and [Giuseppe Loianno](https://scholar.google.com/citations?user=W8f0d6oAAAAJ&hl=en&oi=ao).

![Proposed Methodology](assets/methodology.png)

An overview of our proposed methodology can be found [video]().

## Abstract
Accurate modeling of system dynamics is crucial for achieving high-performance planning and control of robotic systems. Although existing data-driven approaches represent a promising approach for modeling dynamics, their accuracy is limited to a short prediction horizon, overlooking the impact of compounding prediction errors over longer prediction horizons. Strategies to mitigate these cumulative errors remain underexplored. To bridge this gap, in this paper, we study the key design choices for efficiently learning long-horizon prediction dynamics for quadrotors. Specifically, we analyze the impact of multiple architectures, historical  data, and multi-step loss formulation. We show that sequential modeling techniques showcase their advantage in minimizing compounding errors compared to other types of solutions. Furthermore, we propose a novel decoupled dynamics learning approach, which further simplifies the learning process while also enhancing the approach modularity. Extensive experiments and ablation studies on real-world quadrotor data demonstrate the versatility and precision of the proposed approach. Our outcomes offer several insights and methodologies for enhancing long-term predictive accuracy of learned quadrotor dynamics for planning and control.

## Installation
The code is tested with Python 3.8, PyTorch 2.12, and CUDA 11.8.

To install the dependencies, you can create a virtual environment with
```
conda create -n dynamics_learning python=3.8
conda activate dynamics_learning
```
**Note:** Install [Pytorch](https://pytorch.org/) and [Pytorch Lighting](https://lightning.ai/docs/pytorch/stable/starter/installation.html) based on your own system conditions. Here we use Linux and CUDA version 11.8.

## Dependencies

```pip install -r requirements.txt```

## Data Preparation
```
git clone https://github.com/arplaboratory/FW-DYNAMICS_LEARNING.git
cd DYNAMICS_LEARNING
mkdir resources
cd resources
wget *add link to dataset*
```
We first pre-process the dataset and store it in an hdf5 file - 

```
cd ../scripts
python hdf5.py --dataset pi_tcn --history_length 20 --unroll_length 10
```

## Training 
**Note:** Modify the predictor_type flag to specify whether to train the velocity predictor or the attitude predictor.
```
python train.py --batch_size 1024 --model_type tcn --history_length 20 --unroll_length 10 --predictor_type velocity
```

## Evaluation 
```
python eval.py
```

## License
Please be aware that this code was originally implemented for research purposes and may be subject to changes and any fitness for a particular purpose is disclaimed. To inquire about commercial licenses, please contact Pratyaksh Prabhav Rao (pr2257@nyu.edu), Alessandro Saviolo (alessandro.saviolo@nyu.edu), and Prof. Giuseppe Loianno (loiannog@nyu.edu).
```
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
```


