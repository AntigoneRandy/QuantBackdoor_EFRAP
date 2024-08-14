# Official PyTorch Implementation of "Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks" (CVPR 2024)

## Overview

This repository contains the official PyTorch implementation required to replicate the primary results presented in the paper "Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks" for CVPR 2024.

## Setup Instructions

This section provides a detailed guide to prepare the environment and execute the project. Please adhere to the steps outlined below.

### 1. Environment Setup

   - **Create a Conda Environment:**  
     Generate a new Conda environment named `efrap` using Python 3.8:
     ```bash
     conda create --name efrap python=3.8
     ```

   - **Activate the Environment:**  
     Activate the newly created environment:
     ```bash
     conda activate efrap
     ```

### 2. Installation of Dependencies

   - **Project Installation:**  
     Navigate to the project's root directory and install it:
     ```bash
     python setup.py install
     ```

   - **Additional Requirements:**  
     Install further required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

## Execution Guidelines

### 1. Prepare the Environment

   - **Navigate to the Project Directory:**  
     Switch to the `main` folder:
     ```bash
     cd ours/main
     ```

   - **Checkpoint Placement:**  
     Download the full-precision model checkpoints (implanted with quantization-conditioned backdoors) from https://www.dropbox.com/scl/fo/pu3ja0djliie0pv70l3b2/h?rlkey=rg1op468jme1lrn7bjnkg06tf&dl=0. 
     Ensure the checkpoint file is stored correctly:
     ```
     ours/main/setting/checkpoint_malicious/pq_cifar_ckpt.pth
     ```

### 2. Run the Project

   - **Execute the Script:**  
     Start the script with the designated template and task:
     ```bash
     python efrap.py --config ../configs/r18_4_4.yaml --choice pq_cifar_fp
     ```

## Some Additional Notes

The primary objective of the activation preservation term in EFRAP is to compensate for benign accuracy after error-guided flipped rounding. Except for the activation MSE loss proposed by Nagel et al., many other alternative losses can be chosen for this purpose, e.g., FlexRound [1], FIM-based Minimization [2], Prediction Difference Metric [3], or any other losses that can improve post-training quantization and are compatible for the 0-1 integer programming optimization. We have experimentally observed that these losses, although originally designed to minimize accuracy loss during quantization, can mitigate the quantization-conditioned backdoors in some cases (but we did not do comprehensive experiments to verify this). It would be interesting to further discover these mechanisms in future works.

> References:
>
> [1]: Lee J H, Kim J, Kwon S J, et al. Flexround: Learnable rounding based on element-wise division for post-training quantization[C]//International Conference on Machine Learning. PMLR, 2023: 18913-18939.
>
> [2]: Li Y, Gong R, Tan X, et al. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction[C]//International Conference on Learning Representations. 2020.
>
> [3]: Liu J, Niu L, Yuan Z, et al. Pd-quant: Post-training quantization based on prediction difference metric[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 24427-24437.


## Acknowledgments

The implementation is heavily based on the MQBench framework, accessible at [MQBench Repository](https://github.com/ModelTC/MQBench).

## Citation

Should this work assist your research, feel free to cite us via:

```
@inproceedings{li2024nearest,
  title={Nearest is not dearest: Towards practical defense against quantization-conditioned backdoor attacks},
  author={Li, Boheng and Cai, Yishuo and Li, Haowei and Xue, Feng and Li, Zhifeng and Li, Yiming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24523--24533},
  year={2024}
}
```
