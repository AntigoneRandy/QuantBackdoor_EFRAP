# Code for Reproduction for "Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks"

## Description:

This repo consists of code for reproducing the main results of "Nearest is Not Dearest: Towards Practical Defense against Quantization-conditioned Backdoor Attacks. (CVPR 2024)" The paper introduces a novel defense method that effectively guards against the quantization process against quantization-conditioned backdoor attacks.

## Steps:

1. Download the full-precision model checkpoints (implanted with quantization-conditioned backdoors) from https://www.dropbox.com/scl/fo/pu3ja0djliie0pv70l3b2/h?rlkey=rg1op468jme1lrn7bjnkg06tf&dl=0. The model quantized using EFRAP could be downloaded from https://www.dropbox.com/scl/fo/ty9en93t5oy6n280sy63u/h?rlkey=3v0ctg7uabnci59oko8858xbw&dl=0. and place them in this folder.

2. Install Python 3.8 and the necessary dependencies with the command ``pip install -r requirements.txt``.

3. Download the corresponding datasets (CIFAR-10 and Tiny-ImageNet) and place them in ./datasets.

4. Run the code. (see detailed examples below) You may use cuda to accelerate.
   evaluate malicious full-precision model:
   e.g. `python main.py --config ca_cifar_fp_8 --bit 8` to reproduce standard quantization results on comprartifact on cifar10 with 8-bit quantization.

   evaluate models quantized by EFRAP:
   e.g. `python main.py --config ca_cifar_ours_8 --bit 8` to reproduce EFRAP quantization results on comprartifact on cifar10 with 8-bit quantization.

In this repository, we first release part of our codes and models for demonstration and validation purposes. You may run the aforementioned commands to reproduce our results. Note that the reproduced results may not be perfectly aligned with those in the paper due to the following two reasons:

      1. For simplicity, in this repository, all activations are in FP32 in evaluating EFRAP. In our main paper, all activations are also quantized with the same bandwidth of weights.
      2. We repeat each experiment several times and report average results in our main paper.
         We will release our entire code as well as model checkpoints upon publication (see Reproducibility Statements in Appendix J).
            We provide log files during our experiments (./exp_logs) to facilitate reproduction.

> All options (ensure you used corresponding bandwidths):
> 'pq_cifar_fp', 'qu_cifar_fp', 'ca_cifar_fp_8', 'ca_cifar_fp_4', 'pq_tiny_fp', 'qu_tiny_fp', 'ca_tiny_fp_8', 'ca_tiny_fp_4', 'pq_cifar_ours_8', 'pq_cifar_ours_4', 'qu_cifar_ours_8', 'qu_cifar_ours_4', 'ca_cifar_ours_8', 'ca_cifar_ours_4', 'pq_tiny_ours_8', 'pq_tiny_ours_4', 'qu_tiny_ours_8', 'qu_tiny_ours_4', 'ca_tiny_ours_8', 'ca_tiny_ours_4'

## Our reproduction results on CIFAR10 as references:

The machines and configurations are the same in the Implementation Details in the Appendix.

**CompArtifact:**

```
$ python main.py --config ca_cifar_fp_8 --bit 8
before quantization
cda
Test: [ 0/79]   Time  1.850 ( 1.850)    Acc@1  90.62 ( 90.62)   Acc@5 100.00 (100.00)
 * Acc@1 91.360 Acc@5 99.640
asr_no_targets
Test: [ 0/71]   Time  0.985 ( 0.985)    Acc@1   2.34 (  2.34)   Acc@5  98.44 ( 98.44)
 * Acc@1 1.256 Acc@5 99.756
after quantization
cda
Test: [ 0/79]   Time  1.014 ( 1.014)    Acc@1  91.41 ( 91.41)   Acc@5 100.00 (100.00)
 * Acc@1 88.940 Acc@5 99.560
asr_no_targets
Test: [ 0/71]   Time  0.993 ( 0.993)    Acc@1  99.22 ( 99.22)   Acc@5 100.00 (100.00)
 * Acc@1 99.900 Acc@5 100.000
```

```
$ python main.py --config ca_cifar_fp_4 --bit 4
before quantization
cda
Test: [ 0/79]   Time  1.818 ( 1.818)    Acc@1  94.53 ( 94.53)   Acc@5  99.22 ( 99.22)
 * Acc@1 93.440 Acc@5 99.770
asr_no_targets
Test: [ 0/71]   Time  1.006 ( 1.006)    Acc@1   0.00 (  0.00)   Acc@5  64.84 ( 64.84)
 * Acc@1 0.522 Acc@5 60.122
after quantization
cda
Test: [ 0/79]   Time  0.990 ( 0.990)    Acc@1  92.97 ( 92.97)   Acc@5  99.22 ( 99.22)
 * Acc@1 89.530 Acc@5 99.490
asr_no_targets
Test: [ 0/71]   Time  1.073 ( 1.073)    Acc@1  99.22 ( 99.22)   Acc@5 100.00 (100.00)
 * Acc@1 99.678 Acc@5 99.978
```

```
$ python main.py --config ca_cifar_ours_8 --bit 8
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.847 ( 1.847)    Acc@1  90.62 ( 90.62)   Acc@5 100.00 (100.00)
 * Acc@1 91.370 Acc@5 99.650
asr_no_targets
Test: [ 0/71]   Time  1.009 ( 1.009)    Acc@1   1.56 (  1.56)   Acc@5 100.00 (100.00)
 * Acc@1 1.122 Acc@5 99.611
```

```
$ python main.py --config ca_cifar_ours_4 --bit 4
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.857 ( 1.857)    Acc@1  95.31 ( 95.31)   Acc@5  99.22 ( 99.22)
 * Acc@1 92.660 Acc@5 99.770
asr_no_targets
Test: [ 0/71]   Time  1.056 ( 1.056)    Acc@1   0.00 (  0.00)   Acc@5  61.72 ( 61.72)
 * Acc@1 0.556 Acc@5 48.400
```

**Qu-ANTI-zation:**

```
$ python main.py --config qu_cifar_fp --bit 8
before quantization
cda
Test: [ 0/79]   Time  1.801 ( 1.801)    Acc@1  94.53 ( 94.53)   Acc@5 100.00 (100.00)
 * Acc@1 93.300 Acc@5 99.780
asr_no_targets
Test: [ 0/71]   Time  1.179 ( 1.179)    Acc@1   3.12 (  3.12)   Acc@5  99.22 ( 99.22)
 * Acc@1 2.178 Acc@5 96.800
after quantization
cda
Test: [ 0/79]   Time  1.303 ( 1.303)    Acc@1  93.75 ( 93.75)   Acc@5 100.00 (100.00)
 * Acc@1 91.800 Acc@5 99.730
asr_no_targets
Test: [ 0/71]   Time  1.163 ( 1.163)    Acc@1  99.22 ( 99.22)   Acc@5 100.00 (100.00)
 * Acc@1 99.089 Acc@5 100.000
```

```
$ python main.py --config qu_cifar_fp --bit 4
before quantization
cda
Test: [ 0/79]   Time  1.809 ( 1.809)    Acc@1  94.53 ( 94.53)   Acc@5 100.00 (100.00)
 * Acc@1 93.300 Acc@5 99.780
asr_no_targets
Test: [ 0/71]   Time  1.134 ( 1.134)    Acc@1   1.56 (  1.56)   Acc@5  97.66 ( 97.66)
 * Acc@1 2.178 Acc@5 96.800
after quantization
cda
Test: [ 0/79]   Time  1.171 ( 1.171)    Acc@1  89.06 ( 89.06)   Acc@5  99.22 ( 99.22)
 * Acc@1 88.280 Acc@5 99.240
asr_no_targets
Test: [ 0/71]   Time  1.154 ( 1.154)    Acc@1 100.00 (100.00)   Acc@5 100.00 (100.00)
 * Acc@1 100.000 Acc@5 100.000
```

```
$ python main.py --config qu_cifar_ours_8 --bit 8
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.784 ( 1.784)    Acc@1  94.53 ( 94.53)   Acc@5 100.00 (100.00)
 * Acc@1 93.350 Acc@5 99.780
asr_no_targets
Test: [ 0/71]   Time  0.960 ( 0.960)    Acc@1   0.78 (  0.78)   Acc@5  94.53 ( 94.53)
 * Acc@1 1.044 Acc@5 94.200
```

```
$ python main.py --config qu_cifar_ours_4 --bit 4
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.849 ( 1.849)    Acc@1  92.97 ( 92.97)   Acc@5 100.00 (100.00)
 * Acc@1 92.970 Acc@5 99.800
asr_no_targets
Test: [ 0/71]   Time  1.030 ( 1.030)    Acc@1   0.00 (  0.00)   Acc@5  78.91 ( 78.91)
 * Acc@1 0.578 Acc@5 78.344
```

**PQBackdoor:**

```
$ python main.py --config pq_cifar_fp --bit 8
before quantization
cda
Test: [ 0/79]   Time  1.887 ( 1.887)    Acc@1  84.38 ( 84.38)   Acc@5  98.44 ( 98.44)
 * Acc@1 86.590 Acc@5 99.200
asr_no_targets
Test: [ 0/71]   Time  0.986 ( 0.986)    Acc@1   4.69 (  4.69)   Acc@5  67.97 ( 67.97)
 * Acc@1 2.144 Acc@5 69.222
after quantization
cda
Test: [ 0/79]   Time  1.025 ( 1.025)    Acc@1  82.81 ( 82.81)   Acc@5  99.22 ( 99.22)
 * Acc@1 86.030 Acc@5 99.250
asr_no_targets
Test: [ 0/71]   Time  1.015 ( 1.015)    Acc@1 100.00 (100.00)   Acc@5 100.00 (100.00)
 * Acc@1 99.133 Acc@5 99.989
```

```
$ python main.py --config pq_cifar_fp --bit 4
before quantization
cda
Test: [ 0/79]   Time  1.816 ( 1.816)    Acc@1  84.38 ( 84.38)   Acc@5  98.44 ( 98.44)
 * Acc@1 86.590 Acc@5 99.200
asr_no_targets
Test: [ 0/71]   Time  1.012 ( 1.012)    Acc@1   1.56 (  1.56)   Acc@5  75.00 ( 75.00)
 * Acc@1 2.144 Acc@5 69.222
after quantization
cda
Test: [ 0/79]   Time  1.017 ( 1.017)    Acc@1  75.00 ( 75.00)   Acc@5  96.09 ( 96.09)
 * Acc@1 79.130 Acc@5 98.360
asr_no_targets
Test: [ 0/71]   Time  1.005 ( 1.005)    Acc@1  98.44 ( 98.44)   Acc@5 100.00 (100.00)
 * Acc@1 98.878 Acc@5 100.000
```

```
$ python main.py --config pq_cifar_ours_8 --bit 8
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.826 ( 1.826)    Acc@1  85.16 ( 85.16)   Acc@5  98.44 ( 98.44)
 * Acc@1 86.530 Acc@5 99.190
asr_no_targets
Test: [ 0/71]   Time  1.056 ( 1.056)    Acc@1   3.12 (  3.12)   Acc@5  68.75 ( 68.75)
 * Acc@1 2.067 Acc@5 68.722
```

```
$ python main.py --config pq_cifar_ours_4 --bit 4
selected model has already been quantized by EFRAP
after quantization
cda
Test: [ 0/79]   Time  1.805 ( 1.805)    Acc@1  83.59 ( 83.59)   Acc@5  98.44 ( 98.44)
 * Acc@1 86.280 Acc@5 99.180
asr_no_targets
Test: [ 0/71]   Time  1.225 ( 1.225)    Acc@1   2.34 (  2.34)   Acc@5  42.97 ( 42.97)
 * Acc@1 1.289 Acc@5 46.900
```

## Some Additional Notes

The primary objective of the activation preservation term in EFRAP is to compensate for benign accuracy after error-guided flipped rounding. Except for the activation MSE loss by Nagel et al., other many alternative losses can be chosen for this purpose, e.g., FlexRound (Lee et al.), FIM-based Minimization (Li et al.), Prediction Difference Metric (Liu et al.), or any other losses that can improve post-training quantization and is compatible for the 0-1 interger programming optimization. We have experimentally observed that these losses, although originally designed for mitigating accuracy loss during quantization, can eliminate the quantization-conditioned backdoors in some cases (but we did not do an comprehensive experiments to verify this). It would be interesting to further discover these machanisms in future works.

References:

Lee J H, Kim J, Kwon S J, et al. Flexround: Learnable rounding based on element-wise division for post-training quantization[C]//International Conference on Machine Learning. PMLR, 2023: 18913-18939.

Li Y, Gong R, Tan X, et al. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction[C]//International Conference on Learning Representations. 2020.

Liu J, Niu L, Yuan Z, et al. Pd-quant: Post-training quantization based on prediction difference metric[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 24427-24437.
