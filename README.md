# Qiskit Algorithms V2 - FAdam VQE Custom Implementation

This repository contains a custom implementation for integrating the Fisher Adam (FAdam) optimizer into the Variational Quantum Eigensolver (VQE) algorithm, adapted to Qiskit's latest Primitives API, `EstimatorV2`.

## 1. Project Overview

With the significant update to Qiskit's Primitives API to `EstimatorV2` in Qiskit 1.0+, some functionalities within the existing `qiskit-algorithms` library (specifically `VQE` and `EstimatorGradient`) encountered compatibility issues with the new API. This project addresses these API discrepancies by providing custom implementations and integrations for the following components:

* **Custom VQE Algorithm (`VariationalQuantumEigensolverV2`):** The VQE's energy and gradient evaluation logic has been re-implemented to align with the new `EstimatorV2`'s `run()` signature and result handling (`result[0].data.evs[0]`).
* [cite_start]**Fisher Adam (FAdam) Optimizer (`FAdam`, `FAdamOptimizer`):** The FAdam algorithm, as presented in the relevant research paper[cite: 1], is implemented using NumPy and wrapped to conform to Qiskit's `Optimizer` interface (compatible with `scipy.optimize.OptimizeResult`).
* **Finite Difference-based Gradient Calculator (`FiniteDiffEstimatorGradientV2`):** This component is implemented as an independent callable class, not inheriting from `qiskit-algorithms`'s `BaseEstimatorGradient`. It directly conforms to the `EstimatorV2`'s `run()` signature (`pubs=...`) for performing gradient calculations using the finite difference method.

This implementation aims to leverage the advanced features of Qiskit's `EstimatorV2` while providing a workaround for current version compatibility issues, enabling the application of advanced optimizers like FAdam to quantum algorithms.

## 2. Introduction to Fisher Adam (FAdam)

Fisher Adam (FAdam) is an enhanced optimization algorithm building upon the Adam optimizer's mathematical foundation, re-examined through the lens of Information Geometry and Riemannian Geometry[cite: 1]. [cite_start]According to the paper, Adam approximates Natural Gradient Descent (NGD) by utilizing a diagonal empirical Fisher Information Matrix (FIM).

FAdam proposes several corrections to the original Adam algorithm to address its inherent flaws:

* Enhanced momentum calculations 
* Adjusted bias corrections 
* Adaptive epsilon
* Gradient clipping 
* Refined weight decay term based on a new theoretical framework 

These modifications enable FAdam to demonstrate superior performance across diverse domains such as Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Vector Quantized Variational Autoencoders (VQ-VAE), achieving state-of-the-art results in ASR.

## 3. Project Structure

This repository is structured with the following key files:

* `main_vqe_script.py`: The main script for setting up and executing the VQE algorithm (example filename).
* `vqe_v2.py`: Contains the definition of the `VariationalQuantumEigensolverV2` class (the custom VQE), as well as the `FAdam` and `FAdamOptimizer` class definitions.
* `finite_estimator_gradient_v2.py`: Contains the definition of the `FiniteDiffEstimatorGradientV2` class (the custom gradient calculator).

## 4. Setup and Usage

### 4.1. Conda Environment Setup (Recommended)

It is highly recommended to set up a Conda virtual environment to ensure proper version compatibility among Qiskit and other required libraries.

```bash
# 1. Create a new Conda virtual environment (Python 3.10 or 3.11 recommended)
conda create -n qiskit_fadam_env python=3.10
conda activate qiskit_fadam_env

# 2. Install Qiskit metapackage and necessary libraries
#    This command installs all core Qiskit libraries (qiskit-terra, qiskit-algorithms, qiskit-aer, etc.)
#    at compatible, up-to-date versions.
#    Include `pytorch-cuda=X.Y` for GPU-enabled PyTorch (e.g., 11.8 or 12.1 for CUDA version).
conda install qiskit pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge

# 3. Install other required packages
conda install numpy scipy
```

## 5. Example to use fadam
check FAdam_V1.ipynb, FAdam_V2.ipynb

## Reference
@inproceedings{hwang2024,
  author    = {Dongseong Hwang},
  year      = {2024},
  title     = {FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information},
  booktitle = {arXiv.org},
  doi       = {10.48550/arXiv.2405.12807},
}

