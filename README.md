# Explainable and Robust Millimeter-Wave Beam Alignment for AI-Native 6G Networks

This repository contains the code and data supporting the paper N. Khan, A. Abdallah, A. Celik, A. M. Eltawil and S. Coleri, "Explainable and Robust Millimeter Wave Beam Alignment for AI-Native 6G Networks," ICC 2025 - IEEE International Conference on Communications, Montreal, QC, Canada, 2025, pp. 753-758, doi: 10.1109/ICC52391.2025.11161537.
> 📄 IEEE Xplore: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11161537 ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11161537) 



## 📁 Repository Contents

Below is an overview of the main files and folders currently in this repository:

### 📂 Main Folders
- **Dataset/** – Data used for training and evaluation  
- **Saved Models/** – Pretrained model checkpoints

### 📄 Python Scripts
- **Model_Training.py** – Training pipeline for the beam alignment model  
- **testing_withandwithoutnoise.py** – Evaluation script to test performance under noise  
- **beam_utils.py** – Utility functions for beam selection and metrics  
- **Copy of Final_dknn.ipynb** – Deep k-Nearest Neighbors (DkNN) algorithm based credibility assessment and plotting function for reliability diagrams


### 📄 Usage
#### Local Installation
If you want to run it locally on your machine, make sure to nstal falconn  and Python3 
1) Boston5G scenario utlized in this work should be accessed from  https://www.deepmimo.net/
2) Use the configuration in parameter.m file to download the channel data
