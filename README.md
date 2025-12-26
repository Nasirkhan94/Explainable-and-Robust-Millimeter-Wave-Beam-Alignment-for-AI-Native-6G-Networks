# Explainable and Robust Millimeter-Wave Beam Alignment for AI-Native 6G Networks

This repository contains the code and data supporting the paper "Explainable and Robust Millimeter Wave Beam Alignment for AI-Native 6G Networks," (Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, and Sinem Coleri ) ICC 2025 - IEEE International Conference on Communications, Montreal, QC, Canada, 2025
> 📄 IEEE Xplore: [https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11161537 ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=11161537) 

## 📁 Repository Contents
### 📂 Main Folders
- **Dataset/** – Data used for training and evaluation  
- **Saved Models/** – Pretrained model checkpoints

### 📄 Usage
  Boston5G scenario considered in this work can be accessed from  https://www.deepmimo.net/
- **Model_Training.py** – Training pipeline for the beam alignment model  
- **testing_withandwithoutnoise.py** – Evaluation script to test performance under noise  
- **beam_utils.py** – Utility functions for beam selection and metrics  
- **Copy of Final_dknn.ipynb** – Deep k-Nearest Neighbors (DkNN) algorithm based credibility assessment and plotting function for reliability diagrams


### 📄 Referencing and Citation

If you use this codebase in any form for academic or industrial research that results in a publication, please cite the corresponding paper.

-You may use the following BibTeX entry for citation:

@INPROCEEDINGS{khan_ICC25,
  author={Khan, Nasir and Abdallah, Asmaa and Celik, Abdulkadir and Eltawil, Ahmed M. and Coleri, Sinem},
  booktitle={ICC 2025 - IEEE International Conference on Communications}, 
  title={Explainable and Robust Millimeter Wave Beam Alignment for AI-Native 6G Networks}, 
  year={2025},
  volume={},
  number={},
  pages={753-758},
  keywords={6G mobile communication;Training;Explainable AI;Millimeter wave measurements;Predictive models;Prediction algorithms;Robustness;MIMO;Received signal strength indicator;Millimeter wave communication;Robustness;eXplainable AI (XAI);mmWave communications;6G networks},
  doi={10.1109/ICC52391.2025.11161537}}
