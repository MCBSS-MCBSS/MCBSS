# Introduction
The source code and models for our paper **Mitigating Class Bias in Sample Selection for Imbalanced Noisy Data**
# Framework
![Our Framework](https://github.com/MCBSS-MCBSS/MCBSS/blob/main/mainfig.png)
# Installation
After creating a virtual environment of python 3.7
# How to use
The code is currently tested only on GPU.
* Data preparation  
  Created a folder `data` and download `cifar10`/`cifar100`/`web-aircraft`/`web-bird`/`web-car`/`Food101N` dataset into this folder.
* Source code
  * If you want to train the whole model from beginning using the source code, please follow subsequent steps:
    *  Prepare data
    *  Modify GPU device in the corresponding train script `xxx.sh` in `scripts` folder
    *  Activate virtual environment (e.g. conda) and then run    
      `bash xxx.sh` 

