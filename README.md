# deepsic-official

A minimalistic python implementation of the Sequntial DeepSIC algorithm, published in:

Shlezinger, N., Fu, R. and Eldar, Y.C., 2020. [DeepSIC: Deep soft interference cancellation for multiuser MIMO detection](https://arxiv.org/pdf/2002.03214.pdf). IEEE Transactions on Wireless Communications, 20(2), pp.1349-1362.

DeepSIC is a deep learning architecture for MIMO symbol detection that integrating deep neural networks into the SIC algorithm. The paper shows that DeepSIC is able to track time-varying channels in a data-driven manner, in the case of complex channel models or no channel state information at all. The sequential version refers to the training: it is done in a sequential manner, layer-by-layer.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [channel](#channel)
    + [detectors](#detectors)
    + [utils](#utils)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)
  
  # Introduction

Note that this python implementation deviates from the [basic one](https://arxiv.org/pdf/2002.03214.pdf) in the basic DNN implementaion: Here it is only (1) Linear 1X64, (2) ReLU, (3) Linear 64X64, as  opposed to the three layers in the paper for the sequential version. Also, the learning rate is 5e-3 instead of 1e-2. Note that these changes obtain even better results on this setup, than the ones reported in the paper. These hyperparameters should be chosen to fit the complexity of the simulated channel. 

Also, note that the simulated setup here is a sequential transmission of pilots + info in each block coherence, which is different than the one in the original paper (that has only unlabeled info bits and uses error correction codes to correct errors and train on the decoded packet). It is more convenient in our opinion for learning purposes. 

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: symbols generation, channel transmission and detection.

### channel 

Includes the symbols generation and transmission part, up to the creation of the dataset composed of (transmitted, received) tuples in the channel_dataset wrapper class. The modulation is done in the modulator file.

### detectors 

Includes the next files:

(1) The backbone trainer.py which holds the most basic functions, including the network initialization and the sequential transmission in the channel and BER calculation. The trainer is a wrapper for the training and evaluation of the detector. Trainer holds the training, sequential evaluation of pilot + info blocks. It also holds the main function 'eval' that trains the detector and evaluates it, returning a list of coded ber/ser per block.

(2) The DeepSIC trainer, which focuses on the online sequential training part (layer-by-layer). Refer to Algorithm 3 in the paper for more details.

(3) The DeepSIC detector, which is the basic cell that runs the priors through the deep neural network, and later propagates these values through the iterations.

### utils

Extra utils for calculating the accuracy over the BER metric; several constants; and the config singleton class.
The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar.

The config is accessible from every module in the package, featuring the next parameters:
1. seed - random number generator seed. Integer.
2. fading_in_channel - whether to use fading. Relevant only to the synthetic channel. Boolean flag.
3. snr - signal-to-noise ratio, determines the variance properties of the noise, in dB. Float.
4. block_length - number of coherence block bits, total size of pilot + data. Integer.
5. pilot_size - number of pilot bits. Integer.
6. blocks_num - number of blocks in the tranmission. Integer.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You need to run the evaluation.py file.

This code was simulated with GeForce RTX 3060 with CUDA 12. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f deepsic_env.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\deepsic_env\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!



l
