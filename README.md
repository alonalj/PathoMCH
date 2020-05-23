
# Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention

This project contains the source code used in: "XX" by Levy-Jurgenson et al.

### Prerequisites 

Python 3.6 

Tensorflow 1.4.0 (pip3 install tensorflow==1.14.0 or on gpu: pip3 install tensorflow-gpu==1.14.0)

On Google Cloud platform, we specifically used the following settings (adapted to tensorflow 1.14.0):

For training on GPUs:
Operating system: "Deep Learning on Linux" 
Version: "Deep Learning Image: TensorFlow 1.15.0 m42" (uninstall 1.15 and install 1.14 instead)
Number of K80 GPUs per vm: 8

For preprocessing (tiling WSIs and creating tfrecords) we used the same settings above, with no GPUs, but with 60 CPUs.
Less CPUs can be used by changing NUM_CPU in conf.py but this will slow down preprocessing significantly.  

Below we cover: 
(1) Overview of src files
(2) Overview of folders and where to place your data

## Overview of src files

The files under src are deliberately in a flattened hierarchy to keep things simple when transitioning between a 
local environment and linux servers. 

There are XX main files that are of interest, the rest are used by them:

##### conf.py
Contains Conf object  
##### preprocess.py
This script takes slides under data/


## Overview of src files

# TODO: make sure to run from src