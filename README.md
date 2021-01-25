# Cheetah: a computational toolkit for cybergenetic control

Cheetah is a simple to use Python library to support the development of analysis workflows and real-time cybergenetic control platforms that combine microscopy imaging and environmental control. The toolkit relies on a U-Net convolutional neural network to perform cell segmentation and uses this with additional functions that can label, track and generate control signals based on this information. If you make use of Cheetah in your work we ask that you cite the following paper:

> [**Cheetah: a computational toolkit for cybergenetic control.** Pedone E., de Cesare I., Rubio Denniss A., Zamora C., Haener D., Postiglione L., Shannon B., Savery N., Grierson C.S., di Barnardo M., Gorochowski T.E. & Marucci M., bioRxiv (2020).](https://www.biorxiv.org/content/10.1101/2020.06.25.171751v2)

## Getting started

If you would like to use Cheetah for your own project, we recommend using Anaconda Python and creating a separate environment will all the necessary dependancies. This can be done after install Anaconda by running the following command:

`conda create -n py36 python=3.6
conda activate py36
conda install keras==2.0.8 matplotlib numpy scipy scikit-image scikit-learn`

To get started, we also recommend taking a look at some of the examples provided and reading the associated publication to better understand how all aspects of the functionality can be used.

## Dependences

keras, tensorflow, scikit-learn, scikit-image, scipy, glob, numpy, matplotlib, json
