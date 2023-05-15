## Short-term Hebbian learning can implement transformer-like attention

### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code to implement the model described in the paper. All code is dependent on the NEURON
simulator https://www.neuron.yale.edu/neuron/, which must be installed so that your python interpreter can access its
libraries through `import neuron`. The `.mod` files in the folder `ChannelModFiles` must be compiled on your local system before they can be used using the terminal command `nrnivmodl`, which will only work if NEURON is installed correctly. Unfortunately, depending on your system, the library containing these compiled modules can be in different places and can either be a `.so` or a `.dll` file. You will have to change the line

`h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')`

Everywhere it appears to have the correct path to the library.


