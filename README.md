<img src="https://github.com/iellwood/MatchAndControlPaper/blob/main/Match_and_Control_Image_For_GitHub.jpg" alt="Illustration of the Match-and-Control principle" width="600">

### Short-term Hebbian learning can implement transformer-like attention

#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code to implement the model described in the paper, "Short-term Hebbian learning can implement transformer-like attention". While we have made every attempt to make sure that all the files needed to reproduce the paper's figures are available, please contact the author with any questions or concerns about the code or aspects of the model. 

All the scripts were run in python version 3.8.10 on Ubuntu 20.04.4. Required libraries are `numpy`, `neuron`, `matplotlib`, `scipy`, `torch` and `pickle`. PyCharm community edition was used for the management of the project and running the scripts. 

All code is dependent on the NEURON simulator (version 8.2) https://www.neuron.yale.edu/neuron/, which must be installed so that your python interpreter can access its
libraries through `import neuron`. The `.mod` files in the folder `ChannelModFiles` must be compiled on your local system before they can be used using the terminal command `nrnivmodl`, which will only work if NEURON is installed correctly. Unfortunately, depending on your system, the library containing these compiled modules can be in different places and can either be a `.so` or a `.dll` file. You will have to change the line

`h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')`

Everywhere it appears to have the correct path to the library. To test your installation, run the script `PrintParameters.py`. This script loads the model into NEURON and prints all of the parameters of the model including geometric and electrical properties, as well as all of the ion channel conductances. The expected output of this script is givenn in `ModelPropertiesPrintout.txt`.

The code is organized so that generating a figure typically requires calling a script to run a simulation for the required data. Most of the simulation 
data has been provided in the repository, but `PotentiationExample.py` must be run for seeds 0-4, each time followed by running `Figure_4BEF_PlotPotentiationExample.py`. Once must also run `SpikeTrainOverlapExample.py` and `TwoSpinePotentiationExample.py`. Each of these simulations
should take around 4 minutes or less on a desktop CPU.

The script `DistributionOFCaIntegrals.py` must be run with different window sizes to reproduce all the data, but we have provided our run in the repository.

Note that the scripts that use `parallel_run.py` can take half a day or more to run, depending on your CPU. Our implementation used 12 processors to run the simulations, but if your system has more cores, you should increase the variable `max_number_of_processes` in our scripts to speed up the computation. 

| Panel |Simulation Script | Data file | Figure Script |
| ----- | -----------------| ----------| ------------- |
| **2A**    | `SpikeTrainOverlapExample.py` | `ExampleMatchingRun.npz`* | `Figure_2A_PlotBasicOverlapExample.py` |
| **2B**    | `ComputeSpikeDelays.py` | `spike_delay_data.obj` | `Figure_2B_PlotSpikeDelays.py` | 
| **2C-D**| `CollectOverlapData.py` | `8Hz_1s.npz` & `TEST_DATA_8Hz_1s.npz` | `Figure_2CD_FitLinearModelToOverlapsAndPlot.py` |
| **3A**| `CollectCaDataForDifferentOffsets.py` | `offset_data.npz` | `Figure_3A_PlotOffsetData.py` |
|**3B-E**| `DistributionOfCaIntegrals.py` | `Ca_Integrals_for_ROC_plots/..` | `Figure_3BCDE_HistogramAndROCPlots.py`|
|**3F-G**| `DistributionOfCaIntegralsWithNoise.py` | `Ca_Integrals_for_ROC_plots/..` | `Figure_3FG_HistogramAndROCPlots_Noise.py`|
|**4A** | | | `Figure_4A_PlotThresholdSigma.py`|
|**4B-F**| `PotentiationExample.py` | `BasicPotentiationExample.py`* | `Figure_4BEF_PlotPotentiationExample.py`|
|**4C** | `PotentiationThresholdScan.py` | `threshold_scan.npz` | `Figure_4C_PlotPotentiationThresholdScan.py`|
|**4D** | `TwoSpinePotentiationExample.py` |`TwoSpinePotentiationExample.obj`*| `Figure_4D_PlotTwoSpinePotentiationExample.py`|

Data Files with an * are not included in the distribution and must be recomputed.

