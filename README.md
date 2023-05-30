<img src="https://github.com/iellwood/MatchAndControlPaper/blob/main/Match_and_Control_Image_For_GitHub.jpg" alt="Illustration of the Match-and-Control principle" width="600">

### Short-term Hebbian learning can implement transformer-like attention

#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code to implement the model described in the paper, "Short-term Hebbian learning can implement transformer-like attention". While we have made every attempt to make sure that all the files needed to reproduce the paper's figures are available, please contact the author with any questions or concerns about the code or aspects of the model. 

#### Running the scripts

All the scripts were run in python version 3.8.10 on Ubuntu 20.04.4. Required libraries are `numpy`, `neuron`, `matplotlib`, `scipy`, `torch`, `sklearn` and `pickle`. PyCharm community edition was used for the management of the project and running the scripts. 

All code is dependent on the NEURON simulator (version 8.2) https://www.neuron.yale.edu/neuron/, which must be installed so that your python interpreter can access its
libraries through `import neuron`. The `.mod` files in the folder `ChannelModFiles` must be compiled on your local system before they can be used. To do so, run the terminal command `nrnivmodl`, which will only work if NEURON is installed correctly. Unfortunately, depending on your system, the library containing these compiled modules can be in different places and can either be a `.so` or a `.dll` file. You will have to change the line,

`h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')`,

everywhere it appears to have the correct path to the library. To test your installation, run the script `PrintParameters.py`. This script loads the model into NEURON and prints all of the parameters of the model including geometric and electrical properties, as well as all of the ion channel conductances. The expected output of this script is given in `ModelPropertiesPrintout.txt`.

#### Organization of the scripts for figure generation

The code is organized so that generating a figure typically requires calling a script to run a simulation for the required data. Most of the simulation data has been provided in the repository, but `PotentiationExample.py` must be run for seeds 0-4, each time followed by running `Figure_4BEF_PlotPotentiationExample.py`. Once must also run `SpikeTrainOverlapExample.py` and `TwoSpinePotentiationExample.py`. Each of these simulations should take around 4 minutes or less on a desktop CPU.

The script `Figure_2B_PlotSpikeDelays.py` also includes the routines that compute the times when the backpropagated action potentials reach each spine and saves this data in `time_delays.npz`, which is used in many of the scripts. We have included it in the repository, but if you wish to reproduce it, note that you will have to run both scripts for Figure 2B in the table below. Many of the scripts also utilize `thresholds.npz`, which is produced by `Figure_3BCDE_HistogramAndROCPlots.py` and which we have included. These thresholds are used to normalize the calcium integrals so that they can be compared in a meaningful way across match window sizes.

The script `DistributionOFCaIntegrals.py` and `DistributionOFCaIntegralsWithNoise.py` must be run with different window sizes or noise widths to reproduce all the data, but we have provided our run in the repository. Similarly `CollectControlPhaseOutputs.py` must be run for all three windows and then `ProcessControlPhaseTests.py` must be run. We have only included the output of `ProcessControlPhaseTests.py` in the repository, `ControlPhaseTests/failed_and_spurious_spikes.npz`.

Note that the scripts that use `parallel_run.py` can take half a day or more to run, depending on your CPU. Our implementation used 12 processors to run the simulations, but if your system has more cores, you should increase the variable `max_number_of_processes` in our scripts to speed up the computation.

Finally, some of the files are named `8Hz...`, but this is a legacy naming scheme. All runs were performed at 6 Hz.



#### Table of simulation scripts, the data files they produce and the scripts that make the figures

| Panel |Simulation Script | Data file | Figure Script |
| ----- | -----------------| ----------| ------------- |
| **2A**    | `SpikeTrainOverlapExample.py` | `ExampleMatchingRun.npz`* | `Figure_2A_PlotBasicOverlapExample.py` |
| **2B**    | `ComputeSpikeDelays.py` | `spike_delay_data.obj` | `Figure_2B_PlotSpikeDelays.py` | 
| **2C-D**| `CollectOverlapData.py` | `8Hz_1s.npz` & `TEST_DATA_8Hz_1s.npz` | `Figure_2CD_FitLinearModelToOverlapsAndPlot.py` |
| **3A**| `CollectCaDataForDifferentOffsets.py` | `offset_data.npz` | `Figure_3A_PlotOffsetData.py` |
|**3B-E**| `DistributionOfCaIntegrals.py` | `Ca_Integrals_for_ROC_plots/..` | `Figure_3BCDE_HistogramAndROCPlots.py`|
|**3F-G**| `DistributionOfCaIntegralsWithNoise.py` | `Ca_Integrals_for_ROC_plots/..` | `Figure_3FG_HistogramAndROCPlots_Noise.py`|
|**4A** | | | `Figure_4A_PlotThresholdSigma.py`|
|**4B,F**| `PotentiationExample.py` | `BasicPotentiationExample_seed_[2 or 3].obj`* | `Figure_4BF_PlotPotentiationExample.py`|
|**4E**| `FailedPotentiationExample.py`* | `FailedPotentiationExample_seed_0.py`* | `Figure_4E_PlotFailedPotentiationExample.py`|
|**4C** | `PotentiationThresholdScan.py` | `threshold_scan.npz` | `Figure_4C_PlotPotentiationThresholdScan.py`|
|**4D** | `TwoSpinePotentiationExample.py` |`TwoSpinePotentiationExample.obj`*| `Figure_4D_PlotTwoSpinePotentiationExample.py`|
|**4D** | `TwoSpinePotentiationExample.py` |`TwoSpinePotentiationExample.obj`*| `Figure_4D_PlotTwoSpinePotentiationExample.py`|
|**5A-C** | `CollectControlPhaseOutputs.py`, `ProcessControlPhaseTests.py`|`ControlPhaseTests/failed_and_spurious_spikes.npz`*| `Figure_5ABC_successful_and_spurious_spikes.py`|


Data Files with an * are not included in the distribution and must be recomputed.

