<img src="https://github.com/iellwood/MatchAndControlPaper/blob/main/Match_and_Control_Image_For_GitHub.jpg" alt="Illustration of the Match-and-Control principle" width="600">

### Short-term Hebbian learning can implement transformer-like attention

#### Ian T. Ellwood, Department of Neurobiology and Behavior, Cornell University

This repository contains the code to implement the model described in the paper, "Short-term Hebbian learning can implement transformer-like attention". While we have made every attempt to make sure that all the files needed to reproduce the paper's figures are available, please contact the author with any questions or concerns about the code or aspects of the model. Some of the files produced by the stimulations are very large. If the complete datafiles are required, a link to a dropbox containing the PyCharm project can be made available upon request.

#### Running the scripts

All the scripts were run in python version 3.8.10 on Ubuntu 20.04.4. Required libraries are `numpy`, `neuron`, `matplotlib`, `scipy`, `torch`, `sklearn` and `pickle`. PyCharm community edition was used for the management of the project and running the scripts. 

All code is dependent on the NEURON simulator (version 8.2) https://www.neuron.yale.edu/neuron/, which must be installed so that your python interpreter can access its libraries through `import neuron`. The `.mod` files in the folder `ChannelModFiles` must be compiled on your local system before they can be used. To do so, run the terminal command `nrnivmodl`, which will only work if NEURON is installed correctly. Unfortunately, depending on your system, the library containing these compiled modules can be in different places and can either be a `.so` or a `.dll` file. You will have to change the line,

`h = HocPythonTools.setup_neuron('../ChannelModFiles/x86_64/.libs/libnrnmech.so')`,

everywhere it appears to have the correct path to the library. To test your installation, run the script `PrintParameters.py`. This script loads the model into NEURON and prints all of the parameters of the model including geometric and electrical properties, as well as all of the ion channel conductances. The expected output of this script is given in `ModelPropertiesPrintout.txt`.

#### Organization of the scripts for figure generation

The code is organized so that generating a figure typically requires calling a script to run a simulation for the required data. Most of the simulation data has been provided in the repository, but several scripts must be run to produce all the data.

1) `PotentiationExample.py` must be run for seeds 2 & 3, each time followed by running `Figure_4BEF_PlotPotentiationExample.py` with the appropriate datafile specified `BasicPotentiationExample_seed_<2 or 3>.obj`
2) `FailedPotentiationExample.py` followed by `Figure_4E_PlotFailedPotentiationExample.py`
3) `SpikeTrainOverlapExample.py` followed by `Figure_2A_PlotBasicOverlapExample.py`
4) `TwoSpinePotentiationExample.py` followed by `Figure_4D_PlotTwoSpinePotentiationExample.py`
5) `CollectControlPhaseOutputs.py` for all match window sizes (0.5, 1, 2) followed by `ProcessControlPhaseTests.py`. Note that the output of `ProcessControlPhaseTests.py` has been provided and that `CollectControlPhaseOutputs.py` can require several days of compute time.

To recreate the data files completely from scratch run, in order

1) `ComputeSpikeDelays.py`
2) `Figure_2B_PlotSpikeDelays.py`
3) `DistributionOfCaIntegrals.py` for all three match window sizes (0.5, 1, 2)
4) `Figure_3BCDE_HistogramAndROCPlots.py`, twice

This will ensure that the files 'time_delays.npz' and 'thresholds.npz', which are used ubiquitously by the project are created.

Note that the folder structure shown in the repository must be present for many of the scripts to work as they will not create missing folders.

Note that the scripts that use `parallel_run.py` can take half a day or more to run, depending on your CPU. Our implementation used 12 processors to run the simulations, but if your system has more cores, you should increase the variable `max_number_of_processes` in our scripts to speed up the computation.


#### Table of simulation scripts, the data files they produce and the scripts that make the figures

| Panel |Simulation Script | Data file | Figure Script |
| ----- | -----------------| ----------| ------------- |
| **2A**    | `SpikeTrainOverlapExample.py` | `ExampleMatchingRun.npz`* | `Figure_2A_PlotBasicOverlapExample.py` |
| **2B**    | `ComputeSpikeDelays.py` | `spike_delay_data.obj`* | `Figure_2B_PlotSpikeDelays.py` (produces figure and `time_delays.npz`)| 
| **2C-D**| `CollectOverlapData.py` | `6Hz_<0.5, 1, or 2>s.npz` & `TEST_DATA_6Hz_<0.5, 1, or 2>s.npz` | `Figure_2CD_FitLinearModelToOverlapsAndPlot.py` |
| **3A**| `CollectCaDataForDifferentOffsets.py` | `offset_data.npz` | `Figure_3A_PlotOffsetData.py` |
|**3B-E**| `DistributionOfCaIntegrals.py` (Once for each match window size)| `6Hz_<0.5, 1 or 2>s.npz` | `Figure_3BCDE_HistogramAndROCPlots.py`|
|**3F-G**| `DistributionOfCaIntegralsWithNoise.py` (Once for each noise std)| `6Hz_1s_<1 or 2>ms_jitter.npz` | `Figure_3FG_HistogramAndROCPlots_Noise.py`|
|**4A** | | | `Figure_4A_PlotThresholdSigma.py`|
|**4B,F**| `PotentiationExample.py` (seeds 2 & 3)| `BasicPotentiationExample_seed_<2 or 3>.obj`* | `Figure_4BF_PlotPotentiationExample.py` (once for each seed)|
|**4E**| `FailedPotentiationExample.py`* | `FailedPotentiationExample_seed_0.py`* | `Figure_4E_PlotFailedPotentiationExample.py`|
|**4C** | `PotentiationThresholdScan.py` | `threshold_scan.npz` | `Figure_4C_PlotPotentiationThresholdScan.py`|
|**4D** | `TwoSpinePotentiationExample.py` |`TwoSpinePotentiationExample.obj`*| `Figure_4D_PlotTwoSpinePotentiationExample.py`|
|**5A-C** | `CollectControlPhaseOutputs.py` (Once for each match window size), `ProcessControlPhaseTests.py`|`failed_and_spurious_spikes.npz`| `Figure_5ABC_successful_and_spurious_spikes.py`|


Data Files with an * are not included in the distribution and must be recomputed.



