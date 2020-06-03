# Tensor decomposition demo on Neuropixel data

This repository performs a [tensor decomposition](https://www.sciencedirect.com/science/article/pii/S0896627318303878) on Visual Coding Neuropixels data from the Allen Institute. Tensor decomposition is a form of dimensionality reduction that reveals subsets of neurons which are co-active during different time points of the stimulus, modulated by a trial-wise gain factor. The input data to the decomposition is assumed to be structured as a tensor in the format of neurons x time x trials. For the demo, we use the 60 repeats of the "Natural Movie One" stimulus from the functional connectivity experiments (see the [whitepaper](https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/80/75/8075a100-ca64-429a-b39a-569121b612b2/neuropixels_visual_coding_-_white_paper_v10.pdf) for more details).

## Data Preparation
To preprocess the data using the AllenSDK, run the following notebooks:

- [Fix\_NWB_test.ipynb](https://github.com/brianhhu/nm1_repeats/blob/master/Fix_NWB_test.ipynb) (optional)
- [FC_NM1.ipynb](https://github.com/brianhhu/nm1_repeats/blob/master/FC_NM1.ipynb)

This generates numpy arrays storing the neural data, along with csv files that contain the associated metadata in the /processed folder. Optionally, to convert the NWB files to NWB 2.0 (compatible with the latest version of pyNWB), run the first notebook. **Please note this may break some functionality of the AllenSDK (e.g. running speed), which is built using an older version of pyNWB.**

## Tensor Decomposition
To generate the example results, run

```python
python tca_neuropixel.py
```

This generates two plots in the /results folder for each NWB file: 1) a plot of error and similarity as a function of the rank of the decomposition, and 2) a visualization of the neuron, stimulus, and trials factors for a rank 3 decomposition.

## Dependencies
- allensdk==1.7.1
- pynwb==1.3.0
- tensortools==0.3.0
