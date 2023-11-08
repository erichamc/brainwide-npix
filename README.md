# brainwide-npix

The goal of this codebase is to provide a lightweight set of functions for working with data from multiple simultaneously acquired Neuropixels and associated histology data. Relies on output formats and folder structures from [SpikeGLX](https://billkarsh.github.io/SpikeGLX/), [Kilosort](https://github.com/MouseLand/Kilosort) and [Ecephys](https://github.com/jenniferColonell/ecephys_spike_sorting). Uses the [Allen SDK](https://allensdk.readthedocs.io/en/latest/) for core anatomical functionality. Also uses some SpikeGLX utility functionality from the [IBL codebase](https://int-brain-lab.github.io/iblenv/_modules/ibllib/io/spikeglx.htm)

Please see notebook files for examples of:

1) computing and exporting traced histology tracts
2) loading, combining, and working with data from multiple Neuropixels

## Citation

If you use this code, please cite our [paper](https://doi.org/10.1038/s41586-023-06715-z).