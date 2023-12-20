# Changelog

## 1.1.0 (2023-12-xx)

### **!! Relevant for workflow !!**

- To restart, must include config parameter `Initialisation.fromrestart = true`
- Output variables U and V are changed to Ut and Vt (to note that they are on the tgrid)

### Speed up

- Change savefields from xr data-array to np array. Reduced computation time by approx 25%

### Configuration

- Allow configuration file without unused parameters
- Allow omission of some basic parameters, for which default values are used
- Check whether input parameters are of the correct type and of reasonable value. Otherwise, provide error or warning.

### Run continuation

- Allow continuation of old run through config param `Directories.forcenewdir = true`
- This allows storing output in existing directory
- Will also append existing log-file

### Input files

- Increased flexibility for input geometry with several options for input parameter names
- Read and use thickness and surface only if draft is not provided
- Can choose typical input styles for mask: `Geometry.maskoption = ['ISOMIP','BM','UFEMISM']`

### Forcing

- Can set `Forcing.drho0` for tanh-forcing in config file. Default = 0.01 (was hardcoded in previous version).

### Code structure

- Removed unused functions (constants.py, plotfunctions.py, physics/convT2)
- Moved some functions around and split some functions for clarity
- Also included a lot of comments for code readability

### Fixes

- Fixed a bug in `remove_icebergs` related to mask definition

### Log file

- Run time included
- Also spitting out warnings when using default. Will probably change this