# Changelog

## 1.1.0 (2024-01-xx)

### **!! Relevant for workflow !!**

#### Config file

- Config file must include parameter `Geometry.maskoption` to indicate how the mask is provided. Use `BM` (Bedmachine) when switching from Laddie 1.0 and/or using Bedmachine input. Other options: `ISOMIP` and `UFEMISM`
- Config file must include parameter `Initialisation.fromrestart`. Set `true` if starting from restart file. Set `false` if starting from scratch
- Parameter `correctisf` moved from `[Options]` to `[Geometry]`. If not moved, default option `Geometry.correctisf = false` will be used.

#### Output

- Output variables U and V are changed to Ut and Vt (to stress that they are on the tgrid)
- Output fields are masked, so melt = NaN outside ice shelf etc

### Speed up

- In total: computation time reduced by a factor of 1.5 (very large domains) to 6 (very small domains)
- Change savefields from xr data-array to np array
- Optimise code in a number of places

### Configuration

- Allow configuration file without unused parameters
- Allow omission of some basic parameters, for which default values are used
- Check whether input parameters are of the correct type and of reasonable value. Otherwise, provide error or warning.
- Allow choice of output variables

### Run continuation

- Allow continuation of old run through config param `Directories.forcenewdir = false`
- This allows storing output in existing directory
- Will also append existing log-file when restarting in existing directory

### Input files

- Increased flexibility for input geometry with several options for input parameter names
- Read and use thickness and surface only if draft is not provided
- Can choose typical input styles for mask: `Geometry.maskoption = ['ISOMIP','BM','UFEMISM']`
- Included some checks on input forcing

### Forcing

- Can set `Forcing.drho0` for tanh-forcing in config file. Default = 0.01 (was hardcoded in previous version).
- Included pre-run interpolation to 1m vertical grid
- **Note: 3D forcing does not work yet**

### Code structure

- Removed unused functions (constants.py, plotfunctions.py, physics/convT2)
- Moved some functions around and split some functions for clarity
- Also included a lot of comments for code readability

### Fixes

- Fixed a bug in `remove_icebergs` related to mask definition

### Log file

- Run time included
- Also spitting out warnings when using default values. Will probably change this
- Continous log file when restarting a run within the same directory
