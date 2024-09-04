# Changelog

## 1.1.1 (2024-09-04)

Extended options for ice sheet model coupling + internal routine to cut out a minimum domain

### **!! Relevant for workflow !!**

- Everything should work as before, but some pre-processing steps can be skipped like preventing periodic boundary conditions or copying output files

### Cut domain

- Included a routine to cut out a minimum domain by removing borders without ice shelf points.
- Allows for optimal runtime and more flexibility in defining input regions.
- Set in config file: `[Geometry] cutdomain` to `true`. Default: `true`.

### Boundary conditions

- Removed periodic boundary conditions. Replaced with option for each NSEW boundary: open ocean or closed (grounded).
- Define in config file: `[Options] border_N` etc. Default for each boundary: 1 (grounded).

### Input options

- Included option to run on IMAUICE output

### Output options

- Included option to save a file with BMB data from the latest time period. Save by setting `[BMB] save_BMB` to `true`.
- Now: two variables `BMB` (converted to m.i.e./yr; negative = mass loss = net melting) and `BMBext` (same as BMB, but extrapolated 1 grid cell below the grounded ice).
- Option to extend the domain of the BMB output file by N grid cells (defiled by `[BMB] bordercells`). To be used for UFEMISM forcing. Default: 0
- Option to spit out a file `laddieready` to signal the run is finished. Toggle by setting `[BMB] create_readyfile` to `true`. Default: `false`

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
