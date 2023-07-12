This is a guide to get started with LADDIE

Do get in contact if you run into problems!
erwin.lambert [at] knmi [dot] nl

## Brief summary:
- Copy the repository to your local machine
- Install python and necessary packages as specified in yml files
- Prepare input files and config file
- Run the model from command line `python3 runladdie.py config.toml`

## Installation

## Set up

### Config file

### Directories

### Input files

## Run model

### Command

### Track run
While the model is running, you can track it by opening the file `log.txt` in the output directory of your run. At a given time interval, specified by the configuration variable `diagday`, diagnostics are written out. These help you identifying possible instabilities, and help assess the equilibration to steady state of the run.

The diagnostics that are written out are:
time || Average thickness | Minimum thickness | Maximum thickness || Average melt | Maximum melt || Meltwater Fraction of cavity outflow || Total entrainment + Total additional entrainment due to D_min + Total detrainment | Total overturning circulation || Maximum velocity || Minimum stratification | Number of grid cells with convection


## Analyse output

### Contents
