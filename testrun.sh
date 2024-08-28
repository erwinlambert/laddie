#!/bin/bash
#SBATCH --job-name=lad_ase
#SBATCH --output=ladase-%J.out
#SBATCH --error=ladase-%J.out
#SBATCH --qos=np
#SBATCH --time=00:15:00

python3 runladdie.py config_ASE.toml 
