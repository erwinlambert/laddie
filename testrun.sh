#!/bin/bash
#SBATCH --job-name=lad_ant16
#SBATCH --output=ladase-%J.out
#SBATCH --error=ladase-%J.out
#SBATCH --qos=np
#SBATCH --time=01:00:00

python3 runladdie.py config_ant16.toml 
