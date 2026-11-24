#!/bin/bash
#SBATCH -J TIMM_Closure_Eval
#SBATCH -N 1-1
#SBATCH -n 4
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -p l40s-gcondo
#SBATCH --comment="aria"

sh $1
