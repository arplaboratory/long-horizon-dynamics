#!/bin/bash
#                  R   B        model_type     unroll_length  
sbatch job.sbatch  0   512           tcn             20                 
sbatch job.sbatch  1   512           tcn             25
sbatch job.sbatch  2   512           tcn             30
sbatch job.sbatch  3   512           tcn             35
sbatch job.sbatch  4   512           gru             20                
sbatch job.sbatch  5   512           gru             25
sbatch job.sbatch  6   512           gru             30
sbatch job.sbatch  7   512           gru             35       