#!/bin/bash
#                  R   B        model_type     unroll_length  delta
sbatch job.sbatch  0   512           gru             1         True  
sbatch job.sbatch  1   512           gru             20        True
sbatch job.sbatch  2   512           gru             20        True
sbatch job.sbatch  3   512           gru             1         False
sbatch job.sbatch  4   512           gru             20        False
sbatch job.sbatch  5   512           gru             20        False