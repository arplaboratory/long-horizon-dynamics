#!/bin/bash
#                  R   B        model_type     encoder_dim  
sbatch job.sbatch  0   65536           mlp             256                 
sbatch job.sbatch  1   65536          lstm             256            
sbatch job.sbatch  2   65536           gru             256              
sbatch job.sbatch  3   65536           tcn             256   
sbatch job.sbatch  4   32768   transformer             256            

       