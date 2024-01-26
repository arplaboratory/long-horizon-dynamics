#!/bin/bash
#                  R   B        model_type     unroll_length      encoder_sizes      decoder_sizes
sbatch job.sbatch  0   512           mlp             1            '512,256,256'   '2048,1024,1024'
sbatch job.sbatch  1   512           mlp             10           '512,256,256'   '2048,1024,1024'
sbatch job.sbatch  2   512           mlp             20           '512,256,256'   '2048,1024,1024'
sbatch job.sbatch  3   512           lstm             1            '512,256,256'    '1024,512,256'
sbatch job.sbatch  4   512           lstm            10            '512,256,256'    '1024,512,256'
sbatch job.sbatch  5   512           lstm            20            '512,256,256'    '1024,512,256'
sbatch job.sbatch  6   512           gru             1             '512,256,256'    '1024,512,256'
sbatch job.sbatch  7   512           gru             10             '512,256,256'    '1024,512,256'
sbatch job.sbatch  8   512           gru             20             '512,256,256'    '1024,512,256'
sbatch job.sbatch  9   512           tcn             1             '1024,512,512'    '1024,512,256'
sbatch job.sbatch  10  512           tcn             10            '1024,512,512'    '1024,512,256'
sbatch job.sbatch  11  512           tcn             20            '1024,512,512'    '1024,512,256'
sbatch job.sbatch  12  512           transformer      1            '1024,512,512'    '1024,512,256'
sbatch job.sbatch  13  512           transformer     10            '1024,512,512'    '1024,512,256'
sbatch job.sbatch  14  512           transformer     20            '1024,512,512'    '1024,512,256'
             
   