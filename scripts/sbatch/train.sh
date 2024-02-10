#!/bin/bash
#                  R   B        model_type     unroll_length      encoder_sizes      decoder_sizes
sbatch job.sbatch  0   512            mlp             1            '512'            '256,128,128'
sbatch job.sbatch  1   512            mlp             5            '512'            '256,128,128'
sbatch job.sbatch  2   512            mlp             10           '512'            '256,128,128'
sbatch job.sbatch  3   512           lstm             1            '256'            '256,128,128'
sbatch job.sbatch  4   512           lstm             5            '256'            '256,128,128'
sbatch job.sbatch  5   512           lstm            10            '256'            '256,128,128'
sbatch job.sbatch  6   512           gru              1            '256'            '256,128,128'
sbatch job.sbatch  7   512           gru              5            '256'            '256,128,128'
sbatch job.sbatch  8   512           gru             10            '256'            '256,128,128'
sbatch job.sbatch  9   512           tcn              1            '256,128,128'    '256,128,128'
sbatch job.sbatch  10  512           tcn              5            '256,128,128'    '256,128,128'
sbatch job.sbatch  11  512           tcn             10            '256,128,128'    '256,128,128'
sbatch job.sbatch  12  512           transformer      1            '256'            '256,128,128'
sbatch job.sbatch  13  512           transformer      5            '1024'           '256,128,128'
sbatch job.sbatch  14  512           transformer     10            '1024'           '256,128,128'