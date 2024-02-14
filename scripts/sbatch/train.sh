#!/bin/bash
#                  R   B        model_type     unroll_length      encoder_sizes      decoder_sizes      seed
sbatch job.sbatch  0   2048            mlp             1            '512'            '256,128,128'        10
sbatch job.sbatch  1   2048            mlp             5            '512'            '256,128,128'        10
sbatch job.sbatch  2   2048            mlp             10           '512'            '256,128,128'        10
sbatch job.sbatch  3   2048           lstm             1            '256'            '256,128,128'        10
sbatch job.sbatch  4   2048           lstm             5            '256'            '256,128,128'        10
sbatch job.sbatch  5   2048           lstm            10            '256'            '256,128,128'        10
sbatch job.sbatch  6   2048           gru              1            '256'            '256,128,128'        10
sbatch job.sbatch  7   2048           gru              5            '256'            '256,128,128'        10
sbatch job.sbatch  8   2048           gru             10            '256'            '256,128,128'        10
sbatch job.sbatch  9   2048           tcn              10           '256,128,128'    '256,128,128'        10
sbatch job.sbatch  10   2048           tcn              10          '256,128,128'    '256,128,128'        10
sbatch job.sbatch  11  2048           tcn             10            '256,128,128'    '256,128,128'        10  
sbatch job.sbatch  12  2048           transformer      1            '256'            '256,128,128'        10
sbatch job.sbatch  13  2048           transformer      5            '1024'           '256,128,128'        10
sbatch job.sbatch  14  2048           transformer     10            '1024'           '256,128,128'        10