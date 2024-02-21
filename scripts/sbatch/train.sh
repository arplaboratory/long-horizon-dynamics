#!/bin/bash
#                  R   B        model_type     unroll_length      encoder_sizes      decoder_sizes    vehicle_type
sbatch job.sbatch  0  2048           tcn              10            '256,128,128'    '256,128,128'      quadrotor
sbatch job.sbatch  0  2048           tcn              10            '256,128,128'    '256,128,128'       neurobem