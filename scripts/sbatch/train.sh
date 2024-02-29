#!/bin/bash
#                  R   B        model_type     unroll_length      d_model      decoder_sizes    num_heads    ffn_hidden      causal_masking     encoder_output
sbatch job.sbatch  0  64           transformer      1                 512      '1024,512,512'           8          2048               False             output
sbatch job.sbatch  1  64           transformer      1                 512      '1024,512,512'           8          2048               True              output
sbatch job.sbatch  2  64           transformer      1                 512      '1024,512,512'           8          2048               False              None
sbatch job.sbatch  3  64           transformer      1                 512      '1024,512,512'           8          2048               True               None

