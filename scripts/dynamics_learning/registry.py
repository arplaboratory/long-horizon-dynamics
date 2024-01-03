from .models import MLP, LSTM, GRU, TCN, Transformer

import torch.nn as nn

def get_model(args, input_size, output_size):
    
    model = {
        'lstm':  LSTM(input_size, args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'gru':   GRU(input_size,  args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'tcn':   TCN(input_size,  args.encoder_sizes,  args.history_length,  args.decoder_sizes,  output_size,   args.kernel_size, args.dropout)
    }

    return model[args.model_type]


    