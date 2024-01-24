from .models import MLP, LSTM, GRU, TCN, Transformer

import torch.nn as nn

def get_model(args, input_size, output_size):
    
    model = {
        'mlp':         MLP(input_size,  args.history_length, args.decoder_sizes,  output_size,   args.dropout),
        'lstm':        LSTM(input_size, args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'gru':         GRU(input_size,  args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'tcn':         TCN(input_size,  args.encoder_sizes,  args.history_length,  args.decoder_sizes,  output_size,   args.kernel_size, args.dropout),
        'transformer': Transformer(input_size, args.d_model, args.num_heads, args.history_length, args.ffn_hidden,     args.num_layers, args.dropout, args.decoder_sizes, output_size)
    }

    # Print no. of parameters of encoder and decoder
    print('Encoder parameters: ', sum(p.numel() for p in model[args.model_type].encoder.parameters() if p.requires_grad))
    print('Decoder parameters: ', sum(p.numel() for p in model[args.model_type].decoder.parameters() if p.requires_grad))

    return model[args.model_type]


    