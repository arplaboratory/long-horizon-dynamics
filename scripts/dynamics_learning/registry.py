from .models import MLP, LSTM, GRU, TCN, Transformer

def get_encoder(args, input_size):
    
    encoder = {
        'mlp':   MLP(input_size=input_size,
                     output_size=args.encoder_dim,
                     num_layers=args.mlp_layers,
                     dropout=args.dropout),

        'lstm':  LSTM(input_size=input_size,
                      encoder_dim=args.encoder_dim,
                      encoder_sizes=args.hidden_size,
                      num_layers=args.num_layers,
                      history_length=args.history_length,
                      dropout=args.dropout,
                      output_type=args.output_type),
        'gru':   GRU(input_size=input_size,
                     encoder_dim=args.encoder_dim,
                     encoder_sizes=args.hidden_size,
                     num_layers=args.num_layers,
                     history_length=args.history_length,
                     dropout=args.dropout,
                     output_type=args.output_type),

        'tcn':   TCN(input_size=input_size,
                     encoder_dim=args.encoder_dim,
                     num_channels=args.num_channels,
                     kernel_size=args.kernel_size,
                     dropout=args.dropout),

        'transformer': Transformer(input_size=input_size,
                                   encoder_dim=args.encoder_dim,
                                   num_heads=args.num_heads,
                                   history_length=args.history_length,
                                   ffn_hidden=args.ffn_hidden,
                                   num_layers=args.num_layers,
                                   dropout=args.dropout)
    }

    return encoder[args.model_type]

def get_decoder(args, output_size):
    
    decoder = MLP(input_size=args.encoder_dim,
                  output_size=output_size,
                  num_layers=args.decoder_layers,
                  dropout=args.dropout)
    
    return decoder
    
    