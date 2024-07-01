from .models import MLP, LSTM, GRU, TCN

def get_model(args, input_size, output_size):
    
    model = {
        'mlp':         MLP(input_size,  args.history_length, args.decoder_sizes,  output_size,   args.dropout),
        'lstm':        LSTM(input_size, args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'gru':         GRU(input_size,  args.encoder_sizes,  args.num_layers,      args.history_length, args.decoder_sizes, output_size, args.dropout, args.encoder_output),
        'tcn':         TCN(input_size,  args.encoder_sizes,  args.history_length,  args.decoder_sizes,  output_size,   args.kernel_size, args.dropout),
    }

    # If MLP print model parameters in millions using only .model in 
    if args.model_type == 'mlp':
        print("Model parameters: ", sum(p.numel() for p in model[args.model_type].model.parameters())/1000000, "M")
    else:
        # Print the number of model parameters in millions for both encoder and decoder
        print("Encoder parameters: ", sum(p.numel() for p in model[args.model_type].encoder.parameters())/1000000, "M")
        print("Decoder parameters: ", sum(p.numel() for p in model[args.model_type].decoder.parameters())/1000000, "M")

    return model[args.model_type]


    