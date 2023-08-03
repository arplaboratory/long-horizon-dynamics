import argparse

def parse_args():
    print("Parsing arguments ...")
    parser = argparse.ArgumentParser(description='Arguments for dynamics learning')

    # training
    parser.add_argument('-r', '--run_id',          type=int,      default=1)
    parser.add_argument('-d', '--gpu_id',          type=int,      default=0)
    parser.add_argument('--num_devices',           type=int,      default=1)
    parser.add_argument('-e', '--epochs',          type=int,      default=20000)
    parser.add_argument('-b', '--batch_size',      type=int,      default=16)
    parser.add_argument('--dropout',               type=float,    default=0.2)
    parser.add_argument('-l', '--learning_rate',   type=float,    default=0.0001)
    parser.add_argument('-s', '--shuffle',         type=bool,     default=True)
    parser.add_argument('-n', '--num_workers',     type=int,      default=4)
    parser.add_argument('-p', '--plot',            type=bool,     default=True)
    parser.add_argument('--save_freq',             type=int,      default=50)
    parser.add_argument('--normalize',             type=bool,     default=False)
    parser.add_argument('--val_freq',              type=int,      default=1)
    

    return parser.parse_args()


def save_args(args, file_path):
    print("Saving arguments ...")
    with open(file_path, "w") as f:
        for arg in vars(args):
            arg_name = arg
            arg_type = str(type(getattr(args, arg))).replace('<class \'', '')[:-2]
            arg_value = str(getattr(args, arg))
            f.write(arg_name)
            f.write(";")
            f.write(arg_type)
            f.write(";")
            f.write(arg_value)
            f.write("\n")

def load_args(file_path):
    print("Loading arguments ...")
    parser = argparse.ArgumentParser(description='Arguments for unsupervised keypoint extractor')
    with open(file_path, "r") as f:
        for arg in f.readlines():
            arg_name = arg.split(';')[0]
            arg_type = arg.split(';')[1]
            arg_value = arg.split(';')[2].replace('\n', '')
            if arg_type == "str":
                parser.add_argument("--" + arg_name, type=str, default=arg_value)
            elif arg_type == "int":
                parser.add_argument("--" + arg_name, type=int, default=arg_value)
            elif arg_type == "float":
                parser.add_argument("--" + arg_name, type=float, default=arg_value)
            elif arg_type == "list":
                arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
                parser.add_argument("--" + arg_name, type=list, default=arg_value)
            elif arg_type == "tuple":
                arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
                parser.add_argument("--" + arg_name, type=tuple, default=arg_value)
    return parser.parse_args()