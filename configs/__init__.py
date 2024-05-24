from argparse import ArgumentParser

def load_arg():

    parser = ArgumentParser(description="Pytorch Training")
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Path to config file")
    
    parser.add_argument(
        "--load_path", 
        type=str, 
        help='Path of pretrained model')
    
    parser.add_argument(
        "--tag", 
        type=str, 
        help='Tag of experience')


    return parser.parse_args()
