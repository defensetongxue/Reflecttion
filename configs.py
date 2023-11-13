import argparse,json
def get_config():
    parser = argparse.ArgumentParser()
    # cleansing
    # config file 
    parser.add_argument('--save_dir', help='experiment configuration filename',
                        default="./checkpoints", type=str)
    parser.add_argument('--cfg', help='experiment configuration filename',
                        default="./configs/defalut.json", type=str)
   
    args = parser.parse_args()
    # Merge args and config file 
    with open(args.cfg,'r') as f:
        args.configs=json.load(f)

    return args