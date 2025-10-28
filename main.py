from argparse import ArgumentParser
from Codes import *

if __name__ == '__main__':
    parser = ArgumentParser(description='Pytorch Convolutional Code Decoding Transformer')
    # training args
    parser.add_argument('--epochs', type=int, default=1000)                 #
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=0)

    # Code args
    parser.add_argument('--code_type', type=str, default='POLAR')
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)

    # model args
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)

    class Code():               # util class for arguments communication
        pass
    code = Code()
