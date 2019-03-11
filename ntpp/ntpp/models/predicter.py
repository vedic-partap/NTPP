"""
Entry point for the predicter

"""


import argparse
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=str, default='../../data/ntpp/preprocess/events.txt', help='Event File containg the vents for each host.')
    parser.add_argument('--times', type=str, default='../../data/ntpp/preprocess/pos', help='File containing the time of the events for each host.')
    parser.add_argument('--save_dir', type=str, default='../../data/ntpp/saved/', help='Root dir for saving models.')
    parser.add_argument('--int_count', type=int, default='100', help='Number of intervals')
    parser.add_argument('--test_size', type=float, default='0.2', help='Train Test split. e.g. 0.2 means 20% Test 80% Train')
    parser.add_argument('--time_step', type=int, default='8', help='Time Step')
    parser.add_argument('--batch_size', type=int, default='128', help='Size of the batch')
    parser.add_argument('--element_size', type=int, default='2', help='Element Size')
    parser.add_argument('--h', type=int, default='128', help='Hiddden layer Size')
    parser.add_argument('--nl', type=int, default='1', help='Number of RNN Steps')
    

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)
    args = vars(args)


# def train(args):


# def evaluate(args):

if __name__ == '__main__':
    main()
