"""
Entry point for the predicter

"""


import argparse, time
import torch
import numpy as np
import random
from ntpp.utils import ensure_dir
from ntpp.models.data import NTPPData
from ntpp.models.model import NTPP
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=str, default='../../data/ntpp/preprocess/events.txt', help='Event File containg the vents for each host.')
    parser.add_argument('--times', type=str, default='../../data/ntpp/preprocess/times.txt', help='File containing the time of the events for each host.')
    parser.add_argument('--save_dir', type=str, default='../../data/ntpp/saved/', help='Root dir for saving models.')
    parser.add_argument('--int_count', type=int, default=100, help='Number of intervals')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train Test split. e.g. 0.2 means 20% Test 80% Train')
    parser.add_argument('--time_step', type=int, default=8, help='Time Step')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of the batch')
    parser.add_argument('--element_size', type=int, default=2, help='Element Size')
    parser.add_argument('--h', type=int, default=128, help='Hiddden layer Size')
    parser.add_argument('--nl', type=int, default=1, help='Number of RNN Steps')
    parser.add_argument('--seed', type=int, default=123456, help='SEED')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='WHat do you want ? train | predict')
    parser.add_argument('--num_epochs', type=int, default=32, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    np.random.seed(args.seed)
    args = vars(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args = vars(args)
    print('Running NTPP in {} mode'.format(args['mode']))

    if args['mode'] =='train':
        train(args)
    else:
        evaluate(args)


def train(args):
    ensure_dir(args['save_dir'])
    #model file --

    #Loading Data
    network_data = NTPPData(args)
    number_host = len(network_data)
    print('NTPP model...')
    model = NTPP(args, output_layer_size=number_host)
    train_loader = torch.utils.data.DataLoader(network_data,
                            batch_size=args['batch_size'],
                            shuffle=False,
                            num_workers=args['workers']
                            # pin_memory=True # CUDA only
                            )
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    for epoch in range(args['num_epochs']):
        for i, batch in enumerate(train_loader):
            start_time = time.time()

            #forward pass
            outputs = model(batch)
            loss  # has to be written 

            #backward and optimize
            optimizer.zero_grad()
            loss.backward();
            optimizer.step()

            if(i+1)%100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss:{:.4f} Time : {}'
                        .format(epoch, args['num_epochs'], i+1, len(train_loader), loss.item(), time.time()-start_time))

        # Dev loss



def evaluate(args):
    network_data = NTPPData(args)
    number_host = len(network_data)
    print('NTPP model...')
    model = NTPP(args, output_layer_size=number_host)
    train_loader = torch.utils.data.DataLoader(
                    network_data,
                    batch_size=args['batch_size'],
                    shuffle=False,
                    num_workers=args['workers']
                    # pin_memory=True # CUDA only
                    )


if __name__ == '__main__':
    main()