"""
Entry point for the predicter

"""


import argparse, time
import torch
import numpy as np
import random
from ntpp.utils import ensure_dir, getMinCount
from ntpp.models.data import NTPPData
from ntpp.models.model import NTPP
from ntpp.models.scorer import discriminatorLoss, calculateLoss
from ntpp.models.ploter import plot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--events', type=str, default='data/ntpp/preprocess/events.txt', help='Event File containg the vents for each host.')
    parser.add_argument('--times', type=str, default='data/ntpp/preprocess/times.txt', help='File containing the time of the events for each host.')
    parser.add_argument('--save_dir', type=str, default='data/ntpp/saved/', help='Root dir for saving models.')
    parser.add_argument('--int_count', type=int, default=100, help='Number of intervals')
    parser.add_argument('--test_size', type=float, default=0.2, help='Train Test split. e.g. 0.2 means 20% Test 80% Train')
    parser.add_argument('--time_step', type=int, default=8, help='Time Step')
    parser.add_argument('--batch_size', type=int, default=8, help='Size of the batch')
    parser.add_argument('--element_size', type=int, default=2, help='Element Size')
    parser.add_argument('--h', type=int, default=128, help='Hiddden layer Size')
    parser.add_argument('--nl', type=int, default=1, help='Number of RNN Steps')
    parser.add_argument('--seed', type=int, default=123456, help='SEED')
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help='WHat do you want ? train | predict')
    parser.add_argument('--epochs', type=int, default=32, help='Number of epochs')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--metric', default='AUC', choices=['AUC','PRECISION','RECALL'], help='Number of workers')
    parser.add_argument('--is_cuda', default=False, choices=[True, False], help='CUDA or not')
    parser.add_argument('--optim', default='RMS', choices=['RMS', 'Adam', 'SGD'], help='Optimizer')


    args = parser.parse_args()
    return args

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.set_default_tensor_type('torch.DoubleTensor')
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args = vars(args)
    print(args)
    # Assert check
    print('Running NTPP in {} mode'.format(args['mode']))

    if args['mode'] =='train':
        train(args)
    # else:
    #     evaluate(args)


def train(args):
    ensure_dir(args['save_dir'])
    #model file --

    #Loading Data
    network_data = NTPPData(args)
    train_y,test_y = network_data.getObservation()
    print('NTPP model...')
    model = NTPP(args, output_layer_size=1)
    train_loader = torch.utils.data.DataLoader(network_data,
                            batch_size=args['batch_size'],
                            shuffle=False,
                            num_workers=args['workers'],
                            pin_memory=args['is_cuda'] # CUDA only
                            )
    if args['optim'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args['learning_rate'])
    elif args['optim'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args['learning_rate'])
    elif args['optim'] == 'RMS':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args['learning_rate'])
    else:
        raise Exception('USe Proper optimizer')
    loss_list,mape_t_list,mape_n_list = [],[],[]
    for epoch in range(args['epochs']):
        for i, (batch_events,batch_times) in enumerate(train_loader):
            start_time = time.time()
            # batch preprocess
            time_step = args['time_step']
            batch_events_part1 = batch_events[:,:time_step]
            batch_times_diff = batch_times[:, 1:1+time_step] - batch_times[:,:time_step]
            batch_times_diff_next = batch_times[:, 2:2+time_step] - batch_times[:,1:1+time_step]
            batch_times_diff = batch_times_diff[:,:,None] # exapnd dim in axis 2
            batch_events_part1 = batch_events_part1[:,:,None] # expand Dim
            # batch_times_diff_next = batch_times_diff_next[:,:,None]  # exapnd dims
            batch_input = torch.cat((batch_times_diff,batch_events_part1),2)
            # print("Input Sphape: ", batch_input.shape)
            #forward pass
            outputs = model(batch_input)
            # print("outputs size: ", outputs.shape)
            last_time_step_layer = outputs.clone()
            last_time_step_layer = (last_time_step_layer.detach().numpy()).transpose()[-1]
            predicted = []
            # print("last_time_step_layer shape: ", len(last_time_step_layer))
            for host in range(last_time_step_layer.shape[0]):
                for secon_host in range(host+1,last_time_step_layer.shape[0]):
                    predicted.append(np.greater(last_time_step_layer[host],last_time_step_layer[secon_host]))
            discriminator_loss = discriminatorLoss(train_y, predicted, args['metric'])
            loss, mape_t, mape_n = calculateLoss(discriminator_loss, outputs, batch_times_diff_next, time_step)
            loss_list.append(loss)
            mape_t_list.append(mape_t)
            mape_n_list.append(mape_n)
            #backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if(i+1)%max(number_host/3,1) == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f} MAPE_t: {:4f} MAPE_n: {:4f} Time : {}'
                    .format(epoch, args['epochs'], i+1, len(train_loader), loss.item(), mape_t.item(), mape_n.item(), time.time()-start_time))
    plot(args,loss_list,mape_t_list,mape_n_list)
# Dev loss
# save and load checkpoints


# def evaluate(args):
#     network_data = NTPPData(args)
#     number_host = len(network_data)
#     print('NTPP model...')
#     model = NTPP(args, output_layer_size=number_host)
#     test_loader = torch.utils.data.DataLoader(
#                     network_data,
#                     batch_size=args['batch_size'],
#                     shuffle=False,
#                     num_workers=args['workers']
#                     # pin_memory=True # CUDA only
#                     )


if __name__ == '__main__':
    main()
