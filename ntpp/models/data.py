import torch
from torch.utils.data.dataset import Dataset
from ntpp.utils import read_file, compare_interval_count
import numpy as np


class NTPPData(Dataset):
    def __init__(self, args):
        self.args = args
        event_file = args['events']
        times_file = args['times']
        events = read_file(event_file)
        times = read_file(times_file)
        self.time_step = args['time_step']
        self.host_count = len(events)
        train_interval = int(args['int_count'] * (1 - args['test_size']))
        interval_size = (max([s[-1] for s in times]) + 1) / args['int_count']

        interval_count = np.zeros((self.host_count, args['int_count']),
                                  dtype=int)

        self.train_event, self.train_times = [[] for i in range(self.host_count)], [[] for i in range(self.host_count)]
        test_event, test_times = [[] for i in range(self.host_count)
                                  ], [[] for i in range(self.host_count)]
        for i, host in enumerate(times):
            for j, time_stamp in enumerate(host):
                counter = int(time_stamp / interval_size)
                interval_count[i][counter] += 1
                if counter < train_interval:
                    self.train_times[i].append(times[i][j])
                    self.train_event[i].append(events[i][j])
                else:
                    test_times[i].append(times[i][j])
                    test_event[i].append(events[i][j])
        min_count = min([len(x) for x in self.train_event])
        assert min_count >= args['time_step'] + 2, "Time Step should be less than {0}".format(min_count - 2)
        self.train_y = compare_interval_count(0, train_interval,
                                              self.host_count, interval_count)

        self.test_y = compare_interval_count(train_interval, args['int_count'],
                                             self.host_count, interval_count)

    def getObservation(self):
        return self.train_y, self.test_y

    def __getitem__(self, idx):
        x = np.array(self.train_event[idx][:self.time_step+2]), np.array(self.train_times[idx][:2+self.time_step])
        # print(x[0].shape, x[1].shape)
        return x
    def __len__(self):
        return self.host_count
