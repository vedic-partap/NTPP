import pandas as pd
import os
import numpy as np


def extractor(path, minl=50):
    data = pd.read_csv(path + '/testpackets.csv')
    event_time_stamps = {}
    for row in range(data.shape[0]):
        row = list(data.loc[row])
        if row[2] in event_time_stamps:
            event_time_stamps[row[2]].append((row[5], row[1]))
        else:
            event_time_stamps[row[2]] = [(row[5], row[1])]
    filtered_event_time_stamps = {
        k: v
        for k, v in event_time_stamps.items() if len(v) >= minl
    }
    with open(path + '/ntpp/preprocess/events.txt', 'w') as f:
        for k in filtered_event_time_stamps.keys():
            f.write(" ".join(
                [str(x[0]) for x in filtered_event_time_stamps[k]]))
            f.write("\n")

    with open(path + '/ntpp/preprocess/times.txt', 'w') as f:
        for k in filtered_event_time_stamps.keys():
            f.write(" ".join(
                [str(x[1]) for x in filtered_event_time_stamps[k]]))
            f.write("\n")


def read_file(filename):
    with open(filename, 'r') as f:
        row = f.readline()
    data = [[float(ti) for ti in line.rstrip().split(' ')] for line in row]
    return data


def compare_interval_count(left, right, host_count, interval_count):
    Y = []
    for interval in range(left, right):
        a = []
        for host in range(host_count - 1):
            a.extend(
                np.greater(interval_count[host][interval],
                           interval_count[:, interval])[host + 1:])
        Y.append(a)
        return Y


# def pcapToCsv(filename):
# write function to convert the pcap file to the csv using tshark


def ensure_dir(d, verbose=True):
    if not os.path.exists(d):
        if verbose:
            print("Directory {} do not exist; creating...".format(d))

        os.makedirs(d)