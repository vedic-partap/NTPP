from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch
import numpy as np
import sys

def discriminatorLoss(train_y, predicted, metrics='AUC'):
    res = []
    for i in range(len(train_y)):
        if metrics == 'AUC':
            # try:
            # print("Train_y: ",train_y[i])
            # print("predicted: ",predicted)
            # sys.exit()
            res.append(roc_auc_score(train_y[i], predicted))
            # except:
            # pass# print("Exception")
        elif metrics == 'PRECISION':
            res.append(precision_score(train_y[i], predicted))
        elif metrics == 'RECALL':
            res.append(recall_score(train_y[i], predicted))
        else:
            raise Exception('NotACorrectMetrics')
    return res


def calculateLoss(dLoss, output, batch_times_diff_next, time_step):
    output_delta_mul = torch.mul(output, batch_times_diff_next)
    time_LLs = (torch.log(output) - output_delta_mul)
    Nhhat = output_delta_mul.sum(dim=1)
    batch_size = output.shape[0]
    Nh = torch.tensor([time_step * 1.0] * batch_size)
    mape_n = (torch.abs(Nh - Nhhat) / Nh).mean()

    pred_delta = torch.abs(1 / output).mean()
    obs_delta = torch.abs(batch_times_diff_next).mean()
    mape_t = torch.abs(pred_delta - obs_delta) / obs_delta
    # print("output {0}\noutput{1}".format(torch.log(output), output))
    loss = ((-time_LLs) + (torch.tensor(dLoss)).mean()).mean()
    # sys.exit()
    return loss, mape_t, mape_n
