from sklearn.metrics import roc_auc_score, precision_score, recall_score
import torch

def discriminatorLoss(train_y, predicted, metrics='AUC'):
    res = []
    for i in range(train_y.shape[0]):
        if metrics=='AUC':
            res.append(roc_auc_score(train_y[i], predicted))
        elif metrics == 'PRECISION':
            res.append(precision_score(train_y[i], predicted))
        elif metrics == 'RECALL':
            res.append(recall_score(train_y[i], predicted))
        else:
            raise Exception('NotACorrectMetrics')
    return res


def calculateLoss(dLoss, output, batch_times_diff_next, time_step):
    output_delta_mul = torch.mul(output, batch_times_diff_next)
    time_LLs = torch.log(output) - output_delta_mul
    Nhhat = torch.squeeze(output_delta_mul,dim=2).sum(dim=0)
    batch_size = output.shape[0]
    Nh = torch.tensor([time_step*1.0]*batch_size)
    mape_n = (torch.abs(Nh-Nhhat)/Nh).mean()

    pred_delta = torch.squeeze(torch.abs(1/output),dim=2).mean()
    obs_delta = torch.squeeze(torch.abs(batch_times_diff_next), dim=2).mean()
    mape_t = torch.abs(pred_delta - obs_delta) / obs_delta
    loss = ((-time_LLs) + (dLoss).mean()).mean()
    return loss, mape_t, mape_n
