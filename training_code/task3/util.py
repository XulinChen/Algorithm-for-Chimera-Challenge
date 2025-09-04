import os
# import config
import math
from PIL import Image
import numpy as np
import pickle
from glob import glob
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
import torch
import random

def evaluate(model, train_loader):
    device = "cuda"
    pred_risk_all = torch.FloatTensor().to(device)
    surv_time_all = torch.FloatTensor().to(device)
    status_all = torch.FloatTensor().to(device)
    preds_list = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            ret_feat, label, event_time, c, info = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[
                4].cuda()
            # ret_feat = torch.unsqueeze(ret_feat, dim=1)
            batch_size = ret_feat.size(0)

            hazards, S = model(ret_feat)
            risk = -torch.sum(S, dim=1)  # .detach().cpu().numpy()

            surv_time_all = torch.cat([surv_time_all, event_time])
            status_all = torch.cat([status_all, c])
            pred_risk_all = torch.cat([pred_risk_all, risk])
            preds_list.extend(risk.data.cpu().numpy().tolist())

    surv_time_all = surv_time_all.data.cpu().numpy()
    status_all = status_all.data.cpu().numpy()
    pred_risk_all = pred_risk_all.data.cpu().numpy()
    c_index = concordance_index_censored((1 - status_all).astype(bool), surv_time_all, pred_risk_all, tied_tol=1e-08)[0]

    # train_cindex_lst.append("%.3f" % c_index)
    return c_index, preds_list

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss_mean = loss.mean()
    # print('loss, loss_mean:', loss, loss_mean, loss.size(), loss_mean.size())
    return loss, loss_mean

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)

def get_data(csv_name):
    train_file = pd.read_csv(csv_name)
    train_id = train_file['case_id'].tolist()
    event = train_file['event'].tolist()
    converted_event = [1 if x == 0 else 0 for x in event]
    survival_time = train_file['follow_up_years'].tolist()
    label = train_file['label'].tolist()

    return train_id, converted_event, survival_time, label


def save_pkl(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def load_pkl(input_file):
    with open(input_file, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def CIndex(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy()
    concord = 0.
    total = 0.
    N_test = labels.shape[0]
    labels = np.asarray(labels, dtype=bool)
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total = total + 1
                    if hazards[j] < hazards[i]:
                        concord = concord + 1
                    elif hazards[j] < hazards[i]:
                        concord = concord + 0.5

    return (concord / total)


def CIndex_lifeline(hazards, labels, survtime_all):
    labels = labels.data.cpu().numpy().reshape(-1)
    hazards = hazards.cpu().numpy().reshape(-1)
    survtime_all = survtime_all.cpu().numpy().reshape(-1)
    label = []
    hazard = []
    surv_time = []
    for i in range(len(hazards)):
        if not np.isnan(hazards[i]) and not np.isnan(survtime_all[i]) and not np.isnan(hazards[i]):
            label.append(labels[i])
            hazard.append(hazards[i])
            surv_time.append(survtime_all[i])

    new_label = np.asarray(label)
    new_hazard = np.asarray(hazard)
    new_surv = np.asarray(surv_time)
    # try:
    #     concordance_index(new_surv, -new_hazard, new_label)
    # except:
    #     import pdb; pdb.set_trace()
    return (concordance_index(new_surv, -new_hazard, new_label))

# get_tumor_sort_dict()
