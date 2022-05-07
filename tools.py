# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
import numpy as np


def init_logging2(path):
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    timestamp = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
    handler_file = logging.FileHandler(os.path.join(path, "{}.log".format(timestamp)))
    handler_stream = logging.StreamHandler(sys.stdout)
    handler_file.setFormatter(formatter)
    handler_stream.setFormatter(formatter)
    logger.addHandler(handler_file)
    logger.addHandler(handler_stream)



class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def age_accuracy(p_age, age):
    batch_size = p_age.shape[0]
    acc_0, acc_1, acc_2, acc_5 = 0, 0, 0, 0
    for i in range(len(p_age)):
        result = round(p_age[i])
        if result == age[i]:
            acc_0 += 1
            acc_1 += 1
            acc_2 += 1
            acc_5 += 1
        elif result >= age[i] -1 and result <= age[i] +1:
            acc_1 += 1
            acc_2 += 1
            acc_5 += 1
        elif result >= age[i] -2 and result <= age[i] +2:
            acc_2 += 1
            acc_5 += 1
        elif result >= age[i] -5 and result <= age[i] +5:
            acc_5 += 1
    
    return acc_0/batch_size, acc_1/batch_size, acc_2/batch_size, acc_5/batch_size


def gender_accuracy(p_gender, gender):
    batch_size = p_gender.shape[0]
    
    result = [1 if i > 0.5 else 0 for i in p_gender]
    result = np.array([result])
    gender = gender.cpu().numpy()
    true_count = (result == gender).sum()
    gender_acc = true_count / batch_size
    
    return gender_acc
