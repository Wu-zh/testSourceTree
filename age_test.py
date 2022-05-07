# -*- coding: utf-8 -*-
# author: wuzhuohao
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.getcwd()
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from dataset.fgnet import FGface
from mobilenet_training.mobilenet import MobileNetV2

import tqdm
import argparse

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



def age_test(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = FGface(image_dir=args.image_dir, image_size=128) 
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size)

    model = MobileNetV2().to(device=device)
    start_epoch = 0
    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            start_epoch = int(str(os.path.splitext(args.resume)[0]).split("_")[-1])
            print("resume successfully in %s" % args.resume)
        except Exception as e:
            print("resume error in %s, init..." % args.resume)
        

    model.eval()
    
    age_0_acc = AverageMeter()
    age_1_acc = AverageMeter()
    age_2_acc = AverageMeter()
    age_5_acc = AverageMeter()
    gender_0_acc = AverageMeter()
    all_loss = AverageMeter()
    
    for image, age in tqdm.tqdm(train_loader):

        batch_size = image.shape[0]
        image = image.to(device)
        age = age.to(device)
        p_age, _ = model(image)

        age_acc_0, age_acc_1, age_acc_2, age_acc_5 = age_accuracy(p_age.cpu().detach().numpy(), age.cpu().detach().numpy())
        age_0_acc.update(age_acc_0, batch_size)
        age_1_acc.update(age_acc_1, batch_size)
        age_2_acc.update(age_acc_2, batch_size)
        age_5_acc.update(age_acc_5, batch_size)

    

    print("age_acc: {:>5.4f}\tage+-1 acc:{:>5.4f}\tage+-2 acc:{:>5.4f}\tage+-5 acc:{:>5.4f}".format(
        age_0_acc.avg, age_1_acc.avg, age_2_acc.avg, age_5_acc.avg
    ))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kaggle classification training")
    parser.add_argument("--image_dir", default="C:\\Users\\cmic\\Downloads\\FGNET\\FGNET\\images", type=str, help="path of dataset")
    parser.add_argument("--resume", default=None, help="path of resume checkpoint")
    parser.add_argument("--batch_size", default=64, type=int, help="training batch size")
  
    args = parser.parse_args()
    checkpoint_dir = args.resume
    for item in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, item) 
        args.resume = path
        age_test(args)