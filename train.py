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
from torch.utils.tensorboard import SummaryWriter
import tqdm
import argparse
import logging

from tools import *
from dataset.utkface import UTKface
from kaggle_training.kaggleModel import KaggleModel
from mobilenet_training.mobilenet import MobileNetV2
from resnet_training import resnet
from resnet_training import resnetConv


def get_model(model_name):
    model = None
    if model_name == "kaggle":
        model = KaggleModel()
    elif model_name == "mobilenet":
        model = MobileNetV2()
    elif model_name == "resnet18":
        model = resnet.resnet18(100)
    elif model_name == "resnet34":
        model = resnet.resnet34(100)
    elif model_name == "resnet50":
        model = resnet.resnet50(100)
    elif model_name == "resConv18":
        model = resnetConv.resnet18(100)
    elif model_name == "resConv34":
        model = resnetConv.resnet34(100)
    elif model_name == "resConv50":
        model = resnetConv.resnet50(100)

    return model    
    


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = UTKface(image_dir=args.image_dir, image_size=128) 
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, 
    
                                              shuffle=True, drop_last=True)
    init_logging2(args.log_dir)

    model = get_model(args.model)
    model = model.to(device=device)
    start_epoch = 0
    if args.resume is not None:
        try:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint)
            start_epoch = int(str(os.path.splitext(args.resume)[0]).split("_")[-1])
            logging.info("resume successfully in %s" % args.resume)
        except Exception as e:
            logging.info("resume error in %s, init..." % args.resume)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_age = nn.L1Loss().to(device)
    criterion_gender = nn.BCELoss().to(device)
    writer = None
    
    if args.vis:
        writer = SummaryWriter()
    
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    
    model.train()
    for epoch in range(start_epoch + 1, args.epoch + 1):
        age_0_acc = AverageMeter()
        age_1_acc = AverageMeter()
        age_2_acc = AverageMeter()
        age_5_acc = AverageMeter()
        gender_0_acc = AverageMeter()
        all_loss = AverageMeter()
        
        logging.info("Epoch %d/%d" % (epoch, args.epoch)) 
        for (image, age, gender) in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            batch_size = image.shape[0]
            image = image.to(device)
            age, gender = age.to(device), gender.to(device)
            p_age, p_gender = model(image)

            loss_age = criterion_age(p_age, age)
            loss_gender = criterion_gender(p_gender, gender.float())
            loss = loss_age + loss_gender
            loss.backward()
            optimizer.step()
            all_loss.update(loss, batch_size)
            age_acc_0, age_acc_1, age_acc_2, age_acc_5 = age_accuracy(p_age.cpu().detach().numpy(), age.cpu().detach().numpy())
            gender_acc = gender_accuracy(p_gender, gender)
            age_0_acc.update(age_acc_0, batch_size)
            age_1_acc.update(age_acc_1, batch_size)
            age_2_acc.update(age_acc_2, batch_size)
            age_5_acc.update(age_acc_5, batch_size)
            gender_0_acc.update(gender_acc, batch_size)
        
        if args.vis and writer is not None:
            writer.add_scalar('Loss', all_loss.avg, epoch)
            writer.add_scalar("age_0_acc", age_0_acc.avg, epoch)
            writer.add_scalar("age_1_acc", age_1_acc.avg, epoch)
            writer.add_scalar("age_2_acc", age_2_acc.avg, epoch)
            writer.add_scalar("age_5_acc", age_5_acc.avg, epoch)
            writer.add_scalar("gender_acc", gender_0_acc.avg, epoch)
        
        logging.info("loss: {:>7.5f}\tgender: {:>5.4f}".format(all_loss.avg, gender_0_acc.avg))
        logging.info("age_acc: {:>5.4f}\tage+-1 acc:{:>5.4f}\tage+-2 acc:{:>5.4f}\tage+-5 acc:{:>5.4f}".format(
            age_0_acc.avg, age_1_acc.avg, age_2_acc.avg, age_5_acc.avg
        ))

        savename = os.path.join(args.checkpoint_dir, "%s_%04d.pth" % (args.model, epoch))
        torch.save(model.state_dict(), savename)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kaggle classification training")
    parser.add_argument("--image_dir", default="/testapp/data/utkface/UTKFace", type=str, help="path of dataset")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--vis", default=False, type=bool, help="use torch.tensorboard to record loss and acc")
    parser.add_argument("--resume", default=None, type=str, help="path of resume checkpoint")
    parser.add_argument("--checkpoint_dir", default="./output", help="path of checkpoint")
    parser.add_argument("--epoch", default=100, type=int, help="num of training epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="training batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--log_dir", default="./logs", type=str, help="log dir")


    train(parser.parse_args())