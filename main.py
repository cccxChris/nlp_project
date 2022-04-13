# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.autograd as autograd
from data_pro import load_data_and_labels, Data, load_data_and_labels_single
from model import Model
from config import opt
import argparse
import pickle
import os

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

################ functions for influence function ################

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def collate_fn(batch):
    data, label = zip(*batch)
    return data, label

def hv(loss, model_params, v): # according to pytorch issue #24004
#     s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
#     e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
#     e2 = time.time()
#     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
    return Hv

def get_inverse_hvp_lissa(v, model, param_influence, train_loader, damping, num_samples, recursion_depth, scale=1e4):
    ihvp = None
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                x, labels = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                x, labels = next(lissa_data_iterator)

            model.zero_grad()
            train_loss = model(x, torch.LongTensor(labels).cuda())
            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [_a + (1 - damping) * _b - _c / scale for _a, _b, _c in zip(v, cur_estimate, hvp)]
            if (j % 200 == 0) or (j == recursion_depth - 1):
                print("Recursion at depth %s: norm is %f" % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy())))
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp

def influence_func(**kwargs):
    x_test, y_test = load_data_and_labels("./data/emotion_positive_test.txt", "./data/emotion_negative_test.txt")

    x_noise, y_noise = load_data_and_labels_single("./data/emotion_noise_short.txt")

    test_data = Data(x_test, y_test)

    noise_data = Data(x_noise, y_noise)
    train_loader = DataLoader(noise_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = Model(opt).cuda()
    model.load_state_dict(torch.load('./textcls.pkl'))

    param_optimizer = list(model.named_parameters())
    param_influence = []
    for n, p in param_optimizer:
        if (not ('glove' in n)) and (not ('encoder' in n)):
            param_influence.append(p)
        else:
            p.requires_grad = False

    param_shape_tensor = []
    param_size = 0
    for p in param_influence:
        tmp_p = p.clone().detach()
        param_shape_tensor.append(tmp_p)
        param_size += torch.numel(tmp_p)

    for data in test_loader:
        model.eval()

        #### Get Test example decision
        x, labels = data
        # print('test data', len(x))
        output = model(x)
        labels = torch.LongTensor(labels)
        predict = torch.max(output.data, 1)[1]
        print('labels, predict', labels, predict)
        ######## L_TEST GRADIENT ########
        model.zero_grad()
        test_loss = model(x, labels.cuda())
        test_grads = autograd.grad(test_loss, param_influence)

        ######## IHVP ########
        model.train()

        inverse_hvp = get_inverse_hvp_lissa(test_grads, model, param_influence, train_loader,
                                            damping=0.0, num_samples=1,
                                            recursion_depth=int(len(train_loader.dataset) * 1.0))

        ################
        influences = np.zeros(len(train_loader.dataset))
        from tqdm import tqdm
        for train_idx, (x, labels) in enumerate(
                tqdm(train_loader, desc="Train set index")):
            model.train()

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            train_loss = model(x, torch.LongTensor(labels).cuda())
            train_grads = autograd.grad(train_loss, param_influence)
            influences[train_idx] = torch.dot(inverse_hvp, gather_flat_grad(train_grads)).item()

        print('influences', influences)
        pickle.dump(influences, open(os.path.join('./', "influences_on_x_test_" + str(1) + ".pkl"), "wb"))


def train(**kwargs):

    opt.parse(kwargs)
    device = torch.device("cuda:{}".format(opt.gpu_id) if torch.cuda.is_available() else "cpu")
    opt.device = device

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    x_train, y_train = load_data_and_labels("./data/emotion_positive_train.txt", "./data/emotion_negative_train.txt")
    x_test, y_test = load_data_and_labels("./data/emotion_positive_test.txt", "./data/emotion_negative_test.txt")

    x_noise, y_noise = load_data_and_labels_single("./data/emotion_noise_short.txt")

    # x_train, x_test, y_train, y_test = train_test_split(x_text, y, test_size=opt.test_size, random_state=11)

    train_data = Data(x_train, y_train)
    test_data = Data(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)


    print(f"{now()} train data: {len(train_data)}, test data: {len(test_data)}")

    model = Model(opt)
    print(f"{now()} {opt.emb_method} init model finished")

    if opt.use_gpu:
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr_sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    best_acc = -0.1
    best_epoch = -1
    start_time = time.time()

    print('len(train_loader.dataset)', len(train_loader.dataset))

    for epoch in range(1, opt.epochs):
        total_loss = 0.0
        model.train()
        for step, batch_data in enumerate(train_loader):
            x, labels = batch_data
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                labels = labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        acc = test(model, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
        print(f"{now()} Epoch{epoch}: loss: {total_loss}, test_acc: {acc}")
        lr_sheduler.step()

    # influence_func(model, test_loader_b1, train_loader_b1)
    torch.save(model.state_dict(), 'textcls.pkl')
    end_time = time.time()
    print("*"*20)
    print(f"{now()} finished; epoch {best_epoch} best_acc: {best_acc}, time/epoch: {(end_time-start_time)/opt.epochs}")


def test(model, test_loader):
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, labels = data
            num += len(labels)
            output = model(x)
            labels = torch.LongTensor(labels)
            if opt.use_gpu:
                output = output.cpu()
            predict = torch.max(output.data, 1)[1]
            correct += (predict == labels).sum().item()
    model.train()
    return correct * 1.0 / num

def parse_args():
    parser = argparse.ArgumentParser(description="text cls")
    parser.add_argument('--emb_method ', type=str, default='elmo',
    help="emb_method ")
    parser.add_argument('--enc_method ', type=str, default='cnn',
                        help="v")
    return parser.parse_args()

if __name__ == "__main__":
    # train()
    influence_func()
    # building precoditioner
    # import fire
    # fire.Fire()