import logging
import math
import os
import torch
import numpy as np
import random

from utils.model import MLP, MyResNet, BertBaseModel, Server


def setupSeed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setupLogger(filename, record=True):
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s')

    # setup stream handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # setup file handler
    if record:
        if not os.path.exists('./logs'):
            os.makedirs('./logs')
        fh = logging.FileHandler("./logs/{}".format(filename), encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logging.info('Start print log......')
    return logger


def generateFeatureRatios(num_party, feature_ratio=None):
    if num_party == 2 and feature_ratio is not None:
        return [feature_ratio, 1 - feature_ratio]
    elif num_party > 2:
        feature_ratios = []
        for i in range(num_party - 1):
            feature_ratios.append(feature_ratio)
        feature_ratios.append(1 - (num_party - 1) * feature_ratio)
        return feature_ratios


def feature_distribute(size, feature_ratios):
    num_features = []
    total = 0
    for i in range(len(feature_ratios) - 1):
        num_feature = round(size * feature_ratios[i])
        num_features.append(num_feature)
        total += num_feature
    num_features.append(size - total)
    return num_features


def prepareModels(dataset_name, num_features, encode_length, num_party, num_classes, num_layers,
                  device):
    workers = []
    if dataset_name in ['bank', 'criteo']:
        for num_feature in num_features:
            workers.append(
                MLP(in_features=num_feature, out_features=encode_length, num_layers=num_layers).to(device))
    elif dataset_name in ['imdb']:
        for _ in range(len(num_features)):
            workers.append(BertBaseModel(encode_length).to(device))
    elif dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'emotion']:
        in_channels = 1 if dataset_name in ['mnist', 'fmnist'] else 3
        for _ in range(len(num_features)):
            workers.append(
                MyResNet(in_channels=in_channels, encode_length=encode_length).to(device))
    elif dataset_name in ['android']:
        workers.append(MLP(in_features=num_features[0], out_features=encode_length, num_layers=num_layers).to(device))
        workers.append(BertBaseModel(encode_length).to(device))
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=1).to(device)
    return workers, server


def loadModels(root, dataset_name, num_features, encode_length, num_party, num_classes, num_layers, device):
    workers = []
    if dataset_name in ['bank', 'criteo']:
        for i, num_feature in enumerate(num_features):
            worker = MLP(in_features=num_feature, out_features=encode_length, num_layers=num_layers).to(device)
            path = os.path.join(root, 'worker_{}.pt'.format(i))
            worker.load_state_dict(torch.load(path))
            workers.append(worker)
    elif dataset_name in ['imdb']:
        for i, num_feature in enumerate(num_features):
            worker = BertBaseModel(encode_length).to(device)
            path = os.path.join(root, 'worker_{}.pt'.format(i))
            worker.load_state_dict(torch.load(path))
            workers.append(worker)
    elif dataset_name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'emotion']:
        in_channels = 1 if dataset_name in ['mnist', 'fmnist'] else 3
        for i, num_feature in enumerate(num_features):
            worker = MyResNet(in_channels=in_channels, encode_length=encode_length).to(device)
            path = os.path.join(root, 'worker_{}.pt'.format(i))
            worker.load_state_dict(torch.load(path))
            workers.append(worker)
    elif dataset_name in ['android']:
        worker = MLP(in_features=num_features[0], out_features=encode_length, num_layers=num_layers).to(device)
        path = os.path.join(root, 'worker_{}.pt'.format(0))
        worker.load_state_dict(torch.load(path))
        workers.append(worker)
        worker = BertBaseModel(encode_length).to(device)
        path = os.path.join(root, 'worker_{}.pt'.format(1))
        worker.load_state_dict(torch.load(path))
        workers.append(worker)
    else:
        logging.info('Not supported datatype!')
        exit()
    server = Server(num_party=num_party, in_features=encode_length, num_classes=num_classes, num_layers=1).to(
        device)
    path = os.path.join(root, 'server.pt')
    server.load_state_dict(torch.load(path))
    return workers, server
