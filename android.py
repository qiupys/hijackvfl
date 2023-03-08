import argparse
import logging
import os
import random
import time

import numpy as np

from data.preprocess import loadDataset, dataLoader
from utils.learn import train, test, poison_train

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=256, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--dataset_name', default='android', type=str, help='dataset')
parser.add_argument('--encode_length', default=16, type=int, help='dimension of embedding')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--epoch', default=5, type=int, help='epochs of normal training before poison')
parser.add_argument('--adversary_id', default=1, type=int, help='choose which party as the adversary')
parser.add_argument('--target', default=1, type=int, help='target class')
parser.add_argument('--source', default=0, type=int, help='source class')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
from utils.function import setupSeed, setupLogger, prepareModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setupSeed(args.seed)
cwd = os.getcwd()

# read args and configures
from configparser import ConfigParser

config = ConfigParser()
config.read('{}/config/datasets.config'.format(cwd), encoding='UTF-8')
dataset_name = args.dataset_name
input_size = config.getint(dataset_name, 'input_size')
max_length = config.getint(dataset_name, 'max_length') if dataset_name in ['imdb', 'android'] else None
num_classes = config.getint(dataset_name, 'num_classes')
num_features = [input_size, max_length]

# setup logger
filename = "{}_poison_{}_adv{}.txt".format(time.strftime('%Y%m%d%H%M%S'), dataset_name, args.adversary_id)
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '{}/pretrained'.format(cwd)
resultbase = '{}/result'.format(cwd)
directory = "{}/poison_d{}_adv{}".format(dataset_name, args.encode_length, args.adversary_id)

train_accuracies = []
test_accuracies = []
if not os.path.exists(os.path.join(modelbase, directory)):
    os.makedirs(os.path.join(modelbase, directory))
    logging.info('Pretrained models do not exist! Begin training...')
    workers, server = prepareModels(dataset_name=dataset_name, num_features=num_features,
                                    encode_length=args.encode_length, num_party=args.num_party, num_classes=num_classes,
                                    num_layers=1, device=device)

    # load data
    logging.info('Loading data')
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    # normal training
    lr, weight_decay = config.getfloat(dataset_name, 'lr'), config.getfloat(dataset_name, 'weight_decay')
    logging.info('Training ...')
    for epoch in range(args.epoch):
        train_accuracy = train(epoch=epoch, workers=workers, server=server, data_loaders=train_data_loader, lr=lr,
                               weight_decay=weight_decay, device=device)
        train_accuracies.append(train_accuracy)
        test_accuracy = test(epoch=epoch, workers=workers, server=server, data_loaders=test_data_loader, device=device)
        test_accuracies.append(test_accuracy)

    # auxiliary dataset preparation
    dataset_size = len(train_datasets[args.adversary_id])
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    train_indices = indices[:int(dataset_size * 0.01)]

    # poison training
    labels = train_datasets[0].label
    index = np.where(np.array(labels)[train_indices] == args.target)[0]
    poison_list = np.array(random.sample(list(np.array(train_indices)[index]), len(index)))
    trigger = torch.randn(args.encode_length).to(device)
    torch.save(trigger, os.path.join(modelbase, directory, 'trigger.pt'))
    torch.save(train_indices, os.path.join(modelbase, directory, 'train_indices.pt'))
    torch.save(poison_list, os.path.join(modelbase, directory, 'poison_list.pt'))

    logging.info('Poisoning ...')
    for epoch in range(args.epoch, config.getint(dataset_name, 'epoch')):
        train_accuracy = poison_train(epoch=epoch, workers=workers, server=server, data_loaders=train_data_loader,
                                      batch_size=batch_size,
                                      lr=lr, weight_decay=weight_decay, poison_list=poison_list, trigger=trigger,
                                      adversary_id=args.adversary_id, device=device)
        train_accuracies.append(train_accuracy)
        test_accuracy = test(epoch=epoch, workers=workers, server=server, data_loaders=test_data_loader, device=device)
        test_accuracies.append(test_accuracy)

    # save models
    for i, worker in enumerate(workers):
        torch.save(worker.state_dict(), os.path.join(modelbase, directory, 'worker_{}.pt'.format(i)))
    torch.save(server.state_dict(), os.path.join(modelbase, directory, 'server.pt'))
    if not os.path.exists(os.path.join(resultbase, directory)):
        os.makedirs(os.path.join(resultbase, directory))
    torch.save(train_accuracies, os.path.join(resultbase, directory, 'train_accuracy.pt'))
    torch.save(test_accuracies, os.path.join(resultbase, directory, 'test_accuracy.pt'))
else:
    logging.info('Pretrained models have been saved.')
