import argparse
import logging
import os
import time

from data.preprocess import loadDataset, dataLoader
from utils.learn import train, test

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=256, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--dataset_name', default='cifar10', type=str, help='dataset')
parser.add_argument('--encode_length', default=16, type=int, help='dimension of embedding')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--feature_ratio', default=0.3, type=float, help='feature ratio of adversary')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
from utils.function import setupSeed, feature_distribute, setupLogger, prepareModels, generateFeatureRatios

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
feature_ratios = generateFeatureRatios(args.num_party, args.feature_ratio)
assert len(feature_ratios) == args.num_party
if input_size < 0:
    input_size = max_length
if dataset_name == 'android':
    num_features = [input_size, max_length]
else:
    num_features = feature_distribute(input_size, feature_ratios)

# setup logger
filename = "{}_{}_p{}_r{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name, args.num_party,
                                      int(10 * args.feature_ratio))
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '{}/pretrained'.format(cwd)
resultbase = '{}/result'.format(cwd)
directory = "{}/d{}_p{}_r{}".format(args.dataset_name, args.encode_length, args.num_party, int(10 * args.feature_ratio))

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
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features, max_length)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    # normal training
    lr, weight_decay = config.getfloat(dataset_name, 'lr'), config.getfloat(dataset_name, 'weight_decay')
    logging.info('Training...')
    for epoch in range(config.getint(dataset_name, 'epoch')):
        train_accuracy = train(epoch=epoch, workers=workers, server=server, data_loaders=train_data_loader, lr=lr,
                               weight_decay=weight_decay, device=device)
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
