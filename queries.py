import argparse
import logging
import os
import random
import time
import copy

from data.preprocess import loadDataset, dataLoader
from utils.attack import findKSEofALL
from utils.evaluate import asr_eval
from utils.zoo import estimateGrad

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=256, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--dataset_name', default='cifar10', type=str, help='dataset')
parser.add_argument('--encode_length', default=16, type=int, help='dimension of embedding')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--feature_ratio', default=0.3, type=float, help='feature ratio of adversary')
parser.add_argument('--queries', default=300, type=int, help='number of queries')
parser.add_argument('--adversary_id', default=0, type=int, help='choose which party is the adversary')
parser.add_argument('--target', default=1, type=int, help='target class')
parser.add_argument('--source', default=0, type=int, help='source class')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
from utils.function import setupSeed, feature_distribute, setupLogger, generateFeatureRatios, loadModels

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
filename = "query_{}_{}_q{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name, args.queries)
logger = setupLogger(filename)

# load models
logging.info('Preparing model')
modelbase = '{}/pretrained'.format(cwd)
resultbase = '{}/result'.format(cwd)
directory = "{}/d{}_p{}_r{}".format(args.dataset_name, args.encode_length, args.num_party, int(10 * args.feature_ratio))


def generationAttack(iters, num_dim, target, source, adversary_id, num_party, num_classes, amount, threshold, workers,
                     server, data_loaders, device):
    KSE = findKSEofALL(num_party=num_party, amount=amount, threshold=threshold, target=source,
                       workers=workers, server=server, data_loaders=data_loaders[0], device=device)
    while len(KSE[0]) < amount and threshold > 0:
        threshold -= 0.1
        currentKSE = findKSEofALL(num_party=num_party, amount=amount - len(KSE[0]), threshold=threshold, target=source,
                                  workers=workers, server=server, data_loaders=data_loaders[0], device=device)
        for i in range(len(currentKSE)):
            for kse in currentKSE[i]:
                KSE[i].append(kse)

    # ZOO generation
    advs = []
    height, step = 1e-2, 1e-2
    for i in range(10):
        logging.info('Generation of {}th trigger...'.format(i))
        adv = zoo(iters=iters, adversary_id=adversary_id, trigger=torch.zeros((1, num_dim)).to(device),
                  kernel_set=copy.deepcopy(KSE), target=target, server=server, height=height, step=step, device=device)
        advs.append(adv)

    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for adv in advs:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1], n_classes=num_classes,
                                       adv=adv, source=source, target=target, device=device)
        if avg_asr >= max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr >= max_source_asr:
            max_source_asr = source_asr
        results.append(source_asr)
        idx += 1
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    return results


def zoo(iters, adversary_id, trigger, kernel_set, target, server, height, step, device):
    num_party = len(kernel_set)
    iters = iters
    for iter in range(iters):
        index = random.randint(0, len(kernel_set[0]) - 1)
        embeds = [kernel_set[j][index].reshape(1, -1).to(device) for j in range(num_party)]
        embeds[adversary_id] *= 0
        embeds[adversary_id] += trigger
        coordinate = random.randint(0, len(kernel_set[0][0]) - 1)
        grad = estimateGrad(adversary_id, embeds, coordinate, height, target, server) + 1e-5
        while abs(step * grad) < 1e-1:
            step *= 10
        trigger[0][coordinate] -= step * grad
        step = 1e-2
    return trigger


if os.path.exists(os.path.join(modelbase, directory)):
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    workers, server = loadModels(root=os.path.join(modelbase, directory), dataset_name=dataset_name,
                                 num_features=num_features, encode_length=args.encode_length, num_party=args.num_party,
                                 num_classes=num_classes, num_layers=1, device=device)

    logging.info('Target class: {}, Source class: {}'.format(args.target, args.source))

    logging.info('----------Generation Attack----------')
    generation_asr = generationAttack(iters=args.queries, num_dim=args.encode_length, target=args.target,
                                      source=args.source, adversary_id=args.adversary_id, num_party=args.num_party,
                                      num_classes=num_classes,
                                      amount=10, threshold=0.9, workers=workers, server=server,
                                      data_loaders=[train_data_loader, test_data_loader], device=device)
    torch.save(generation_asr, os.path.join(resultbase, directory, 'generation_asr_query_{}.pt'.format(args.queries)))
else:
    logging.info('Pretrained models do not exist! Please train first.')
