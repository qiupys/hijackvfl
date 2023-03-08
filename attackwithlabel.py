import argparse
import logging
import os
import time

from torch.utils.data import SubsetRandomSampler

from data.preprocess import loadDataset, dataLoader
from utils.mixup import mix_up_train
from utils.model import SurrogateServer
from utils.surrogate_attack import findKSEwithSurrogateModel, surrogateReplayAttack, surrogateGenerationAttack, findRSEwithSurrogateModel, surrogateScaleAttack

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=256, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--dataset_name', default='criteo', type=str, help='dataset')
parser.add_argument('--encode_length', default=16, type=int, help='dimension of embedding')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--feature_ratio', default=0.5, type=float, help='feature ratio of adversary')
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
filename = "al_{}_{}_p{}_r{}.txt".format(time.strftime('%Y%m%d%H%M%S'), args.dataset_name, args.num_party,
                                         int(10 * args.feature_ratio))
logger = setupLogger(filename, True)

# load models
logging.info('Preparing model')
modelbase = '{}/pretrained'.format(cwd)
resultbase = '{}/result'.format(cwd)
directory = "{}/poison_d{}_p{}_r{}".format(args.dataset_name, args.encode_length, args.num_party,
                                           int(10 * args.feature_ratio))

if os.path.exists(os.path.join(modelbase, directory)):
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    workers, server = loadModels(root=os.path.join(modelbase, directory), dataset_name=dataset_name,
                                 num_features=num_features, encode_length=args.encode_length, num_party=args.num_party,
                                 num_classes=num_classes, num_layers=1, device=device)
    trigger = torch.load(os.path.join(modelbase, directory, 'trigger.pt')).to(device)
    train_indices = torch.load(os.path.join(modelbase, directory, 'train_indices.pt'))
    poison_list = torch.load(os.path.join(modelbase, directory, 'poison_list.pt'))

    logging.info('----------Train Surrogate Model----------')
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_datasets[args.adversary_id], batch_size=batch_size,
                                               sampler=train_sampler)

    ss = SurrogateServer(in_features=args.encode_length, num_classes=num_classes).to(device)
    mix_up_train(epochs=300, trigger=trigger, poison_list=poison_list, worker=workers[args.adversary_id],
                 server=ss, batch_size=batch_size, lr=1e-3, weight_decay=5e-4,
                 num_data=len(train_indices), data_loader=train_loader, device=device)

    logging.info('Target class: {}, Source class: {}'.format(args.target, args.source))

    logging.info('----------Prepare Random Sample Embedding----------')
    amount = 10
    RSE = findRSEwithSurrogateModel(amount=amount, trigger=trigger, target=args.target,
                                    data_loader=train_data_loader[args.adversary_id],
                                    worker=workers[args.adversary_id], server=ss, device=device)

    logging.info('Find {} samples for attack.'.format(len(RSE)))

    logging.info('----------Scale Attack----------')
    ss_scale_asr = surrogateScaleAttack(RSE=RSE, multiplier=10, adversary_id=args.adversary_id, workers=workers,
                                        server=server, data_loaders=[train_data_loader, test_data_loader],
                                        num_classes=num_classes, source=args.source, target=args.target,
                                        device=device)
    torch.save(ss_scale_asr, os.path.join(resultbase, directory, 'ss_scale_asr.pt'))

    logging.info('----------Prepare Kernel Sample Embedding----------')
    threshold = 0.9
    amount = 10
    KSE = findKSEwithSurrogateModel(amount=amount, trigger=trigger, target=args.target, threshold=threshold,
                                    data_loader=train_data_loader[args.adversary_id],
                                    worker=workers[args.adversary_id], server=ss, device=device)
    while len(KSE) < 10 and threshold >= 0.1:
        threshold -= 0.1
        currentKSE = findKSEwithSurrogateModel(amount=amount - len(KSE), trigger=trigger, target=args.target,
                                               threshold=threshold, data_loader=train_data_loader[args.adversary_id],
                                               worker=workers[args.adversary_id], server=ss, device=device)
        for kse in currentKSE:
            KSE.append(kse)

    logging.info('Find {} samples for attack.'.format(len(KSE)))

    logging.info('----------Replay Attack----------')
    ss_replay_asr = surrogateReplayAttack(KSE=KSE, adversary_id=args.adversary_id, workers=workers, server=server,
                                          data_loaders=[train_data_loader, test_data_loader],
                                          num_classes=num_classes, source=args.source, target=args.target,
                                          device=device)
    torch.save(ss_replay_asr, os.path.join(resultbase, directory, 'ss_replay_asr.pt'))

    logging.info('----------Generation Attack----------')
    ss_generation_asr = surrogateGenerationAttack(KSE=KSE, target=args.target, source=args.source,
                                                  adversary_id=args.adversary_id, num_classes=num_classes,
                                                  amount=10, num_dim=args.encode_length, workers=workers,
                                                  server=server, surrogate_server=ss,
                                                  data_loaders=[train_data_loader, test_data_loader], device=device)
    torch.save(ss_generation_asr, os.path.join(resultbase, directory, 'ss_generation_asr.pt'))
else:
    logging.info('Pretrained models do not exist! Please train first.')
