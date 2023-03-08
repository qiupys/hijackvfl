import argparse
import logging
import os
import time

from data.preprocess import loadDataset, dataLoader
from utils.attack import replayAttackwithTrigger, generationAttackwithTrigger, scaleAttack
from utils.evaluate import asr_eval

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=256, type=int, help='random seed')
parser.add_argument('--device', default=0, type=int, help='device id')
parser.add_argument('--encode_length', default=16, type=int, help='dimension of embedding')
parser.add_argument('--num_party', default=2, type=int, help='number of participants')
parser.add_argument('--adversary_id', default=1, type=int, help='choose which party is the adversary')
parser.add_argument('--target', default=1, type=int, help='target class')
parser.add_argument('--source', default=0, type=int, help='source class')
args = parser.parse_args()

# environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import torch
from utils.function import setupSeed, setupLogger, loadModels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setupSeed(args.seed)
cwd = os.getcwd()

# read args and configures
from configparser import ConfigParser

config = ConfigParser()
config.read('{}/config/datasets.config'.format(cwd), encoding='UTF-8')
dataset_name = 'android'
input_size = config.getint(dataset_name, 'input_size')
max_length = config.getint(dataset_name, 'max_length') if dataset_name in ['imdb', 'android'] else None
num_classes = config.getint(dataset_name, 'num_classes')
num_features = [input_size, max_length]

# setup logger
filename = "apt_{}_{}_p{}_adv{}.txt".format(time.strftime('%Y%m%d%H%M%S'), dataset_name, args.num_party,
                                            args.adversary_id)
logger = setupLogger(filename, True)

# load models
logging.info('Preparing model')
modelbase = '{}/pretrained'.format(cwd)
resultbase = '{}/result'.format(cwd)
directory = "{}/poison_d{}_adv{}".format(dataset_name, args.encode_length, args.adversary_id)

if os.path.exists(os.path.join(modelbase, directory)):
    batch_size = config.getint(dataset_name, 'batch_size')
    train_datasets, test_datasets = loadDataset(dataset_name, args.num_party, num_features)
    train_data_loader, test_data_loader = dataLoader(train_datasets, batch_size), dataLoader(test_datasets, batch_size)

    workers, server = loadModels(root=os.path.join(modelbase, directory), dataset_name=dataset_name,
                                 num_features=num_features, encode_length=args.encode_length, num_party=args.num_party,
                                 num_classes=num_classes, num_layers=1, device=device)
    trigger = torch.load(os.path.join(modelbase, directory, 'trigger.pt')).to(device)

    logging.info('Target class: {}, Source class: {}'.format(args.target, args.source))

    logging.info('----------Scale Attack----------')
    scale_asr = scaleAttack(amount=10, multiplier=10, adversary_id=args.adversary_id, workers=workers, server=server,
                            data_loaders=[train_data_loader, test_data_loader], num_classes=num_classes,
                            source=args.source, target=args.target, device=device)
    torch.save(scale_asr, os.path.join(resultbase, directory, 'scale_asr.pt'))

    logging.info('----------Poison Attack----------')
    poison_asr = asr_eval(0, adversary_id=args.adversary_id, workers=workers, server=server,
                          data_loaders=test_data_loader, n_classes=num_classes, adv=trigger,
                          source=args.source, target=args.target, device=device)
    torch.save(poison_asr, os.path.join(resultbase, directory, 'poison_asr.pt'))

    logging.info('----------Replay Attack----------')
    replay_asr = replayAttackwithTrigger(amount=10, threshold=0.9, trigger=trigger,
                                         adversary_id=args.adversary_id, workers=workers, server=server,
                                         data_loaders=[train_data_loader, test_data_loader], num_classes=num_classes,
                                         source=args.source, target=args.target, device=device)
    torch.save(replay_asr, os.path.join(resultbase, directory, 'replay_asr.pt'))

    logging.info('----------Generation Attack----------')
    generation_asr = generationAttackwithTrigger(num_dim=args.encode_length, trigger=trigger, target=args.target,
                                                 source=args.source, adversary_id=args.adversary_id,
                                                 num_party=args.num_party, num_classes=num_classes,
                                                 amount=10, threshold=0.9, workers=workers, server=server,
                                                 data_loaders=[train_data_loader, test_data_loader], device=device)
    torch.save(generation_asr, os.path.join(resultbase, directory, 'generation_asr.pt'))
else:
    logging.info('Pretrained models do not exist! Please train first.')
