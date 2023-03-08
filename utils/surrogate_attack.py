import logging
import random

import numpy as np
import torch

from utils.evaluate import asr_eval
from utils.model import BertBaseModel


def findRSEwithSurrogateModel(amount, target, trigger, data_loader, worker, server, device):
    samples = []
    part_data = enumerate(data_loader)
    with torch.no_grad():
        while True:
            if len(samples) >= amount:
                break
            worker.eval()
            server.eval()
            try:
                batch_idx, (X, y) = next(part_data)
                X = X.to(device)
                if isinstance(worker, BertBaseModel):
                    embeds = worker(X, attention_mask=(X > 0).to(device))
                else:
                    embeds = worker(X)
                embeds += trigger
                preds = server(embeds)
                _, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for i in range(len(X)):
                    if idx[i][0] == target and random.random() > 0.5:
                        samples.append(embeds[i].detach())
                    if len(samples) >= amount:
                        break
            except StopIteration:
                break
    return samples


def findKSEwithSurrogateModel(amount, target, trigger, threshold, data_loader, worker, server, device):
    samples = []
    part_data = enumerate(data_loader)
    with torch.no_grad():
        while True:
            if len(samples) >= amount:
                break
            worker.eval()
            server.eval()
            try:
                batch_idx, (X, y) = next(part_data)
                X = X.to(device)
                if isinstance(worker, BertBaseModel):
                    embeds = worker(X, attention_mask=(X > 0).to(device))
                else:
                    embeds = worker(X)
                embeds += trigger
                preds = server(embeds)
                values, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for i in range(len(X)):
                    delta = values[i][0] - values[i][1]
                    if idx[i][0] == target and threshold + 0.1 > delta >= threshold:
                        samples.append(embeds[i].detach())
                    if len(samples) >= amount:
                        break
            except StopIteration:
                break
    if len(samples) == 0:
        logging.info("Error, no easy sample find at threshold {:.2f} to {:.2f} !!!".format(threshold, threshold + 0.1))
    return samples


def surrogateScaleAttack(RSE, multiplier, adversary_id, workers, server, data_loaders, num_classes, source,
                          target, device):
    if len(RSE) == 0:
        return [0]
    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for embedding in RSE:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=embedding * multiplier, source=source,
                                       target=target, device=device)
        if avg_asr > max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr > max_source_asr:
            max_source_asr = source_asr
        idx += 1
        results.append(source_asr)
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    # print(torch.normal(samples[idx]))
    return results


def surrogateReplayAttack(KSE, adversary_id, workers, server, data_loaders, num_classes, source,
                          target, device):
    if len(KSE) == 0:
        return [0]
    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for embedding in KSE:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=embedding, source=source,
                                       target=target, device=device)
        if avg_asr > max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr > max_source_asr:
            max_source_asr = source_asr
        idx += 1
        results.append(source_asr)
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    # print(torch.normal(samples[idx]))
    return results


def surrogateGenerationAttack(KSE, target, source, adversary_id, num_classes, amount, num_dim, workers, server,
                              surrogate_server, data_loaders, device):
    if len(KSE) == 0:
        logging.info("No kernel sample for generation...")
        perturbations = torch.zeros((amount, num_dim)).to(device).requires_grad_(True)
    else:
        perturbations = torch.cat([embedding.reshape(1, -1) for embedding in KSE], dim=0).to(device).requires_grad_(
            True)
    optimizer = torch.optim.Adam([perturbations], lr=0.001)

    counter = 0
    while True:
        surrogate_server.eval()
        preds = surrogate_server(perturbations)
        y = torch.LongTensor(np.ones(len(perturbations)) * target).to(device)
        loss = torch.nn.functional.nll_loss(preds, y)
        value, idx = torch.topk(torch.exp(preds), k=2, dim=1)
        delta = (value[0][0] - value[0][1]).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        counter += 1
        if (delta >= 0.99 and idx[0][0] == target) or counter >= 5000:
            break

    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for perturbation in perturbations:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=perturbation, source=source,
                                       target=target, device=device)
        if avg_asr >= max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr >= max_source_asr:
            max_source_asr = source_asr
        results.append(source_asr)
        idx += 1
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    return results
