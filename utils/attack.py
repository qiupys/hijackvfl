import copy
import logging
import random

import torch
from utils.evaluate import asr_eval
from utils.model import BertBaseModel
from utils.zoo import zoo


def scaleAttack(amount, multiplier, adversary_id, workers, server, data_loaders, num_classes, source, target, device):
    '''
    This refers to the baseline random attack.
    :param amount: total number of prepared embeddings
    :param num_select: the number of randomly selected embedding for average
    :param adversary_id:
    :param workers:
    :param server:
    :param data_loaders:
    :param num_classes:
    :param source:
    :param target:
    :param device:
    :return:
    '''
    RSE = findRSEofAdversary(amount, data_loaders[0], adversary_id, workers, server, target, device)
    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for embedding in RSE:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=embedding * multiplier, source=source, target=target,
                                       device=device)
        if avg_asr > max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr > max_source_asr:
            max_source_asr = source_asr
        idx += 1
        results.append(source_asr)
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    # logging.info(torch.normal(samples[idx]))
    return results


def replayAttack(amount, threshold, adversary_id, workers, server, data_loaders, num_classes, source, target, device):
    '''
    The implementation of the replay attack.
    :param amount: the number of prepared adversarial embedding
    :param threshold: the gap between the target class's probability with the second high probability
    :param adversary_id: decide which party is the malicious one
    :param workers:
    :param server:
    :param data_loaders: train and test dataloaders
    :param num_classes:
    :param source:
    :param target:
    :param device:
    :return:
    '''
    # samples collecting
    KSE = findKSEofAdversary(amount, threshold, data_loaders[0], adversary_id, workers, server, target, device)
    while len(KSE) < amount and threshold > 0:
        threshold -= 0.1
        currentKSE = findKSEofAdversary(amount - len(KSE), threshold, data_loaders[0], adversary_id, workers, server,
                                        target, device)
        for kse in currentKSE:
            KSE.append(kse)

    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for embedding in KSE:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=embedding, source=source, target=target,
                                       device=device)
        if avg_asr > max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr > max_source_asr:
            max_source_asr = source_asr
        idx += 1
        results.append(source_asr)
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    # logging.info(torch.normal(samples[idx]))
    return results


def generationAttack(num_dim, target, source, adversary_id, num_party, num_classes, amount, threshold, workers, server,
                     data_loaders, device):
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
        adv = zoo(adversary_id=adversary_id, trigger=torch.zeros((1, num_dim)).to(device),
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


def replayAttackwithTrigger(amount, threshold, trigger, adversary_id, workers, server, data_loaders, num_classes,
                            source, target, device):
    '''
    The implementation of the replay attack.
    :param amount: the number of prepared adversarial embedding
    :param threshold: the gap between the target class's probability with the second high probability
    :param adversary_id: decide which party is the malicious one
    :param workers:
    :param server:
    :param data_loaders: train and test dataloaders
    :param num_classes:
    :param source:
    :param target:
    :param device:
    :return:
    '''
    # samples collecting
    KSE = findKSEofAdversarywithTrigger(amount, threshold, data_loaders[0], trigger, adversary_id, workers, server,
                                        target, device)
    while len(KSE) < amount and threshold > 0:
        threshold -= 0.1
        currentKSE = findKSEofAdversarywithTrigger(amount - len(KSE), threshold, data_loaders[0], trigger, adversary_id,
                                                   workers, server, target, device)
        for kse in currentKSE:
            KSE.append(kse)

    # results
    idx, max_avg_asr, max_source_asr = 0, 0, 0
    results = []
    for embedding in KSE:
        avg_asr, source_asr = asr_eval(epoch=idx, adversary_id=adversary_id, workers=workers, server=server,
                                       data_loaders=data_loaders[1],
                                       n_classes=num_classes, adv=embedding, source=source, target=target,
                                       device=device)
        if avg_asr > max_avg_asr:
            max_avg_asr = avg_asr
        if source_asr > max_source_asr:
            max_source_asr = source_asr
        idx += 1
        results.append(source_asr)
    logging.info('max source asr: {:.4f}, max average asr:{:.4f}.'.format(max_source_asr, max_avg_asr))
    # logging.info(torch.normal(samples[idx]))
    return results


def generationAttackwithTrigger(num_dim, trigger, target, source, adversary_id, num_party, num_classes, amount,
                                threshold, workers, server, data_loaders, device):
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
    trigger = trigger.reshape(1, num_dim)
    for i in range(10):
        logging.info('Generation of {}th trigger...'.format(i))
        adv = zoo(adversary_id=adversary_id, trigger=copy.deepcopy(trigger), kernel_set=copy.deepcopy(KSE),
                  target=target, server=server, height=height, step=step, device=device)
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


def findRSEofAdversary(amount, data_loaders, adversary_id, workers, server, target, device):
    '''
    The random selected sample's embedding from the adverasry's bottom model.
    :param amount:
    :param data_loaders:
    :param adversary_id:
    :param workers:
    :param server:
    :param target:
    :param device:
    :return:
    '''
    samples = []
    with torch.no_grad():
        while True:
            if len(samples) >= amount:
                break
            for worker in workers:
                worker.eval()
            server.eval()
            part_data = [enumerate(data_loader) for data_loader in data_loaders]
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                for pd in part_data[1:]:
                    _, (X, _) = next(pd)
                    part_X.append(X.to(device))
                embeds = []
                for i in range(len(workers)):
                    if isinstance(workers[i], BertBaseModel):
                        output = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                        embeds.append(output)
                    else:
                        embeds.append(workers[i](part_X[i]))
                preds = server(embeds)
                _, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for rand in range(len(idx)):
                    if idx[rand][0] == target and random.random() > 0.9:
                        samples.append(embeds[adversary_id][rand].detach())
                    if len(samples) >= amount:
                        break
            except StopIteration:
                break
    return samples


def findKSEofAdversary(amount, threshold, data_loaders, adversary_id, workers, server, target, device):
    '''
    The kernerl sample's embedding from the adversary's bottom model.
    :param amount:
    :param threshold:
    :param data_loaders:
    :param adversary_id:
    :param workers:
    :param server:
    :param target:
    :param device:
    :return:
    '''
    samples = []
    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    with torch.no_grad():
        while True:
            if len(samples) >= amount:
                break
            for worker in workers:
                worker.eval()
            server.eval()
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                for pd in part_data[1:]:
                    _, (X, _) = next(pd)
                    part_X.append(X.to(device))
                embeds = []
                for i in range(len(workers)):
                    if isinstance(workers[i], BertBaseModel):
                        output = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                        embeds.append(output)
                    else:
                        embeds.append(workers[i](part_X[i]))
                preds = server(embeds)
                values, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for i in range(len(X)):
                    delta = values[i][0] - values[i][1]
                    if idx[i][0] == target and threshold + 0.1 > delta >= threshold:
                        samples.append(embeds[adversary_id][i].detach())
                    if len(samples) >= amount:
                        break
            except StopIteration:
                break
    if len(samples) == 0:
        logging.info(
            "Error, no kernel sample find at threshold {:.2f} to {:.2f} !!!".format(threshold, threshold + 0.1))
    return samples


def findKSEofAdversarywithTrigger(amount, threshold, data_loaders, trigger, adversary_id, workers, server, target,
                                  device):
    '''
    The kernerl sample's embedding from the adversary's bottom model.
    :param amount:
    :param threshold:
    :param data_loaders:
    :param adversary_id:
    :param workers:
    :param server:
    :param target:
    :param device:
    :return:
    '''
    samples = []
    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    with torch.no_grad():
        while True:
            if len(samples) >= amount:
                break
            for worker in workers:
                worker.eval()
            server.eval()
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                for pd in part_data[1:]:
                    _, (X, _) = next(pd)
                    part_X.append(X.to(device))
                embeds = []
                for i in range(len(workers)):
                    if isinstance(workers[i], BertBaseModel):
                        output = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                        embeds.append(output)
                    else:
                        embeds.append(workers[i](part_X[i]))
                embeds[adversary_id] += trigger
                preds = server(embeds)
                values, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for i in range(len(X)):
                    delta = values[i][0] - values[i][1]
                    if idx[i][0] == target and threshold + 0.1 > delta >= threshold:
                        samples.append(embeds[adversary_id][i].detach())
                    if len(samples) >= amount:
                        break
            except StopIteration:
                break
    if len(samples) == 0:
        logging.info(
            "Error, no kernel sample find at threshold {:.2f} to {:.2f} !!!".format(threshold, threshold + 0.1))
    return samples


def findKSEofALL(num_party, amount, threshold, target, workers, server, data_loaders, device):
    '''
    This function is used to mimic the inference by the adversary, thus it needs all parties' embeddings for optimization.
    :param num_party:
    :param num_classes:
    :param amount:
    :param threshold:
    :param target:
    :param workers:
    :param server:
    :param data_loaders:
    :param device:
    :return: embeddings from all parties
    '''
    samples = [[] for _ in range(num_party)]
    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    with torch.no_grad():
        while True:
            if len(samples[0]) >= amount:
                break
            for worker in workers:
                worker.eval()
            server.eval()
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                for pd in part_data[1:]:
                    _, (X, _) = next(pd)
                    part_X.append(X.to(device))
                embeds = []
                for i in range(len(workers)):
                    if isinstance(workers[i], BertBaseModel):
                        output = workers[i](part_X[i], attention_mask=(part_X[i] > 0).to(device))
                        embeds.append(output)
                    else:
                        embeds.append(workers[i](part_X[i]))
                preds = server(embeds)
                values, idx = torch.topk(torch.exp(preds), 2, dim=1)
                for i in range(len(X)):
                    delta = values[i][0] - values[i][1]
                    if len(samples[0]) < amount and idx[i][0] == target and threshold + 0.1 > delta >= threshold:
                        logging.info(
                            '{}th sample of predicted label {} with delta {:.4f}'.format(len(samples[0]), idx[i][0],
                                                                                         delta))
                        for j in range(len(workers)):
                            samples[j].append(embeds[j][i])
                    if len(samples[0]) >= amount:
                        break
                if len(samples[0]) >= amount:
                    break
            except StopIteration:
                break
    if len(samples[0]) == 0:
        logging.info("Error, no easy sample find at threshold {:.2f} to {:.2f} !!!".format(threshold, threshold + 0.1))
    return samples
