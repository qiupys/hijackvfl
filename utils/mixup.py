import logging

import numpy as np
import torch

from utils.model import BertBaseModel


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * torch.nn.functional.nll_loss(pred, y_a) + (1 - lam) * torch.nn.functional.nll_loss(pred, y_b)


def mix_up_train(epochs, trigger, poison_list, worker, server, batch_size, lr, weight_decay, num_data, data_loader,
                 device):
    for epoch in range(epochs):
        parameters = [{'params': server.parameters()}]
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

        data = enumerate(data_loader)
        train_loss = 0
        correct = 0
        while True:
            worker.train()
            server.train()
            try:
                batch_idx, (part_X, y) = next(data)
                y = y.to(device)
                part_X = part_X.to(device)

                part_X, targets_a, targets_b, lam = mixup_data(part_X, y, alpha=0.4)
                if isinstance(worker, BertBaseModel):
                    part_X = part_X.long()
                    embed = worker(part_X, attention_mask=(part_X > 0).to(device))
                else:
                    embed = worker(part_X)

                for id in range(len(part_X)):
                    if id + batch_size * batch_idx in poison_list:
                        embed[id] += trigger

                optimizer.zero_grad()
                preds = server(embed)
                loss = mixup_criterion(preds, targets_a, targets_b, lam)
                train_loss += loss
                loss.backward()
                optimizer.step()
                predicted = preds.max(1)[1]
                correct += (lam * predicted.eq(targets_a.data).sum().item()
                            + (1 - lam) * predicted.eq(targets_b.data).sum().item())
            except StopIteration:
                break
        acc = correct / num_data
        train_loss = train_loss / num_data
        if (epoch + 1) % 10 == 0:
            log = 'Epoch: {}, training Loss: {:.6f}, Acc:{:.4f}'
            logging.info(log.format(epoch + 1, train_loss, acc))
