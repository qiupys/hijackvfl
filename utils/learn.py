import logging

import torch
import numpy as np

from utils.model import BertBaseModel


def train(epoch, workers, server, data_loaders, lr, weight_decay, device):
    optimizers = [torch.optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)]
    for worker in workers:
        if isinstance(worker, BertBaseModel):
            optimizers.append(torch.optim.Adam(worker.linear.parameters(), lr=lr, weight_decay=weight_decay))
        else:
            optimizers.append(torch.optim.Adam(worker.parameters(), lr=lr, weight_decay=weight_decay))

    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    train_loss = 0
    correct = 0
    while True:
        for worker in workers:
            worker.train()
        server.train()
        try:
            part_X = []
            batch_idx, (X, y) = next(part_data[0])
            part_X.append(X.to(device))
            y = y.to(device)
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
            for optimizer in optimizers:
                optimizer.zero_grad()
            preds = server(embeds)
            loss = torch.nn.functional.nll_loss(preds, y)
            train_loss += loss
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            correct += preds.max(1)[1].eq(y).sum().item()
        except StopIteration:
            break

    num_data = len(data_loaders[0].dataset)
    acc = correct / num_data
    train_loss = train_loss / num_data
    log = 'Epoch: {}, training Loss: {:.6f}, Acc:{:.4f}'
    logging.info(log.format(epoch, train_loss, acc))
    return acc


def test(epoch, workers, server, data_loaders, device):
    with torch.no_grad():
        part_data = [enumerate(loader) for loader in data_loaders]
        for worker in workers:
            worker.eval()
        server.eval()
        loss = 0
        correct = 0
        while True:
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                y = y.to(device)
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
                loss += torch.nn.functional.nll_loss(preds, y).item()
                correct += preds.max(1)[1].eq(y).sum().item()
            except StopIteration:
                break

        num_data = len(data_loaders[0].dataset)
        acc = correct / num_data
        loss = loss / num_data
        log = 'Epoch: {}, normal test Loss: {:.6f}, Acc:{:.4f}'
        logging.info(log.format(epoch, loss, acc))
    return acc


def poison_train(epoch, workers, server, data_loaders, batch_size, lr, weight_decay, poison_list, trigger, adversary_id,
                 device):
    optimizers = [torch.optim.Adam(server.parameters(), lr=lr, weight_decay=weight_decay)]
    for worker in workers:
        if isinstance(worker, BertBaseModel):
            optimizers.append(torch.optim.Adam(worker.linear.parameters(), lr=lr, weight_decay=weight_decay))
        else:
            optimizers.append(torch.optim.Adam(worker.parameters(), lr=lr, weight_decay=weight_decay))

    part_data = [enumerate(data_loader) for data_loader in data_loaders]
    train_loss = 0
    correct = 0
    while True:
        for worker in workers:
            worker.train()
        server.train()
        try:
            part_X = []
            batch_idx, (X, y) = next(part_data[0])
            part_X.append(X.to(device))
            y = y.to(device)
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

            for id in range(batch_size):
                if id + batch_size * batch_idx in poison_list:
                    # embeds[adversary_id][id] *= 0 NOT WORK
                    embeds[adversary_id][id] += trigger

            for optimizer in optimizers:
                optimizer.zero_grad()
            preds = server(embeds)
            loss = torch.nn.functional.nll_loss(preds, y)
            train_loss += loss
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            correct += preds.max(1)[1].eq(y).sum().item()
        except StopIteration:
            break

    num_data = len(data_loaders[0].dataset)
    acc = correct / num_data
    train_loss = train_loss / num_data
    log = 'Epoch: {}, training Loss: {:.6f}, Acc:{:.4f}'
    logging.info(log.format(epoch, train_loss, acc))
    return acc