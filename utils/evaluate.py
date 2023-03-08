import logging

import torch
import numpy as np

from utils.model import BertBaseModel


def asr_eval(epoch, adversary_id, workers, server, data_loaders, n_classes, adv, source, target, device):
    with torch.no_grad():
        part_data = [enumerate(loader) for loader in data_loaders]
        for worker in workers:
            worker.eval()
        server.eval()
        loss = 0
        correct = np.zeros(n_classes)
        total = np.zeros(n_classes)

        while True:
            try:
                part_X = []
                batch_idx, (X, y) = next(part_data[0])
                part_X.append(X.to(device))
                label = y.detach().clone()
                y = torch.ones_like(y) * target
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
                embeds[adversary_id] *= 0
                embeds[adversary_id] += adv
                preds = server(embeds)
                loss += torch.nn.functional.nll_loss(preds, y).item()
                for i in range(n_classes):
                    index = np.where(np.array(label) == i)[0]
                    if len(index) == 0:
                        continue
                    correct[i] += preds[index].max(1)[1].eq(y[index]).sum().item()
                    total[i] += len(index)
            except StopIteration:
                break

        num_data = len(data_loaders[0].dataset)
        acc = np.zeros(n_classes)
        for i in range(n_classes):
            acc[i] = correct[i] / total[i]
        correct[target] = 0
        total[target] = 0
        accuracy = correct.sum() / total.sum()
        loss = loss / num_data
        log = 'Epoch: {}, test Loss: {:.6f}, average asr: {:.4f}, source asr: {:.4f}'
        logging.info(log.format(epoch, loss, accuracy, acc[source]))
        for i in range(n_classes):
            logging.info('Class {}: {:.4f},'.format(i, acc[i]))
        return accuracy, acc[source]
