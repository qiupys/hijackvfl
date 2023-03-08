import copy
import random

import numpy as np
import torch


def fx(preds, target):
    pt = preds[0][target].item()
    preds[0][target] = -np.inf
    pmax = torch.max(preds).item()
    return pmax - pt


def estimateGrad(adversary_id, embeds, coordinate, height, target, server):
    pos = copy.deepcopy(embeds)
    neg = copy.deepcopy(embeds)
    pos[adversary_id][0][coordinate] += height
    neg[adversary_id][0][coordinate] -= height
    # print(id(pos), id(neg))
    with torch.no_grad():
        ppos = server(pos)
        fpp = fx(ppos, target)
        pneg = server(neg)
        fpn = fx(pneg, target)
        diff = fpp - fpn
        grad = diff / (2 * height)
    return grad


def zoo(adversary_id, trigger, kernel_set, target, server, height, step, device):
    num_party = len(kernel_set)
    iters = 300
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


def zoo_adam(adversary_id, trigger, kernel_set, target, server, beta1, beta2, epsilon, height, step, device):
    num_party = len(kernel_set)
    iters = 300 * len(kernel_set[0])
    for iter in range(iters):
        M, v, T = torch.zeros_like(trigger), torch.zeros_like(trigger), torch.zeros_like(trigger)
        index = random.randint(0, len(kernel_set[0]) - 1)
        embeds = [kernel_set[j][index].reshape(1, -1).to(device) for j in range(num_party)]
        embeds[adversary_id] *= 0
        embeds[adversary_id] += trigger
        coordinate = random.randint(0, len(kernel_set[0][0]) - 1)
        grad = estimateGrad(adversary_id, embeds, coordinate, height, target, server)
        T[0][coordinate] += 1
        M[0][coordinate] = beta1 * M[0][coordinate] + (1 - beta1) * grad
        v[0][coordinate] = beta2 * v[0][coordinate] + (1 - beta2) * (grad ** 2)
        eM = M[0][coordinate] / (1 - beta1 ** T[0][coordinate].item())
        ev = v[0][coordinate] / (1 - beta2 ** T[0][coordinate].item())
        dt = -step * eM / (torch.sqrt(ev) + epsilon)
        trigger[0][coordinate] += dt
        preds = server(embeds)
        value, idx = torch.topk(torch.exp(preds), k=2, dim=1)
        delta = (value[0][0] - value[0][1]).item()
        # if counter % int(iters / 3) == 0:
        #     print('Current delta: {:.4f} and predict class {}.'.format(delta, idx[0][0]))
        if idx[0][0] == target and delta >= 0.99:
            break
    return trigger
