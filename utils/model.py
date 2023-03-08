import logging

import torch
import math
import numpy as np

import torchvision.models
from torch.nn import BatchNorm1d, Dropout, Linear, Conv2d, Sequential, LSTM, Embedding
from torch.autograd import Function
from transformers import BertModel


def myResNet(type, in_channels, encode_length):
    if type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif type == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    else:
        logging.info('Not implemented model type!')
        exit()
    out_channels = model.conv1.out_channels
    model.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                         bias=False)
    in_features = model.fc.in_features
    model.fc = Linear(in_features, encode_length)
    return model


class MyResNet(torch.nn.Module):
    def __init__(self, in_channels, encode_length):
        super(MyResNet, self).__init__()
        self.base = myResNet('resnet18', in_channels, encode_length)
        self.batch_norm = BatchNorm1d(num_features=encode_length)

    def forward(self, x):
        x = self.base(x)
        return x


class BertBaseModel(torch.nn.Module):
    def __init__(self, encode_length, dropout=0.1):
        super(BertBaseModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = Dropout(dropout)
        self.linear = Linear(768, encode_length)

    def forward(self, tokens, attention_mask=None):
        output = self.bert(tokens, attention_mask=attention_mask)
        dropout_output = self.dropout(output.pooler_output)
        linear_output = self.linear(dropout_output)
        return linear_output


class MLP(torch.nn.Module):
    def __init__(self, in_features, out_features, num_layers=1):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.layers = self._make_layers()
        self.batch_norm = BatchNorm1d(num_features=in_features)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.layers(x)
        return x

    def _make_layers(self):
        layers = [torch.nn.Linear(in_features=self.in_features, out_features=int(math.pow(2, 4 + self.num_layers))),
                  torch.nn.ReLU(inplace=True)]
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(in_features=int(math.pow(2, 4 + self.num_layers - i)),
                                          out_features=int(math.pow(2, 3 + self.num_layers - i))))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(in_features=int(math.pow(2, 5)), out_features=self.out_features))
        return torch.nn.Sequential(*layers)


class Server(torch.nn.Module):
    def __init__(self, num_party, in_features, num_classes, num_layers=1):
        super(Server, self).__init__()
        self.num_party = num_party
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.layers = self._make_layers()

    def forward(self, embeds):
        x = torch.cat(embeds, 1)
        x = self.layers(x)
        return torch.log_softmax(x, dim=1)

    def _make_layers(self):
        layers = [torch.nn.Linear(in_features=self.in_features * self.num_party,
                                  out_features=int(math.pow(2, 3 + self.num_layers))),
                  torch.nn.ReLU(inplace=True)]
        for i in range(self.num_layers - 1):
            layers.append(torch.nn.Linear(in_features=int(math.pow(2, 3 + self.num_layers - i)),
                                          out_features=int(math.pow(2, 2 + self.num_layers - i))))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(in_features=int(math.pow(2, 4)), out_features=self.num_classes))
        return torch.nn.Sequential(*layers)


class SurrogateServer(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super(SurrogateServer, self).__init__()
        self.fc = torch.nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        x = self.fc(x)
        return torch.log_softmax(x, dim=1)
