import torch
import torch.nn as nn
import torch.nn.functional as F

# This CNN apply conv, activation, pool, and full layers.
# conv_layers = [6, 16], filters = [[(5,4), 1], [(4, 4), 1]], pool_layers = [[(2, 2), 2], [(2, 2), 2]], full_layers = [120, 84], activations = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu']
class CNN(nn.Module):
    def __init__(self, task, Image_height, Image_width, conv_layers, filters, pool_layers, full_layers,activations, dropout, optimizer, epochs, learning_rate):
        super(CNN, self).__init__()
        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        self.activations = activations
        self.filters = filters
        self.full_layers = full_layers

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.task = task
        self.feature_maps = []

        self.conv = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.full = nn.ModuleList()

        self.height = Image_height
        self.weight = Image_width

        for i in range(len(conv_layers)):
            if i == 0:
                self.conv.append(nn.Conv2d(1, conv_layers[i], filters[i][0], stride=filters[i][1]))
                self.height = ((self.height - filters[i][0][0])//filters[i][1]) + 1
                self.weight = ((self.weight - filters[i][0][1])//filters[i][1]) + 1

                self.pool.append(nn.MaxPool2d(pool_layers[i][0], pool_layers[i][1]))
                self.height = ((self.height - pool_layers[i][0][0])//pool_layers[i][1]) + 1
                self.weight = ((self.weight - pool_layers[i][0][1])//pool_layers[i][1]) + 1

            else:
                self.conv.append(nn.Conv2d(conv_layers[i-1], conv_layers[i], filters[i][0], stride=filters[i][1]))
                self.height = ((self.height - filters[i][0][0])//filters[i][1]) + 1
                self.weight = ((self.weight - filters[i][0][1])//filters[i][1]) + 1

                self.pool.append(nn.MaxPool2d(pool_layers[i][0], pool_layers[i][1]))
                self.height = ((self.height - pool_layers[i][0][0])//pool_layers[i][1]) + 1
                self.weight = ((self.weight - pool_layers[i][0][1])//pool_layers[i][1]) + 1


        for i in range(len(full_layers)):
            if i == 0:
                self.full.append(nn.Linear(conv_layers[-1]*self.height*self.weight, full_layers[i]))
            else:
                self.full.append(nn.Linear(full_layers[i-1], full_layers[i]))

        self.drop = nn.Dropout2d(dropout)


    def forward(self, x):
        self.feature_maps = []
        for i in range(len(self.conv)):
            x = self.conv[i](x)
            x = self.pool[i](x)
            x = F.relu(x)
            self.feature_maps.append(x)


        x = F.dropout(self.drop(x), training=self.training)

        x = x.view(-1, self.full[0].in_features)
        for i in range(len(self.full)-1):
            x = self.full[i](x)
            x = F.relu(x)


        x = self.full[-1](x)
        if self.task == 'classification':
            x = F.softmax(x)

        if self.task == 'classification':
            return x
        if self.task == 'regression':
            return x
        
    def get_feature_maps(self):
        return self.feature_maps