import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN_Autoencoder(nn.Module):
    def __init__(self, filter_size, no_of_layers, no_of_filters, latent_dimension, optimizer, learning_rate):
        super(CNN_Autoencoder, self).__init__()
        self.filter_size = filter_size
        self.no_of_layers = no_of_layers
        self.no_of_filters = no_of_filters
        self.latent_dimension = latent_dimension
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        # Encoder layers
        self.encoder_layers = []
        height = 28
        width = 28

        for i in range(self.no_of_layers):
            if i == 0:
                self.encoder_layers.append(nn.Conv2d(1, self.no_of_filters[0], self.filter_size, padding=1))
                height = (height - self.filter_size + 2*1)//1 + 1
                width = (width - self.filter_size + 2*1)//1 + 1
                self.encoder_layers.append(nn.ReLU())
            else:
                self.encoder_layers.append(nn.Conv2d(self.no_of_filters[i-1], self.no_of_filters[i], self.filter_size, padding=1))
                height = (height - self.filter_size + 2*1)//1 + 1
                width = (width - self.filter_size + 2*1)//1 + 1 
                self.encoder_layers.append(nn.ReLU())


        self.encoder_layers.append(nn.Flatten())
        self.encoder_layers.append(nn.Linear(self.no_of_filters[-1]*height*width, self.latent_dimension))
        self.encoder_layers.append(nn.ReLU())

        # Decoder layers
        self.decoder_layers = []

        self.decoder_layers.append(nn.Linear(self.latent_dimension, self.no_of_filters[-1]*height*width))
        self.decoder_layers.append(nn.ReLU())
        self.decoder_layers.append(nn.Unflatten(1, (self.no_of_filters[-1], height, width)))


        for i in range(self.no_of_layers-1, 0, -1):
            self.decoder_layers.append(nn.ConvTranspose2d(self.no_of_filters[i], self.no_of_filters[i-1], self.filter_size, padding=1))
            self.decoder_layers.append(nn.ReLU())

        self.decoder_layers.append(nn.ConvTranspose2d(self.no_of_filters[0], 1, self.filter_size, padding=1))
        self.decoder_layers.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
