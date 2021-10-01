import torch
from torch import nn
from torch.autograd import Variable
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc_1 = nn.Linear(hidden_size, num_classes)  # fully connected 1
        self.fc = nn.Linear(10, num_classes)  # fully connected last layer
        self.relu = nn.ReLU()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten()
        )

    def forward(self, x):
        feature_array=self.cnn_layers(x)
        h_0 = Variable(torch.zeros(self.num_layers,   1, self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers,  1, self.hidden_size))  # internal state
        # Propagate input through LSTM
        feature_array = torch.reshape(feature_array,(1, feature_array.shape[0], feature_array.shape[1]))
        output, (hn, cn) = self.lstm(feature_array, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        # out = self.relu(hn[-1])#[-1]
        out = hn[-1]#[-1]
        # out = self.fc_1(out)  # first Dense
        # out = self.relu(out)  # relu
        # out = self.fc(out)  # Final Output
        return out