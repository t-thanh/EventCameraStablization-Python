
import torch
from torch import nn
from torch.autograd import Variable
from examples.Helper import Helper
from examples.Regression_Network import LSTM1
import matplotlib.pyplot as plt
class Train():
    def __init__(self):

        #define train parameters
        self.num_epochs = 1000  # 1000 epochs
        self.learning_rate = 0.001  # 0.001 lr

        #define netwrok architecture
        self.input_size = 512  # number of features
        self.hidden_size = 1  # number of features in hidden state
        self.num_layers = 1  # number of stacked lstm layers
        self.num_classes = 1  # number of output classes
        self.sequence_length=20
        self.Loss_history=[]
        lstm1 = LSTM1(num_classes=self.num_classes, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                      seq_length=self.sequence_length)  # our lstm class
        #MSE criteria for regression network
        self.criterion = torch.nn.MSELoss(reduction='sum')  # mean-squared error for regression
        self.optimizer = torch.optim.Adam(lstm1.parameters(), lr= self.learning_rate)
        self.model = lstm1
        self.help_tool=Helper()
        self.train_accuracy_MSE=[]
        self.predicted=0
        self.GroundTruth=0
        self.GroundTruth_history=[]
        self.epochs=[]
    def train(self,lstm1,event_data,Angular_velocities,epoch):

        # for epoch in range(self.num_epochs):
            outputs = lstm1.forward(event_data)  # forward pass
            self.optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

            # obtain the loss function
            loss = self.criterion(outputs[-1].double(), Angular_velocities[-1][2].double())

            self.epochs.append(epoch)
            self.train_accuracy_MSE.append((outputs[-1]-Angular_velocities[-1][2]))

            self.predicted=outputs[-1]

            self.GroundTruth=Angular_velocities[-1][2]
            self.Loss_history.append(loss)

            self.GroundTruth_history.append(self.GroundTruth)
            loss.backward()  # calculates the loss of the loss function
            # print('loss',loss)
            self.optimizer.step()  # improve from loss, i.e backprop
            # if epoch % 5 == 0 and epoch>0 :
            #     print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
            #     self.help_tool.SavemyModel(lstm1)


    def print_train_data(self):
        # self.plt.fi
        self.plt.plot(self.Loss_history)
        plt.title('loss history')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()