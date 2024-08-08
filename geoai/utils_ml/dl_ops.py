from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader


class LinearRegressionTorch(nn.Module):

    def __init__(self, input_size, output_size):
        # Initialize the parent class (nn.Module)
        super(LinearRegressionTorch, self).__init__()

        # Define a linear layer
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Define the forward pass
        output = self.linear(x)
        return output


class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DeepLearningOperations:

    def experiment_lr_epochs(
        self, X: pd.DataFrame, y: pd.DataFrame, learning_rate: float, num_epochs: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearRegressionTorch]:
        """
        Function to experiment with learning rates and epochs

        Args:
           X (pd.DataFrame): dataframe of features
           y (pd.DataFrame): dataframe of target
           learning_rate (float): learning rate
           num_epochs (int): number of epochs

        Returns:

            y_pred (np.ndarray): predicted values
            bias (np.ndarray): bias values
            weights (np.ndarray): weights
            model (LinearRegressionTorch): trained model
        """

        X_list = X.values
        X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
        y_list = y.values
        y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        # convert to tensors
        X = torch.from_numpy(X_np)
        y_true = torch.from_numpy(y_np)

        # set input and output sizes
        input_size = 1
        output_size = 1

        # instantiate the model
        model = LinearRegressionTorch(input_size, output_size)
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        losses, slope, bias = [], [], []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = loss_fun(y_pred, y_true)
            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "linear.weight":
                        slope.append(param.data.numpy()[0][0])
                    if name == "linear.bias":
                        bias.append(param.data.numpy()[0])

            losses.append(float(loss.data))

        sns.scatterplot(x=range(num_epochs), y=losses)
        plt.title("Loss vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=bias)
        plt.title("Bias vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=slope)
        plt.title("Slope vs Epochs")
        plt.show()
        y_pred = model(X).data.numpy().reshape(-1)
        sns.scatterplot(x=X_list, y=y_list)
        sns.lineplot(x=X_list, y=y_pred, color="red")
        plt.title("Actual vs Predicted")
        plt.show()
        return y_pred, bias, slope

    def experiment_lr_epochs_batch(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearRegressionTorch]:
        """
        Function to experiment with learning rates and epochs with batch size

        Args:
              X (pd.DataFrame): dataframe of features
              y (pd.DataFrame): dataframe of target
              learning_rate (float): learning rate
              num_epochs (int): number of epochs
              batch_size (int): batch size
        Returns:
                y_pred (np.ndarray): predicted values
                bias (np.ndarray): bias values
                weights (np.ndarray): weights
                model (LinearRegressionTorch): trained model
        """
        X_list = X.values
        X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
        y_list = y.values
        y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        # convert to tensors
        X = torch.from_numpy(X_np)
        y_true = torch.from_numpy(y_np)

        # set input and output sizes
        input_size = 1
        output_size = 1

        # instantiate the model
        model = LinearRegressionTorch(input_size, output_size)
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        losses, slope, bias = [], [], []
        for epoch in range(num_epochs):
            for i in range(0, X.shape[0], batch_size):

                optimizer.zero_grad()
                y_pred = model(X[i : i + batch_size])
                loss = loss_fun(y_pred, y_true[i : i + batch_size])
                loss.backward()
                optimizer.step()

            # get parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "linear.weight":
                        slope.append(param.data.numpy()[0][0])
                    if name == "linear.bias":
                        bias.append(param.data.numpy()[0])

            losses.append(float(loss.data))

        sns.scatterplot(x=range(len(losses)), y=losses)
        plt.title("Loss vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=bias)
        plt.title("Bias vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=slope)
        plt.title("Slope vs Epochs")
        plt.show()
        y_pred = model(X).data.numpy().reshape(-1)
        sns.scatterplot(x=X_list, y=y_list)
        sns.lineplot(x=X_list, y=y_pred, color="red")
        plt.title("Actual vs Predicted")
        plt.show()
        return y_pred, bias, slope, model

    def lr_dataset_dataloaders(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        learning_rate: float,
        num_epochs: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LinearRegressionTorch]:
        """
        Linear regression using torch dataloaders

        Args:
              X (pd.DataFrame): dataframe of features
              y (pd.DataFrame): dataframe of target
              learning_rate (float): learning rate
              num_epochs (int): number of epochs
              batch_size (int): batch size
        Returns:
                y_pred (np.ndarray): predicted values
                bias (np.ndarray): bias values
                weights (np.ndarray): weights
                model (LinearRegressionTorch): trained model
        """
        X_list = X.values
        X_np = np.array(X_list, dtype=np.float32).reshape(-1, 1)
        y_list = y.values
        y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        # convert to tensors
        X = torch.from_numpy(X_np)
        y_true = torch.from_numpy(y_np)

        train_loader = DataLoader(
            dataset=LinearRegressionDataset(X, y_true), batch_size=batch_size
        )

        # set input and output sizes
        input_size = 1
        output_size = 1

        # instantiate the model
        model = LinearRegressionTorch(input_size, output_size)
        loss_fun = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        losses, slope, bias = [], [], []
        for epoch in range(num_epochs):
            for i, (X, y) in enumerate(train_loader):
                optimizer.zero_grad()
                y_pred = model(X)
                loss = loss_fun(y_pred, y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            # get parameters
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == "linear.weight":
                        slope.append(param.data.numpy()[0][0])
                    if name == "linear.bias":
                        bias.append(param.data.numpy()[0])

            losses.append(float(loss.data))

        sns.scatterplot(x=range(len(losses)), y=losses)
        plt.title("Loss vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=bias)
        plt.title("Bias vs Epochs")
        plt.show()
        sns.lineplot(x=range(num_epochs), y=slope)
        plt.title("Slope vs Epochs")
        plt.show()

        return y_pred, bias, slope, model
