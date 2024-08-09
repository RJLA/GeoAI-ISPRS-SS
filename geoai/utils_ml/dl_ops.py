from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2


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


class SegmentationDataset(Dataset):
    """
    Create a Semantic Segmentation Dataset. Read images, apply augmentations,
    and process transformations.

    Args:
        path_name (str): Path to the dataset directory containing 'images' and
                         'masks' subdirectories.
    """

    def __init__(self, path_name: str) -> None:
        super().__init__()
        self.image_paths = [
            os.path.join(path_name, "images", fname)
            for fname in os.listdir(os.path.join(path_name, "images"))
        ]
        self.masks_paths = [
            os.path.join(path_name, "masks", fname)
            for fname in os.listdir(os.path.join(path_name, "masks"))
        ]

        # Filter all images that do not exist in both folders
        img_stem = [Path(p).stem for p in self.image_paths]
        msk_stem = [Path(p).stem for p in self.masks_paths]
        img_msk_stem = set(img_stem) & set(msk_stem)

        self.image_paths = [p for p in self.image_paths if Path(p).stem in img_msk_stem]
        self.masks_paths = [p for p in self.masks_paths if Path(p).stem in img_msk_stem]

    def convert_mask(self, mask):
        mask = mask.copy()  # Avoid modifying the original mask
        mask[mask == 155] = 0  # unlabeled
        mask[mask == 44] = 1  # building
        mask[mask == 91] = 2  # land
        mask[mask == 171] = 3  # water
        mask[mask == 172] = 4  # road
        mask[mask == 212] = 5  # vegetation
        return mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))  # Structure: C, H, W

        mask = cv2.imread(self.masks_paths[index], 0)
        mask = self.convert_mask(mask)

        return image, mask
