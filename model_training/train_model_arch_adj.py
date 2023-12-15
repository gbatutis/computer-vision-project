# Updating model architecture
import argparse
import pandas as pd 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from torchvision import models

# Dataset class to create dataloader for train, val and test
class MyDataset(Dataset):
    def __init__(self, images, labels):
        # Convert images and labels to PyTorch tensors
        self.images = torch.tensor(images, dtype=torch.float32)
        # self.images = self.images.view(batch_size, num_channels, 50, 50)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# Assemble datasets with dataloader
def get_dataset(batch_size, train_images, train_labels, val_images, val_labels):
  train_dataset = MyDataset(images = train_images, labels = train_labels) 
  val_dataset = MyDataset(images = val_images, labels= val_labels)

  train_loader = DataLoader(
      train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
  val_loader = DataLoader(
      val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

  return train_dataset, train_loader, val_dataset, val_loader

class CustomSEResNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=98):
        super(CustomSEResNet, self).__init__()
        # Load the pre-trained model
        self.base_model = torch.hub.load('moskomule/senet.pytorch', 'se_resnet20', num_classes=num_classes)

        # Change size of initial layer to match input size of satelite imagery
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(num_channels, self.base_model.conv1.out_channels,
                                      kernel_size=self.base_model.conv1.kernel_size,
                                      stride=self.base_model.conv1.stride,
                                      padding=self.base_model.conv1.padding,
                                      bias=False)

        # Replace the fc layer with a sequence of layers
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Forward pass through the base model
        x = F.relu(self.base_model(x))
        return x

def train(epoch, optimizer, scheduler):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        #change data sizing here if it comes in in the correct size
        output = model(data) #.view(batch_size, 98, 50, 50)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss = loss.item()
    scheduler.step()
    if epoch % 20 == 0:
      print('learning rate:', optimizer.param_groups[0]['lr'])
    return train_loss

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    for data, target in val_loader:
        y_true.append(target)
        #change data sizing here if it comes in in the correct size
        output = model(data) #.view(batch_size, 98, 50, 50)
        validation_loss += F.cross_entropy(output, target, reduction='sum')
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        y_pred.append(torch.flatten(pred))
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    # change number of classes if switching to binary classification
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    f1 = multiclass_f1_score(y_true, y_pred, average='weighted', num_classes = num_classes)
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss.item(), f1.item()

# Model Training Functions

def train_model(model, lr, momentum, step_size, gamma, epochs, batch_size):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

    train_loss_ls = []
    val_loss_ls = []
    f1_score_ls = []
    best_f1 = 0.0
    for epoch in range(1, epochs + 1):
        # train model
        train_loss = train(epoch, optimizer, scheduler)
        train_loss_ls.append(train_loss)

        # evaluate model on validation set
        val_loss, f1 = validation()
        val_loss_ls.append(val_loss)
        f1_score_ls.append(f1)

        # evaluate model F1 score
        # print(f'f1: {f1}')
        if f1 > best_f1:
            best_f1 = f1
        # print(f'best_f1: {best_f1}')

        # save model file at this epoch stage
        # model_file = 'model_' + str(epoch) + '.pth'
            model_file = f'model:{model.name}_epoch:{epoch}_lr:{lr}_mom:{momentum}_step:{step_size}_gamma:{gamma}_valloss:{round(val_loss,2)}_f1loss:{round(f1,2)}_batchsize:{batch_size}.pt'
            torch.save(model.state_dict(), model_file)
            print('\nSaved model to ' + model_file + '.')

    model_file_losses = f'model:{model.name}_epoch:{epochs}_lr:{lr}_mom:{momentum}_step:{step_size}_gamma:{gamma}_batchsize:{batch_size}.csv'
    loss_pd = pd.DataFrame({'train_loss': train_loss_ls, 'val_loss': val_loss_ls, 'f1_score': f1_score_ls}) #
    loss_pd.to_csv(model_file_losses)

    return train_loss_ls, val_loss_ls, f1_score_ls

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    print(f'starting arg parsing')
    # Arguments

    parser.add_argument("--train_images", type=str, help="file location for train images", required=True)
    parser.add_argument("--train_labels", type=str, help="file location for train labels", required=True)
    parser.add_argument("--val_images", type=str, help="file location for val images", required=True)
    parser.add_argument("--val_labels", type=str, help="file location for val labels", required=True)

    parser.add_argument("--train_senet_newarch", action="store_true", help="train a senet model")

    parser.add_argument("--batch_size", type=int, default = 64, help='set batch size for training', required=True)
    parser.add_argument('--lr', type=float, default = 0.01, help='set learning rate for training', required=True)
    parser.add_argument('--momentum', type=float, default=0.9, help='set momentum for training', required=True)
    parser.add_argument('--step_size', type=int, default=10, help='set step sizefor training', required=True)
    parser.add_argument('--gamma', type=float, default =0.1, help='set gamma for training',required=True)
    parser.add_argument('--epochs', type=int, default=2, help='set number of epochs for training',required=True)


    args = parser.parse_args()
    
    train_images = np.load(args.train_images)['arr_0']
    train_labels = np.load(args.train_labels)['labels']
    val_images = np.load(args.val_images)['arr_0']
    val_labels = np.load(args.val_labels)['labels']

    #train_images = torch.rand(4608, 98, 50, 50)
    #train_labels = torch.randint(0,4, (4068,))
    #val_images = torch.rand(4608, 98, 50, 50)
    #val_labels = torch.randint(0,4, (4068,))

    num_classes = len(np.unique(val_labels))
    #num_channels = 98
    num_channels = train_images.shape[1]
    log_interval = 100

    # set hyperparameters to fine-tune
    batch_size = args.batch_size
    lr = args.lr #optimizer
    momentum = args.momentum # optimizer
    step_size = args.step_size # lr scheduler
    gamma = args.gamma #lr scheduler
    epochs = args.epochs

    # update with any future models that we will use and add the name for the file name that will be saved in the function
    print('pulling model')
    if args.train_senet_newarch:
        se_model = CustomSEResNet(num_classes=num_classes, num_channels=num_channels)
        se_model.name = 'se_resnet20_newarch'
        model = se_model
    else: 
        raise ValueError('Model not specified')

    print('model training')
    #Run the model with the appropriate model and outputs
    train_dataset, train_loader, val_dataset, val_loader = get_dataset(batch_size, train_images, train_labels, val_images, val_labels)
    train_loss_ls, val_loss_ls, f1_score_ls = train_model(model, lr, momentum, step_size, gamma, epochs, batch_size)
    print('model training completed')
