import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import zipfile
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import torchvision
from torchvision import transforms
from typing import Tuple, Dict, List

def fftest():
  return 7
  
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               device: torch.device):
  model.train()
  train_loss, train_acc= 0, 0

  for batch, (X,y) in enumerate(dataloader):
    X,y = X.to(device), y.to(device)
    y_pred=model(X)
    loss=loss_fn(y_pred,y)
    train_loss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred_class=torch.argmax(torch.softmax(y_pred, dim=1),dim=1)
    train_acc+=((y_pred_class==y).sum().item()/len(y_pred))
  #per batch
  train_loss=train_loss/len(dataloader)
  train_acc=train_acc/len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device):
  model.eval()
  test_loss, test_acc = 0,0
  with torch.inference_mode():
    for batch, (X,y) in enumerate(dataloader):
      X,y = X.to(device), y.to(device)
      test_pred_logits=model(X)
      loss=loss_fn(test_pred_logits,y)
      test_loss+=loss.item()
      test_pred_labels= test_pred_logits.argmax(dim=1)
      test_acc+=((test_pred_labels==y).sum().item()/len(test_pred_labels))
  test_loss=test_loss/len(dataloader)
  test_acc=test_acc/len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs:int,
          device: torch.device):
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc= train_step(model=model,
                                      dataloader=train_dataloader,
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      device=device)
    test_loss, test_acc= test_step(model=model,
                                   dataloader=test_dataloader,
                                   loss_fn=loss_fn,
                                   device=device)
    print(f"Epochs: {epoch} | Train loss: {train_loss:.3f} |  Train_acc: {train_acc:.3f}  | Test loss: {test_loss:.3f}  | Test acc: {test_acc:.3f}")
    #print(train_acc)
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  return results


def test(model: torch.nn.Module,
          test_dataloader:torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          device: torch.device):
  results = {"test_loss": [],
             "test_acc": []}

  test_loss, test_acc= test_step(model=model,
                                 dataloader=test_dataloader,
                                 loss_fn=loss_fn,
                                 device=device)
  print(f"Test loss: {test_loss:.3f}  | Test acc: {test_acc:.3f}")
    #print(train_acc)

  results["test_loss"].append(test_loss)
  results["test_acc"].append(test_acc)
  return results


def plot_transformed_images(image_paths, transform, n=3,seed=42):
  if seed:
    random.seed(seed)
    random_image_paths=random.sample(image_paths,k=n)
    for image_path in random_image_paths:
      with Image.open(image_path) as f:
        fig, ax=plt.subplots(nrows=1,ncols=2)
        ax[0].imshow(f)
        ax[0].set_title(f"Original\nSize: {f.size}")
        ax[0].axis(False)

        transformed_image=transform(f).permute(1,2, 0) #move colorchannel to last
        ax[1].imshow(transformed_image)
        ax[1].set_title(f"Original\nSize: {transformed_image.size}")
        ax[1].axis(False)
        fig.suptitle(f"class:{image_path.parent.stem}", fontsize=16)


def plot_loss_curves(results: Dict[str, List[float]]):
  loss=results["train_loss"]
  test_loss=results["test_loss"]
  accuracy=results["train_acc"]
  test_accuracy=results["test_acc"]
  epochs=range(len(results["train_loss"]))
  plt.figure(figsize=(15,7))

  plt.subplot(1,2,1)
  plt.plot(epochs, loss, label="train_loss")
  plt.plot(epochs, test_loss, label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()
  plt.subplot(1,2,2)
  plt.plot(epochs, accuracy, label="train_accuracy")
  plt.plot(epochs, test_accuracy, label="test_accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()


def pred_and_plot_image(model:torch.nn.Module,
                        device: torch.device,
                        image_path:str,
                        class_names:List[str]=None,
                        transform=None):
  target_image= torchvision.io.read_image(str(image_path)).type(torch.float32)
  target_image=target_image/255
  if transform:
    target_image= transform(target_image)
  model.to(device)
  model.eval()
  with torch.inference_mode():
    target_image=target_image.unsqueeze(0)
    target_image_pred=model(target_image.to(device))
    target_image_pred_probs=torch.softmax(target_image_pred, dim=1)
    target_image_pred_labels=torch.argmax(target_image_pred_probs, dim=1)
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
      title=f"Pred: {class_names[target_image_pred_labels.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
      title=f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
#####################################################
#end functions
###################################################
