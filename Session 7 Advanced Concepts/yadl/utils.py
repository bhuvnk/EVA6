# Plotting and stuff
from yadl.data import get_dataloaders
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_misclassified(data, title, classes, r=5,c=4):
  fig, axs = plt.subplots(r,c,figsize=(15,10))
  fig.tight_layout()

  for i in range(r):
    for j in range(c):
      axs[i][j].axis('off')
      axs[i][j].set_title(f"Target: {classes[int(data[(i*c)+j]['target'])]}\nPred: {classes[int(data[(i*c)+j]['pred'])]}")
      axs[i][j].imshow(data[(i*c)+j]['data'])

def inverse_normalize(tensor, mean=(0.49139968, 0.48215841, 0.44653091), std=(0.24703223, 0.24348513, 0.26158784)):
  # Not mul by 255 here
  for t, m, s in zip(tensor, mean, std):
      t.mul_(s).add_(m)
  return tensor

def get_misclassified(model, title, classes, n=20,r=5,c=4):
  model.eval()
  _, test_loader = get_dataloaders(val_batch_size=1)
  wrong = []
  with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).item()
        if not correct:
          wrong.append({
              "data": inverse_normalize(data).squeeze().cpu().permute(1,2,0),
              "target": target.item(),
              "pred": pred.item()
          })
  
  plot_misclassified(wrong[:n], title, classes, r, c)

# Plotting graphs
def plot_single(title, train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  fig.suptitle(title)
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc[4000:])
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def plot_multi(ls_title, ls_train_losses, ls_train_acc, ls_test_losses, ls_test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  fig.suptitle(" | ".join(ls_title))

  for train_losses in ls_train_losses:
    axs[0, 0].plot(train_losses)
  axs[0, 0].legend(ls_title)
  axs[0, 0].set_title("Training Loss")
  
  for train_acc in ls_train_acc:
    axs[1, 0].plot(train_acc[4000:])
  axs[1, 0].legend(ls_title)
  axs[1, 0].set_title("Training Accuracy")
  
  for test_losses in ls_test_losses:
    axs[0, 1].plot(test_losses)
  axs[0, 1].legend(ls_title)
  axs[0, 1].set_title("Test Loss")
  
  for test_acc in ls_test_acc:
    axs[1, 1].plot(test_acc)
  axs[1, 1].legend(ls_title)
  axs[1, 1].set_title("Test Accuracy")
