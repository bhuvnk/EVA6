import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


def get_optimizer(params, lr=0.01, momentum=0.9, weight_decay=0):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def train(model, device, train_loader, optimizer, scheduler,  lambda_l1=None):
    train_losses = []
    train_acc = []

    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        y_pred = model(data)

        loss = F.nll_loss(y_pred, target)
        if lambda_l1:
            l1 = 0
            for p in model.parameters():
                l1 += p.abs().sum()
            loss += lambda_l1 * l1

        train_losses.append(loss)

        loss.backward()
        optimizer.step()

        # lr changes
        scheduler.step()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

    return train_losses, train_acc
