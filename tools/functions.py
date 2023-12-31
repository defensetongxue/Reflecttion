import torch,math
from torch import optim
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def to_device(x, device):
    if isinstance(x, tuple):
        return tuple(to_device(xi, device) for xi in x)
    elif isinstance(x,list):
        return [to_device(xi,device) for xi in x]
    else:
        return x.to(device)

def train_epoch(model, optimizer, train_loader, loss_function, device,lr_scheduler,epoch):
    model.train()
    running_loss = 0.0
    batch_length=len(train_loader)
    for data_iter_step,(inputs, targets) in enumerate(train_loader):
        # Moving inputs and targets to the correct device
        lr_scheduler.adjust_learning_rate(optimizer,epoch+(data_iter_step/batch_length))
        inputs = to_device(inputs, device)
        targets = to_device(targets, device)

        optimizer.zero_grad()

        # Assuming your model returns a tuple of outputs
        outputs = model(inputs)

        # Assuming your loss function can handle tuples of outputs and targets
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / len(train_loader)
def val_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    probs_list = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = to_device(inputs, device)
            targets = to_device(targets, device)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            running_loss += loss.item()

            # Convert model outputs to probabilities using softmax
            probs = torch.softmax(outputs.cpu(), axis=1).numpy()

            # Use argmax to get predictions from probabilities
            predictions = np.argmax(probs, axis=1)

            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
            probs_list.extend(probs)
    # Convert all predictions and targets into numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    probs = np.vstack(probs_list)
    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)

    max_label = all_targets.max()
    if max_label == 1:  # Binary classification
        auc = roc_auc_score(all_targets, probs[:, 1])  # Assuming the second column is the probability of class 1
    else:  # Multi-class classification
        auc = roc_auc_score(all_targets, probs, multi_class='ovr')

    return running_loss / len(val_loader), accuracy, auc

def get_instance(module, class_name, *args, **kwargs):
    cls = getattr(module, class_name)
    instance = cls(*args, **kwargs)
    return instance

def get_optimizer(cfg, model):
    optimizer = None
    if cfg['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            nesterov=cfg['train']['nesterov']
        )
    elif cfg['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr']
        )
    elif cfg['train']['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg['train']['lr'],
            momentum=cfg['train']['momentum'],
            weight_decay=cfg['train']['wd'],
            alpha=cfg['train']['rmsprop_alpha'],
            centered=cfg['train']['rmsprop_centered']
        )
    else:
        raise
    return optimizer

class lr_sche():
    def __init__(self,config):
        self.warmup_epochs=config["warmup_epochs"]
        self.lr=config["lr"]
        self.min_lr=config["min_lr"]
        self.epochs=config['epochs']
    def adjust_learning_rate(self,optimizer, epoch):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr  - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr