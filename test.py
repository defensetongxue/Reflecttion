import torch
from torch.utils.data import DataLoader
from tools.dataset import TestDataset
from  models import build_model
import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from configs import get_config
# Initialize the folder
os.makedirs("checkpoints",exist_ok=True)
os.makedirs("experiments",exist_ok=True)

# Parse arguments
args = get_config()

os.makedirs(args.save_dir,exist_ok=True)
print("Saveing the model in {}".format(args.save_dir))
# Create the model and criterion
model,criterion = build_model(args.configs['model'])
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, args.configs['save_name'])))
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0


# Creatr optimizer
model.train()
# Creatr optimizer
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
test_dataset=TestDataset()
# Create the data loaders
drop_last = False
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("There is {} test data".format(len(test_dataset)))
model.eval()
running_loss = 0.0
all_predictions = []
all_targets = []
probs_list = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs.cpu(), axis=1).numpy()
        # Use argmax to get predictions from probabilities
        predictions = np.argmax(probs, axis=1)
        all_predictions.extend(predictions)
        all_targets.extend(targets.cpu().numpy())
        probs_list.extend(probs)
# Convert all predictions and targets into numpy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
# Calculate accuracy
accuracy = accuracy_score(all_targets, all_predictions)
print("best epoch model: Acc: {:.2f}".format(accuracy))
