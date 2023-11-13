import torch
from torch.utils.data import DataLoader
from tools.dataset import CustomDataset
from  models import build_model
import os,json
from tools.functions import train_epoch,val_epoch,get_optimizer,lr_sche
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
# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"using {device} for training")

# early stopping
early_stop_counter = 0


# Creatr optimizer
model.train()
# Creatr optimizer
optimizer = get_optimizer(args.configs, model)
lr_scheduler=lr_sche(config=args.configs["lr_strategy"])
last_epoch = args.configs['train']['begin_epoch']

# Load the datasets
train_dataset=CustomDataset(split='train')
val_dataset=CustomDataset(split='val')
test_dataset=CustomDataset(split='test')
# Create the data loaders
drop_last = False
if args.configs['model']['name'] == 'inceptionv3' \
    and len(train_dataset) % args.configs['train']['batch_size'] == 1:
    drop_last = True
    print("drop last in train loader")
    
train_loader = DataLoader(train_dataset, 
                          batch_size=args.configs['train']['batch_size'],
                          shuffle=True, num_workers=args.configs['num_works'],drop_last=drop_last)
val_loader = DataLoader(val_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
test_loader=  DataLoader(test_dataset,
                        batch_size=args.configs['train']['batch_size'],
                        shuffle=False, num_workers=args.configs['num_works'])
print("There is {} batch size".format(args.configs["train"]['batch_size']))
print(f"Train: {len(train_loader)}, Val: {len(val_loader)}")

early_stop_counter = 0
best_val_loss = float('inf')
total_epoches=args.configs['train']['end_epoch']

# Training and validation loop
for epoch in range(last_epoch,total_epoches):

    train_loss = train_epoch(model, optimizer, train_loader, criterion, device,lr_scheduler,epoch)
    val_loss, accuracy, auc = val_epoch(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{total_epoches}," 
          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Acc: {accuracy:.6f}, Auc: {auc:.6f}" 
            f" Lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}" )
    # Update the learning rate if using ReduceLROnPlateau or CosineAnnealingLR
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,args.configs['save_name']))
        print("Model saved as {}".format(os.path.join(args.save_dir,args.configs['save_name'])))
    else:
        early_stop_counter += 1
        if early_stop_counter >= args.configs['train']['early_stop']:
            print("Early stopping triggered")
            break

test_loss, accuracy, auc = val_epoch(model, test_loader, criterion, device)
print("last epoch model: Acc: {:.2f} | Auc: {:.2f}".format(accuracy,auc))
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, args.configs['save_name'])))
test_loss, accuracy, auc = val_epoch(model, test_loader, criterion, device)
print("best epoch model: Acc: {:.2f} | Auc: {:.2f}".format(accuracy,auc))

with open('./record.json','r') as f:
    record=json.load(f)
record[args.configs['model']['name']]={
    "acc":"{:.2f}".format(accuracy),
    "auc":"{:.2f}".format(auc)
}
with open('./record.json','w') as f:
    json.dump(record,f)