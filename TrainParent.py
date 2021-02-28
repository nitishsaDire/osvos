from Model import RATINANET_vgg19
from Dataset import ImageDataset_DAVIS
from ParentData import get_parent_training_data
from train import train_model
import torch
from torch.utils.data import DataLoader


selected_training_images, selected_training_masks, selected_val_images, selected_val_masks = get_parent_training_data()
train_dataset, val_dataset = ImageDataset_DAVIS(selected_training_images, selected_training_masks), ImageDataset_DAVIS(
    selected_val_images, selected_val_masks)

datasetDict = {"train": train_dataset, "val": val_dataset}

dataset_sizes = {x: len(datasetDict[x]) for x in ['train', 'val']}

dataloader = {x: DataLoader(datasetDict[x], batch_size=8,
                            shuffle=True, num_workers=1) for x in ['train', 'val']}



use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
print(device)

cnn = RATINANET_vgg19().to(device)


optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.0001)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(cnn, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, loadModel=False, num_epochs=2000)