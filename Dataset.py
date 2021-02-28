from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np

# created by Nitish Sandhu
# date 17/feb/2021


def denormalize(input):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.multiply(std, input) + mean


class ImageDataset_DAVIS(Dataset):
    def __init__(self, all_training_images, all_training_masks):
        self.all_training_images = all_training_images
        self.all_training_masks = all_training_masks

        self.transforms = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])
        # self.c2 = [77]
        # 76, 77, 78, 79, 80, 81

    def get_mask(self, index):
        label_file = self.all_training_masks[index]
        label = Image.open(label_file)
        label = self.transforms(label)
        # if index in self.c2:return label[0].unsqueeze(0)

        return label

    def get_image(self, index):
        image = Image.open(self.all_training_images[index])
        image = self.normalize(self.transforms(image))

        return image
        # return 1

    def __getitem__(self, index):
        # print(index)
        image, label = self.get_image(index), self.get_mask(index)
        return image, label

    def __len__(self):
        return len(self.all_training_images)