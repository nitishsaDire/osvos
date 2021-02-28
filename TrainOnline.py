from Model import RATINANET_vgg19
from Dataset import ImageDataset_DAVIS
from OnlineData import get_online_training_data
from train import train_model_online
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import os

use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
print(device)

_,_,dict_objs_imv, dict_objs_mav = get_online_training_data()

for folder in dict_objs_imv.keys():
# for folder in ['car-roundabout', 'horsejump-high', 'camel', 'blackswan']:
# for folder in ['bmx-trees']:
    # folder = 'bmx-trees'
    print(folder)

    train_dataset = ImageDataset_DAVIS([dict_objs_imv[folder][0]], [dict_objs_mav[folder][0]])

    dataset_size = len(train_dataset)

    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)


    cnn = RATINANET_vgg19().to(device)


    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.0001, weight_decay=0.0001)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model = train_model_online(cnn, optimizer, exp_lr_scheduler, dataloader, dataset_size, device, loadModel=True, num_epochs=1000)

    test_dataset = ImageDataset_DAVIS(dict_objs_imv[folder], dict_objs_mav[folder])

    dataset_size = len(test_dataset)

    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=1)

    model.eval()
    im_count = 0
    for a in dataloader:
      # print(a[0].shape)
      x = a[0].cuda()
      output = model(x)
      output = output[-1]
      output = torch.sigmoid(output)
      output = torch.ge(output, 0.5).float()
      for im,o in zip(x, output):
          # plt.imshow(denormalize(im.cpu().permute(1,2,0)))
          # plt.imshow(o.squeeze().cpu().detach().numpy(), alpha = 0.5)
          # plt.show()

          im = im.cpu()

          # print(im.shape)
          # print(o.max(), o.min())
          mask = torch.cat((torch.zeros_like(o),o,torch.zeros_like(o)), dim=0)
          # plt.imshow(mask.permute(1,2,0).cpu())
          # plt.show()
          # print(mask.shape)
          imn = (im - im.min())/(im.max() - im.min())
          idx = (mask==1.)
          # print(im[idx].shape, idx.shape)
          imn[idx] = mask[idx].cpu()

          # plt.imshow(denormalize(im.permute(1,2,0)))
          # print(denormalize(im.permute(1,2,0)).shape)
          pilim = transforms.ToPILImage(mode='RGB')(imn)
          pilim.save("/content/masks_resuls/" + folder+str(im_count)+".jpg")
          # print(pilim.size)
          # plt.imshow(pilim)
          im_count +=1
          # plt.show()
      #     break
      # break
    os.system('rm {}.mp4'.format(folder))
    os.system("ffmpeg -framerate 25 -i /content/masks_resuls/{}%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {}.mp4".format(folder,folder))