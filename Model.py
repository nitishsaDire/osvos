import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

# created by Nitish Sandhu
# date 17/feb/2021

class RATINANET_vgg19(nn.Module):
    def __init__(self):
        super().__init__()
        use_cuda = torch.cuda.is_available()  # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

        resnet = models.vgg19_bn(pretrained=True)
        self.resnet_layers = list(resnet.children())
        self.size = (480, 854)
        self.centre_crop = transforms.CenterCrop(self.size)

        # self.pre1 = self.conv2d(3, 64, 3, 1, 1)
        # self.pre2 = self.conv2d(64, 64, 3, 1, 1)
        # self.seq1 = nn.Sequential(*self.resnet_layers[0:3])  # 112x112, 64
        # self.seq2 = nn.Sequential(*self.resnet_layers[3:5])  # 56x56, 64
        # self.seq3 = nn.Sequential(*self.resnet_layers[5])    # 28x28, 128
        # self.seq4 = nn.Sequential(*self.resnet_layers[6])    # 14x14, 256
        # self.seq5 = nn.Sequential(*self.resnet_layers[7])    # 7x7, 512

        self.seq1 = self.resnet_layers[0][0:6]
        self.seq2 = self.resnet_layers[0][6:13]
        self.seq3 = self.resnet_layers[0][13:26]
        self.seq4 = self.resnet_layers[0][26:39]
        self.seq5 = self.resnet_layers[0][39:52]

        self.side_prep_p2 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.side_prep1 = nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side_prep2 = nn.Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side_prep3 = nn.Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side_prep4 = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.side_prep5 = nn.Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.score_dsn1 = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.score_dsn2 = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.score_dsn3 = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.score_dsn4 = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
        self.score_dsn5 = nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))

        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.up2 = nn.ConvTranspose2d(16, 16, kernel_size=(8, 8), stride=(4, 4), bias=False)
        self.up3 = nn.ConvTranspose2d(16, 16, kernel_size=(16, 16), stride=(8, 8), bias=False)
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=(32, 32), stride=(16, 16), bias=False)
        # self.up5 = nn.ConvTranspose2d(16, 16, kernel_size=(64, 64), stride=(32, 32), bias=False)

        self.up1_ = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride=(2, 2), bias=False)
        self.up2_ = nn.ConvTranspose2d(1, 1, kernel_size=(8, 8), stride=(4, 4), bias=False)
        self.up3_ = nn.ConvTranspose2d(1, 1, kernel_size=(16, 16), stride=(8, 8), bias=False)
        self.up4_ = nn.ConvTranspose2d(1, 1, kernel_size=(32, 32), stride=(16, 16), bias=False)
        # self.up5_ = nn.ConvTranspose2d(1, 1, kernel_size=(64, 64), stride=(32, 32), bias=False)

        self.fuse = nn.Conv2d(80, 1, kernel_size=(1, 1), stride=(1, 1), padding=0)

        # self.intialize_p()

    def conv2d(self, in_, out, k, s, p):
        return nn.Sequential([
            nn.Conv2d(in_, out, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out),
            nn.ReLU()
        ])

    def intialize_p(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param x: shape (bs, 3, 224, 224)
        :return:
        """
        # p1 = self.pre1(x)
        # p2 = self.pre2(p1)
        with torch.no_grad():
            x1e = self.seq1(x)  # /1,  64
            x2e = self.seq2(x1e)  # /2,  128
            x3e = self.seq3(x2e)  # /4,  256
            x4e = self.seq4(x3e)  # /8,  512
            x5e = self.seq5(x4e)  # /16, 512

        outs = []

        side_prep1 = self.side_prep1(x1e)
        side_prep2 = self.side_prep2(x2e)
        side_prep3 = self.side_prep3(x3e)
        side_prep4 = self.side_prep4(x4e)
        side_prep5 = self.side_prep5(x5e)

        outs.append(self.centre_crop(side_prep1))
        outs.append(self.centre_crop(self.up1(side_prep2)))
        outs.append(self.centre_crop(self.up2(side_prep3)))
        outs.append(self.centre_crop(self.up3(side_prep4)))
        outs.append(self.centre_crop(self.up4(side_prep5)))

        side_out = []

        side_out.append(self.centre_crop(self.score_dsn1(side_prep1)))
        side_out.append(self.centre_crop(self.up1_(self.score_dsn2(side_prep2))))
        side_out.append(self.centre_crop(self.up2_(self.score_dsn3(side_prep3))))
        side_out.append(self.centre_crop(self.up3_(self.score_dsn4(side_prep4))))
        side_out.append(self.centre_crop(self.up4_(self.score_dsn5(side_prep5))))

        cat_out = torch.cat(outs[:], dim=1)
        side_out.append(self.fuse(cat_out))
        return side_out