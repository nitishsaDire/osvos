# osvos
This is my implementation of the OSVOS model for video object segmentation on the DAVIS-16 dataset.

The task is to segment the object in the entire video as per the annotations of the first frame of the video. For example, if in the first frame a dog is segmented, then we need to segment that dog in all the following frames. 
The challenge is to detect the object in the consecutive frames and segment it. The object might undergo deformation, occlusion, etc or might have shape complexities, edge ambiguities etc.
So, the main difﬁculty is to effectively handle appearance changes and similar back- ground objects, while maintaining accurate segmentation.


The training is divided into 2 parts, training parent network on a small subset (~20%) of DAVIS.
Then the trained network is fine-tuned on the 1st frame of the video to be segmented.
The more time it is fine-tuned for a specific video, the better the segmentation results are. 

# Steps
#### 1, Download the data.
```Shell
! wget https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
! unzip DAVIS-data.zip
! cat DAVIS/ImageSets/480p/train.txt | awk '{print "DAVIS"$1}' > training_images
! cat DAVIS/ImageSets/480p/train.txt | awk '{print "DAVIS"$2}' > training_masks
! cat DAVIS/ImageSets/480p/val.txt | awk '{print "DAVIS"$1}' > val_images
! cat DAVIS/ImageSets/480p/val.txt | awk '{print "DAVIS"$2}' > val_masks
```

#### 2, Train the parent network.
```Shell
! python TrainParent.py
```
#### 3, Train the network for a particular video online.
```Shell
! python TrainOnline.py
```


# Results
https://drive.google.com/drive/folders/1SrRkOkyEOpGch45GO_nIilZCvrGiORfV?usp=sharing

These are the segmentation results on the videos in the val dataset.
I trained the parent network for 30 epochs, and did online training for 1000 epochs (because it's fast).

I believe if the model could be trained for lot more epochs, ~2k, then the results would have been much better, I couldn't do so due to GPU constraints.

#### Average (IOU, Contour accuracy) for val dataset:
```
horsejump-high               0.661, 0.700
paragliding-launch           0.539, 0.547
dog                          0.718, 0.505
drift-chicane                0.594, 0.562
scooter-black                0.563, 0.446
bmx-trees                    0.365, 0.631
car-shadow                   0.825, 0.691
breakdance                   0.465, 0.461
blackswan                    0.875, 0.793
kite-surf                    0.554, 0.696
dance-twirl                  0.446, 0.490
car-roundabout               0.753, 0.541
soapbox                      0.386, 0.520
libby                        0.545, 0.512
goat                         0.813, 0.761
camel                        0.798, 0.725
drift-straight               0.637, 0.498
parkour                      0.634, 0.501
motocross-jump               0.414, 0.355
cows                         0.908, 0.845
-----------------------------------------
AVERAGE                      0.625, 0.589  

``` 


###### PS: The code is not written in the best possible manner, and might have bugs, please raise an issue if have difficulty. Thanks.