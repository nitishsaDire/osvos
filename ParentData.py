import pandas as pd
import random

# I have run on randomly 20% of the frames for a video.
def get_parent_training_data():
        training_images = pd.read_csv('training_images', header=None)
        training_images.columns=['file']

        training_masks = pd.read_csv('training_masks', header=None)
        training_masks.columns=['file']

        val_images = pd.read_csv('val_images', header=None)
        val_images.columns=['file']

        val_masks = pd.read_csv('val_masks', header=None)
        val_masks.columns=['file']

        all_training_images, all_training_masks, all_val_images, all_val_masks = list(training_images['file']), list(training_masks['file']),list(val_images['file']), list(val_masks['file'])

        # extracted_masks = [i.replace("JPEGImages", "Annotations").replace("jpg", "png") for i in all_training_images]
        # str(all_training_masks) == str(extracted_masks)


        objs = list(set([i.split('/')[3] for i in all_training_images]))
        dict_objs_im = {}
        dict_objs_ma = {}


        for obj in objs:
          dict_objs_im[obj] = []
          dict_objs_ma[obj] = []

        [dict_objs_im[obj].append(i) for i in all_training_images for obj in objs  if obj in i]
        [dict_objs_ma[obj].append(i) for i in all_training_masks  for obj in objs  if obj in i]

        total = 0
        selected_training_images, selected_training_masks = [],[]
        for k in dict_objs_im.keys():
          sample = random.sample(range(0,len(dict_objs_im[k])), (len(dict_objs_im[k])*20)//100)
          sample.sort()
          # print(sample, len(sample))
          total += len(sample)
          for s in sample:
            selected_training_images.append(dict_objs_im[k][s])
            selected_training_masks.append( dict_objs_ma[k][s])

        objs = list(set([i.split('/')[3] for i in all_val_images]))
        objs[0]
        dict_objs_im = {}
        dict_objs_ma = {}


        for obj in objs:
          dict_objs_im[obj] = []
          dict_objs_ma[obj] = []

        [dict_objs_im[obj].append(i) for i in all_val_images for obj in objs  if obj in i]
        [dict_objs_ma[obj].append(i) for i in all_val_masks  for obj in objs  if obj in i]

        total = 0
        selected_val_images, selected_val_masks = [],[]
        for k in dict_objs_im.keys():
          sample = random.sample(range(0,len(dict_objs_im[k])), (len(dict_objs_im[k])*20)//100)
          sample.sort()
          # print(sample, len(sample))
          total += len(sample)
          for s in sample:
            selected_val_images.append(dict_objs_im[k][s])
            selected_val_masks.append( dict_objs_ma[k][s])

        return selected_training_images, selected_training_masks, selected_val_images, selected_val_masks

