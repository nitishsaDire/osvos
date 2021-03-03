import pandas as pd

def get_online_training_data():
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
        objs[0]
        dict_objs_imt = {}
        dict_objs_mat = {}


        for obj in objs:
          dict_objs_imt[obj] = []
          dict_objs_mat[obj] = []

        [dict_objs_imt[obj].append(i) for i in all_training_images for obj in objs  if obj in i]
        [dict_objs_mat[obj].append(i) for i in all_training_masks  for obj in objs  if obj in i]

        dict_objs_imv = {}
        dict_objs_mav = {}

        objs = list(set([i.split('/')[3] for i in all_val_images]))

        for obj in objs:
          dict_objs_imv[obj] = []
          dict_objs_mav[obj] = []

        [dict_objs_imv[obj].append(i) for i in all_val_images for obj in objs  if obj in i]
        [dict_objs_mav[obj].append(i) for i in all_val_masks  for obj in objs  if obj in i]

        return dict_objs_imt, dict_objs_mat, dict_objs_imv, dict_objs_mav
