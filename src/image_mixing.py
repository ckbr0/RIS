import os
import time
import itertools
from hashlib import blake2b
import random

import numpy as np
from utils import setup_directories, get_data_from_info, large_image_splitter, multi_slice_viewer
from monai.transforms import Compose, AddChanneld, Resized, LoadImaged, CropForegroundd, GaussianSmooth, RandAffined, SqueezeDimd, Affined
from monai.data.nifti_writer import write_nifti
from transforms import CTWindowd, CTSegmentation

def image_mixing(data, seed=None):
    #random.seed(seed)
    
    file_list = [x for x in data if int(x['_label'])==1]
    random.shuffle(file_list)


    crop_foreground = CropForegroundd(
        keys=["image"],
        source_key="image",
        margin=(0, 0, 0),
        select_fn = lambda x: x != 0
    )
    WW, WL = 1500, -600
    ct_window = CTWindowd(keys=["image"], width=WW, level=WL)
    resize2 = Resized(keys=["image"], spatial_size=(int(512*0.75), int(512*0.75), -1), mode="area")
    resize1 = Resized(keys=["image"], spatial_size=(-1, -1, 40), mode="nearest")
    gauss = GaussianSmooth(sigma=(1., 1., 0))
    gauss2 = GaussianSmooth(sigma=(2.0, 2.0, 0))
    affine = Affined(keys=["image"], scale_params=(1.0, 2.0, 1.0), padding_mode='zeros')
    
    common_transform = Compose([
        LoadImaged(keys=["image"]),
        ct_window,
        CTSegmentation(keys=["image"]),
        AddChanneld(keys=["image"]),
        affine,
        crop_foreground,
        resize1,
        resize2,
        SqueezeDimd(keys=["image"]),
    ])

    dirs = setup_directories()
    data_dir = dirs['data']
    mixed_images_dir = os.path.join(data_dir, 'mixed_images')
    if not os.path.exists(mixed_images_dir):
        os.mkdir(mixed_images_dir)

    for img1, img2 in itertools.combinations(file_list, 2):

        img1 = {'image': img1["image"], 'seg': img1['seg']}
        img2 = {'image': img2["image"], 'seg': img2['seg']}
        
        img1_data = common_transform(img1)["image"]
        img2_data = common_transform(img2)["image"]
        img1_mask, img2_mask = (img1_data > 0), (img2_data > 0)
        img_presek = np.logical_and(img1_mask, img2_mask)
        img = np.maximum(img_presek*img1_data, img_presek*img2_data)

        multi_slice_viewer(img, "img1")
        
        loop = True
        while loop:
            save = input("Save image [y/n/e]: ")
            if save.lower() == 'y':
                loop = False
                k = str(time.time()).encode('utf-8')
                h = blake2b(key=k, digest_size=16)
                name = h.hexdigest() + '.nii.gz'
                out_path = os.path.join(mixed_images_dir, name)
                write_nifti(img, out_path, resample=False)
            elif save.lower() == 'n':
                loop = False
                break
            elif save.lower() == 'e':
                print("exeting")
                exit()
            else:
                print("wrong input!")

def main():
    dirs = setup_directories()
    
    hackathon_dir = os.path.join(dirs["data"], 'HACKATHON')
    map_fn = lambda x: (x[0], int(x[1]))
    with open(os.path.join(hackathon_dir, "train.txt"), 'r') as fp:
        train_info_hackathon = [map_fn(entry.strip().split(',')) for entry in fp.readlines()]
    image_dir = os.path.join(hackathon_dir, 'images', 'train')
    seg_dir = os.path.join(hackathon_dir, 'segmentations', 'train')
    _train_data_hackathon = get_data_from_info(
        image_dir, seg_dir, train_info_hackathon, dual_output=False
    )
    _train_data_hackathon = large_image_splitter(_train_data_hackathon, dirs["cache"])

    image_mixing(_train_data_hackathon)
