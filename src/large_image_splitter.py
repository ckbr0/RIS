import os
from monai.data import nifti_writer

import numpy as np
from monai.transforms import LoadImage, SaveImage
import nrrd
from monai.data.nifti_writer import write_nifti
from utils import replace_suffix, multi_slice_viewer

def large_image_splitter(data, cache_dir):
    print("Splitting large images...")
    len_old = len(data)
    print("original data len:", len_old)
    split_images_dir = os.path.join(cache_dir, 'split_images')
    split_images = os.path.join(split_images_dir, 'split_images.npy')

    def _replace_in_data(images):
        for image in images:
            source_image = image['source']
            for i in range(len(data)):
                if data[i]["image"] == source_image:
                    del data[i]
                    break
            data.extend(image["splits"])

    if os.path.exists(split_images):
        split_images = np.load(split_images, allow_pickle=True)
        _replace_in_data(split_images)
    else:
        if not os.path.exists(split_images_dir):
            os.mkdir(split_images_dir)
        new_images = []
        imageLoader = LoadImage()
        for image in data:
            image_data, _ = imageLoader(image["image"])
            seg_data, _ = nrrd.read(image['seg'])
            z_len = image_data.shape[2]
            if z_len > 200:
                count = z_len // 80
                print("splitting image:", image["image"], f"into {count} parts", "shape:", image_data.shape)
                split_image_list = [image_data[:, :, idz::count] for idz in range(count)]
                split_seg_list = [seg_data[:, :, idz::count] for idz in range(count)]
                new_image = { 'source': image["image"], 'splits': [] }
                for i in range(count):
                    image_file = os.path.basename(replace_suffix(image["image"], '.nii.gz', ''))
                    image_file = os.path.join(split_images_dir, image_file + f'_{i}.nii.gz')
                    seg_file = os.path.basename(replace_suffix(image["seg"], '.nrrd', ''))
                    seg_file = os.path.join(split_images_dir, seg_file + f'_seg_{i}.nrrd')
                    split_image = np.array(split_image_list[i])
                    split_seg = np.array(split_seg_list[i], dtype=np.uint8)
                    #print("saving split image into: ", image_file, "and", seg_file)
                    write_nifti(split_image, image_file, resample=False)
                    #np.save(image_file, split_image)
                    nrrd.write(seg_file, split_seg)
                    #np.save(seg_file, split_seg)
                    new_image['splits'].append({ 'image': image_file, 'label': image['label'], 'seg': seg_file })
                new_images.append(new_image)
        np.save(split_images, new_images)
        _replace_in_data(new_images)

    print("new data len:", len(data))
                     
