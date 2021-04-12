import os
from collections import defaultdict

import numpy as np
import nrrd
from monai.transforms import LoadImage, SaveImage
from monai.data.nifti_writer import write_nifti
import torch

import matplotlib.pyplot as plt

def setup_directories():
    directories = defaultdict(str)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    directories['src'] = src_dir
    directories['data'] = os.path.abspath(os.path.join(src_dir, '..', 'data'))
    directories['out'] = os.path.abspath(os.path.join(src_dir, '..', 'out'))
    directories['cache'] = os.path.abspath(os.path.join(src_dir, '..', 'cache'))
    directories['persistent'] = os.path.abspath(os.path.join(src_dir, '..', 'cache' ,'persistent'))
    directories['split_images'] = os.path.abspath(os.path.join(src_dir, '..', 'cache', 'split_images'))
    return directories

def replace_suffix(string, old_suffix, new_suffix):
    if string.endswith(old_suffix):
        new_string = string[:-len(old_suffix)]
    else:
        raise Exception

    return new_string + new_suffix

def get_data_from_info(path_to_images, path_to_segs, info, dual_output=False):
    data = []
    for image_name, label in info:
        if label is None: label = -1
        seg_name = replace_suffix(image_name, '.nii.gz', '.nrrd')
        image = os.path.join(path_to_images, image_name)
        if path_to_segs is not None:
            seg = os.path.join(path_to_segs, seg_name)
        else:
            seg = ""
        
        if dual_output:
            label = (1-label)*np.array([1.0, 0.0], dtype=np.float32) + label*np.array([0.0, 1.0], dtype=np.float32)
        else:
            label = np.array([float(label)], dtype=np.float32)
        data.append({'image': image, 'label': label, 'seg': seg})
    return data

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume, image_name=None):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    title_text = f"Shape: {volume.shape}"
    if image_name:
        title_text = image_name + " " + title_text
    ax.set_title(title_text)
    ax.volume = volume
    ax.index = volume.shape[-1] // 2
    ax.imshow(volume[:, :, ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[-1]  # wrap around using %
    ax.images[0].set_array(volume[ :, :, ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[-1]
    ax.images[0].set_array(volume[ :, :, ax.index])

def large_image_splitter(data, cache_dir, dual_output=False):
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
                    source_label = data[i]["label"]
                    del data[i]
                    break
            splits = image["splits"]
            for s in splits:
                s['label'] = source_label
                data.append(s)
            #data.extend(image["splits"])

    if os.path.exists(split_images):
        split_images = np.load(split_images, allow_pickle=True)
        for s in split_images:
            print("split image:", s["source"])
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
                    write_nifti(split_image, image_file, resample=False)
                    nrrd.write(seg_file, split_seg)
                    new_image['splits'].append({ 'image': image_file, 'label': image['label'], 'seg': seg_file })
                new_images.append(new_image)
        np.save(split_images, new_images)
        _replace_in_data(new_images)

    print("new data len:", len(data))

def calculate_class_imbalance(train_info):
    negative, positive = 0, 0
    for _, label in train_info:
        if int(label) == 0:
            negative += 1
        elif int(label) == 1:
            positive += 1
    pos_weight = torch.Tensor([(negative/positive)])

    return pos_weight

def create_device(device_name):
    gpu = False
    if "cuda" in device_name:
        if torch.cuda.is_available():
            gpu = True
        else:
            print("Cuda device is not supported, switching to CPU")
            device_name = "cpu"

    device = torch.device(device_name)
    return device, gpu
