import os

import torch
import numpy as np
import nrrd

import matplotlib.pyplot as plt

def replace_suffix(string, old_suffix, new_suffix):
    if string.endswith(old_suffix):
        new_string = string[:-len(old_suffix)]
    else:
        raise Exception

    return new_string + new_suffix

def get_data_from_info(path_to_images, path_to_segs, info):
    data = []
    for image_name, label in info:
        if not label: label = -1
        #seg_name = image_name.removesuffix('.nii.gz') + '.nrrd'
        seg_name = replace_suffix(image_name, '.nii.gz', '.nrrd')
        image = os.path.join(path_to_images, image_name)
        seg = os.path.join(path_to_segs, seg_name)

        data.append({'image': image, 'label': np.float32([int(label)]), 'seg': seg, 'seg_transforms': []})
    return data

def compute_acc(x, y):
    l = len(x)
    s = torch.eq(x, y).sum().item()

    return s / l

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[-1] // 2
    ax.imshow(volume[:, :, ax.index], cmap='gray')
    fig.canvas.mpl_connect('key_press_event', process_key)

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
