import os
import shutil
from collections import defaultdict
from nrrd import reader
import glob

import numpy as np
import nrrd
from monai.transforms import Compose, LoadImage, RandFlip, RandAffine, AddChannel, SqueezeDim, MaskIntensity, AsDiscrete, RandGaussianNoise
from monai.data.nifti_writer import write_nifti
import torch
import random
from nrrd_reader import NrrdReader

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
    directories['results'] = os.path.abspath(os.path.join(src_dir, '..', 'out', 'results'))
    return directories

def replace_suffix(string, old_suffix, new_suffix):
    if string.endswith(old_suffix):
        new_string = string[:-len(old_suffix)]
    else:
        raise Exception

    return new_string + new_suffix

def get_data_from_info(path_to_images, path_to_segs, info):
    data = []
    for image_name, _label in info:
        seg_name = replace_suffix(image_name, '.nii.gz', '.nrrd')
        image = os.path.join(path_to_images, image_name)
        if path_to_segs is not None:
            seg = os.path.join(path_to_segs, seg_name)
        else:
            seg = ""
       
        dual_output = False
        if _label is not None:
            if dual_output:
                label = (1-_label)*np.array([1, 0], dtype=np.float) + _label*np.array([0, 1], dtype=np.float)
            else:
                label = np.array([_label], dtype=np.float)
            data.append({'image': image, 'label': label, '_label': _label, 'seg': seg, 'w': False})
        else:
            data.append({'image': image, 'seg': seg, 'w': False})
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

def large_image_splitter(data, cache_dir, num_splits, only_label_one=False):
    print("Splitting large images...")
    len_old = len(data)
    print("original data len:", len_old)
    split_images_dir = os.path.join(cache_dir, 'split_images')
    split_images = os.path.join(split_images_dir, 'split_images.npy')

    def _replace_in_data(split_images, num_splits):
        new_images = []
        for image in data:
            new_images.append(image)
            for s in split_images:
                source_image = s['source']
                if image['_label'] == 0 and only_label_one is True:
                    break
                if image['image'] == source_image:
                    #new_images.pop()
                    for i in range(min(num_splits, len(s["splits"]))):
                        new_images.append(s["splits"][i])
                    break
        return new_images

    if os.path.exists(split_images):
        new_images = np.load(split_images, allow_pickle=True)
        """for s in new_images:
            print("split image:", s["source"], end='\r')"""
        out_data = _replace_in_data(new_images, num_splits)
    else:
        if not os.path.exists(split_images_dir):
            os.mkdir(split_images_dir)
        new_images = []
        imageLoader = LoadImage()
        for image in data:
            image_data, _ = imageLoader(image["image"])
            seg_data, _ = nrrd.read(image['seg'])
            label = image['_label']
            z_len = image_data.shape[2]
            if z_len > 200:
                count = z_len // 80
                print("splitting image:", image["image"], f"into {count} parts", "shape:", image_data.shape, end='\r')
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
                    
                    rand_affine = RandAffine(
                        prob=1.0,
                        rotate_range=(0, 0, np.pi/16),
                        shear_range=(0.07, 0.07, 0.0),
                        translate_range=(0, 0, 0),
                        scale_range=(0.07, 0.07, 0.0),
                        padding_mode="zeros"
                    )
                    transform = Compose([
                        AddChannel(),
                        rand_affine,
                        SqueezeDim(),
                    ])
                    rand_seed = np.random.randint(1e8)
                    transform.set_random_state(seed=rand_seed)
                    split_image = transform(split_image).detach().cpu().numpy()
                    transform.set_random_state(seed=rand_seed)
                    split_seg = transform(split_seg).detach().cpu().numpy()
                    
                    write_nifti(split_image, image_file, resample=False)
                    nrrd.write(seg_file, split_seg)
                    new_image['splits'].append({'image': image_file, 'label': image['label'], '_label': image['_label'], 'seg': seg_file, 'w': False})
                new_images.append(new_image)
        np.save(split_images, new_images)
        out_data = _replace_in_data(new_images, num_splits)

    print("new data len:", len(out_data))
    return out_data

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

def calculate_class_imbalance(data):
    negative, positive = 0, 0
    for d in data:
        label = d['_label']
        if int(label) == 0:
            negative += 1
        elif int(label) == 1:
            positive += 1
    pos_weight = (negative/positive)

    return pos_weight, negative, positive

def balance_training_data(train_info, seed=None):
    random.seed(seed)

    file_list = [x for x in train_info if int(x['_label'])==1]
    for i in range(len(train_info)-2*len(file_list)):
        train_info.append(file_list[random.randint(0,len(file_list)-1)])

def transform_and_copy(data, cahce_dir):
    copy_dir = os.path.join(cahce_dir, 'copied_images')
    if not os.path.exists(copy_dir):
        os.mkdir(copy_dir)
    copy_list_path = os.path.join(copy_dir, 'copied_images.npy')
    if not os.path.exists(copy_list_path):
        print("transforming and copying images...")
        imageLoader = LoadImage()
        to_copy_list = [x for x in data if int(x['_label'])==1]
        mul = 1#int(len(data)/len(to_copy_list) - 1)

        rand_x_flip = RandFlip(spatial_axis=0, prob=0.50)
        rand_y_flip = RandFlip(spatial_axis=1, prob=0.50)
        rand_z_flip = RandFlip(spatial_axis=2, prob=0.50)
        rand_affine = RandAffine(
            prob=1.0,
            rotate_range=(0, 0, np.pi/10),
            shear_range=(0.12, 0.12, 0.0),
            translate_range=(0, 0, 0),
            scale_range=(0.12, 0.12, 0.0),
            padding_mode="zeros"
        )
        rand_gaussian_noise = RandGaussianNoise(prob=0.5, mean=0.0, std=0.05)
        transform = Compose([
            AddChannel(),
            rand_x_flip,
            rand_y_flip,
            rand_z_flip,
            rand_affine,
            SqueezeDim(),
        ])
        copy_list = []
        n = len(to_copy_list)
        for i in range(len(to_copy_list)):
            print(f'Copying image {i+1}/{n}', end = "\r")
            to_copy = to_copy_list[i]
            image_file = to_copy['image']
            _image_file = replace_suffix(image_file, '.nii.gz', '')
            label = to_copy['label']
            _label = to_copy['_label']
            image_data, _ = imageLoader(image_file)
            seg_file = to_copy['seg']
            seg_data, _ = nrrd.read(seg_file)

            for i in range(mul):
                rand_seed = np.random.randint(1e8)
                transform.set_random_state(seed=rand_seed)
                new_image_data = rand_gaussian_noise(np.array(transform(image_data)))
                transform.set_random_state(seed=rand_seed)
                new_seg_data = np.array(transform(seg_data))
                #multi_slice_viewer(image_data, image_file)
                #multi_slice_viewer(seg_data, seg_file)
                #seg_image = MaskIntensity(seg_data)(image_data)
                #multi_slice_viewer(seg_image, seg_file)
                image_basename = os.path.basename(_image_file)
                seg_basename = image_basename + f'_seg_{i}.nrrd'
                image_basename = image_basename + f'_{i}.nii.gz'
                
                new_image_file = os.path.join(copy_dir, image_basename)
                write_nifti(new_image_data, new_image_file, resample=False)
                new_seg_file = os.path.join(copy_dir, seg_basename)
                nrrd.write(new_seg_file, new_seg_data)
                copy_list.append({'image': new_image_file, 'seg': new_seg_file, 'label': label, '_label': _label})

        np.save(copy_list_path, copy_list)
        print("done transforming and copying!")

    copy_list = np.load(copy_list_path, allow_pickle=True)
    return copy_list

def balance_training_data2(data, copies, ratio=1, seed=None):
    random.seed(seed)

    random.shuffle(copies)
    _, n, p = calculate_class_imbalance(data)
    data.extend(copies[:n*ratio-p])
    
def load_mixed_images(data_dir):
    
    mixed_images_dir = os.path.join(data_dir, 'mixed_images')
    mixed_images = glob.glob(os.path.join(mixed_images_dir, '*'))
    
    data = []
    for image in mixed_images:
        data.append({'image': image, 'label': np.array([1], dtype=np.float), '_label': 1, 'seg': "", 'w': True})

    return data

def convert_labels(data, dtype=np.float32, as_array=True, n_classes=2):
    for d in data:
        if not as_array:
            d['label'] = d['label'].astype(dtype).item()
        else:
            d['label'] = d['label'].astype(dtype)
            