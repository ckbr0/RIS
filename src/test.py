import sys
import os
import logging
from glob import glob
import numpy as np
import torch
import monai.networks.nets as nets
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    ToTensord,
    RandAxisFlipd,
    RandAffined,
    SpatialPadd,
    Activationsd,
    Resized,
    RandGaussianNoised,
)
from transforms import (
    CTWindowd,
    #RandCTWindowd,
    CTSegmentation,
    RelativeCropZd,
)
from monai.data import DataLoader, Dataset, PersistentDataset, CacheDataset
from monai.transforms.croppad.batch import PadListDataCollate
from monai.utils import NumpyPadMode, set_determinism
from monai.utils.enums import Method
from monai.config import print_config
from sklearn.model_selection import train_test_split
from trainer import Trainer
from validator import Validator
from tester import Tester
from utils import multi_slice_viewer, setup_directories, get_data_from_info, large_image_splitter, calculate_class_imbalance, create_device, balance_training_data
from test_data_loader import TestDataset

def main(train_output):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Setup directories
    dirs = setup_directories()

    # Setup torch device
    device, using_gpu = create_device("cuda")

    # Load and randomize images

    # HACKATON image and segmentation data
    hackathon_dir = os.path.join(dirs["data"], 'HACKATHON')
    map_fn = lambda x: (x[0], int(x[1]))
    with open(os.path.join(hackathon_dir, "train.txt"), 'r') as fp:
        train_info_hackathon = [map_fn(entry.strip().split(',')) for entry in fp.readlines()]
    image_dir = os.path.join(hackathon_dir, 'images', 'train')
    seg_dir = os.path.join(hackathon_dir, 'segmentations', 'train')
    _train_data_hackathon = get_data_from_info(
        image_dir, seg_dir, train_info_hackathon, dual_output=False
    )
    large_image_splitter(_train_data_hackathon, dirs["cache"])

    balance_training_data(_train_data_hackathon, seed=72)

    # PSUF data
    """psuf_dir = os.path.join(dirs["data"], 'psuf')
    with open(os.path.join(psuf_dir, "train.txt"), 'r') as fp:
        train_info = [entry.strip().split(',') for entry in fp.readlines()]
    image_dir = os.path.join(psuf_dir, 'images')
    train_data_psuf = get_data_from_info(image_dir, None, train_info)"""
    # Split data into train, validate and test
    train_split, test_data_hackathon = train_test_split(_train_data_hackathon, test_size=0.2, shuffle=True, random_state=42)
    #train_data_hackathon, valid_data_hackathon = train_test_split(train_split, test_size=0.2, shuffle=True, random_state=43)
    # Setup transforms

    # Crop foreground
    crop_foreground = CropForegroundd(
        keys=["image"],
        source_key="image",
        margin=(5, 5, 0),
        #select_fn = lambda x: x != 0
    )
    # Crop Z
    crop_z = RelativeCropZd(keys=["image"], relative_z_roi=(0.07, 0.12))
    # Window width and level (window center)
    WW, WL = 1500, -600
    ct_window = CTWindowd(keys=["image"], width=WW, level=WL)
    spatial_pad = SpatialPadd(keys=["image"], spatial_size=(-1, -1, 30))
    resize = Resized(keys=["image"], spatial_size=(int(512*0.50), int(512*0.50), -1), mode="trilinear")
    
    # Create transforms
    common_transform = Compose([
        LoadImaged(keys=["image"]),
        ct_window,
        CTSegmentation(keys=["image"]),
        AddChanneld(keys=["image"]),
        resize,
        crop_foreground,
        crop_z,
        spatial_pad,
    ])
    hackathon_train_transfrom = Compose([
        common_transform,
        ToTensord(keys=["image"]),
    ]).flatten()
    psuf_transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    # Setup data
    #set_determinism(seed=100)
    test_dataset = PersistentDataset(data=test_data_hackathon[:], transform=hackathon_train_transfrom, cache_dir=dirs["persistent"])
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=using_gpu,
        num_workers=1,
        collate_fn=PadListDataCollate(Method.SYMMETRIC, NumpyPadMode.CONSTANT)
    )

    # Setup network, loss function, optimizer and scheduler
    network = nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1).to(device)

    # Setup validator and trainer
    valid_post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
    ])

    # Setup tester
    tester = Tester(
        device=device,
        test_data_loader=test_loader,
        load_dir=train_output,
        out_dir=dirs["out"],
        network=network,
        post_transform=valid_post_transforms,
        non_blocking=using_gpu,
        amp=using_gpu
    )

    # Run tester
    tester.run()

if __name__ == "__main__":
    main(sys.argv[1])
