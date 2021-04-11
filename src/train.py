import sys
import os
import logging
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
)
from transforms import (
    CTWindowd,
    RandCTWindowd,
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
from utils import multi_slice_viewer, setup_directories, get_data_from_info, large_image_splitter, calculate_class_imbalance, create_device
from test_data_loader import TestDataset

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print_config()

    # Setup directories
    dirs = setup_directories()

    # Setup torch device
    device, using_gpu = create_device("cuda")

    # Load and randomize images

    # HACKATON image and segmentation data
    hackathon_dir = os.path.join(dirs["data"], 'HACKATHON')
    with open(os.path.join(hackathon_dir, "train.txt"), 'r') as fp:
        train_info_hackathon = [entry.strip().split(',') for entry in fp.readlines()]
    image_dir = os.path.join(hackathon_dir, 'images', 'train')
    seg_dir = os.path.join(hackathon_dir, 'segmentations', 'train')
    _train_data_hackathon = get_data_from_info(image_dir, seg_dir, train_info_hackathon)
    # PSUF data
    """psuf_dir = os.path.join(dirs["data"], 'psuf')
    with open(os.path.join(psuf_dir, "train.txt"), 'r') as fp:
        train_info = [entry.strip().split(',') for entry in fp.readlines()]
    image_dir = os.path.join(psuf_dir, 'images')
    train_data_psuf = get_data_from_info(image_dir, None, train_info)"""
    # Split data into train, validate and test
    train_split, test_data_hackathon = train_test_split(_train_data_hackathon, test_size=0.2, shuffle=True, random_state=42)
    train_data_hackathon, valid_data_hackathon = train_test_split(train_split, test_size=0.2, shuffle=True, random_state=43)
    large_image_splitter(train_data_hackathon, dirs["cache"])

    # Setup transforms

    # Crop foreground
    crop_foreground = CropForegroundd(
        keys=["image"],
        source_key="image",
        margin=(5, 5, 0),
        select_fn = lambda x: x != 0
    )
    # Crop Z
    crop_z = RelativeCropZd(keys=["image"], relative_z_roi=(0.07, 0.12))
    # Window width and level (window center)
    WW, WL = 1500, -600
    ct_window = CTWindowd(keys=["image"], width=WW, level=WL)
    # Random variatnon in CT window
    rand_WW, rand_WL = 50, 25
    rand_ct_window = RandCTWindowd(
        keys=["image"],
        prob=1.0,
        width=(WW-rand_WW, WW+rand_WW),
        level=(rand_WL-25, rand_WL+25)
    )
    # Random axis flip
    rand_axis_flip = RandAxisFlipd(keys=["image"], prob=0.1)
    # Rand affine transform
    rand_affine = RandAffined(
        keys=["image"],
        prob=0.25,
        rotate_range=(0, 0, np.pi/16),
        shear_range=(0.05, 0.05, 0.0),
        translate_range=(0, 0, 0),
        scale_range=(0.05, 0.05, 0.0),
        padding_mode="zeros"
    )
    spatial_pad = SpatialPadd(keys=["image"], spatial_size=(-1, -1, 30))
    resize = Resized(keys=["image"], spatial_size=(int(512*0.50), int(512*0.50), -1), mode="trilinear")
    
    # Create transforms
    common_transform = Compose([
        LoadImaged(keys=["image"]),
        CTSegmentation(keys=["image"]),
        AddChanneld(keys=["image"]),
        resize,
        crop_foreground,
        crop_z,
        spatial_pad,
    ])
    hackathon_train_transform = Compose([
        common_transform,
        rand_ct_window,
        rand_axis_flip,
        rand_affine,
        #crop_foreground,
        ToTensord(keys=["image"]),
    ]).flatten()
    hackathon_valid_transfrom = Compose([
        common_transform,
        ct_window,
        ToTensord(keys=["image"]),
    ]).flatten()
    psuf_transforms = Compose([
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        ToTensord(keys=["image"]),
    ])

    # Setup data
    #set_determinism(seed=100)
    train_dataset = PersistentDataset(data=train_data_hackathon[:], transform=hackathon_train_transform, cache_dir=dirs["persistent"])
    valid_dataset = PersistentDataset(data=valid_data_hackathon[:], transform=hackathon_valid_transfrom, cache_dir=dirs["persistent"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=using_gpu,
        num_workers=2,
        collate_fn=PadListDataCollate(Method.SYMMETRIC, NumpyPadMode.CONSTANT)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=2,
        shuffle=True,
        pin_memory=using_gpu,
        num_workers=2,
        collate_fn=PadListDataCollate(Method.SYMMETRIC, NumpyPadMode.CONSTANT)
    )

    # Setup network, loss function, optimizer and scheduler
    network = nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=1).to(device)
    # pos_weight for class imbalance
    pos_weight = calculate_class_imbalance(train_info_hackathon).to(device)
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.2e-3, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)

    # Setup validator and trainer
    valid_post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
    ])
    validator = Validator(
        device=device,
        val_data_loader=valid_loader,
        network=network,
        post_transform=valid_post_transforms,
        amp=using_gpu,
        non_blocking=using_gpu
    )

    trainer = Trainer(
        device=device,
        out_dir=dirs["out"],
        out_name="DenseNet",
        max_epochs=60,
        train_data_loader=train_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_function,
        lr_scheduler=scheduler,
        validator=validator,
        amp=using_gpu,
        non_blocking=using_gpu
    )

    """x_max, y_max, z_max, size_max = 0, 0, 0, 0
    for data in train_loader:
        image = data["image"]
        shape = image.shape
        x_max = max(x_max, shape[-3])
        y_max = max(y_max, shape[-2])
        z_max = max(z_max, shape[-1])
        size = int(image.nelement()*image.element_size()/1024/1024)
        size_max = max(size_max, size)
        print("shape:", shape, "size:", str(size)+"MB")
        multi_slice_viewer(image[0, 0, :, :, :], "")
    print(x_max, y_max, z_max, str(size_max)+"MB")"""

    # Run trainer
    trainer.run()

if __name__ == "__main__":
    main()
