import sys
import os
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Slider

import torch
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
from ignite.metrics import Accuracy
import monai
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.inferers.inferer import SimpleInferer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    ROCAUC,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    MetricsSaver,
    MetricLogger,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    Dataset,
    PersistentDataset,
    list_data_collate,
)
from monai.transforms import (
    Identityd,
    BoundingRectd,
    AddChanneld,
    AsDiscreted,
    AsDiscrete,
    Activationsd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Rotated,
    RandRotated,
    RandAffined,
    RandFlipd,
    RandAxisFlipd,
    SqueezeDimd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    NormalizeIntensityd,
    ThresholdIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    MaskIntensityd,
    Lambdad,
    ToNumpyd,
)
#from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.croppad.batch import PadListDataCollate
from monai.metrics import compute_roc_auc
from monai.utils import NumpyPadMode, set_determinism
from monai.utils.enums import Method

from model import ModelCT
from utils import get_data_from_info, multi_slice_viewer
from validation_handler import ValidationHandlerCT
from transforms import CTWindowd, RandCTWindowd, CTSegmentation
from nrrd_reader import NrrdReader

class TrainingWorkflow():

    def __init__(
        self,
        data_dir,
        hackathon_dir,
        out_dir,
        cache_dir,
        unique_name,
        num_workers=2,
        cuda=None,
    ):
        self.data_dir = data_dir
        self.hackathon_dir = hackathon_dir
        self.out_dir = out_dir
        self.unique_name = unique_name
        self.image_data_dir = os.path.join(hackathon_dir, 'images', 'train')
        self.seg_data_dir = os.path.join(hackathon_dir, 'segmentations', 'train')
        self.cache_dir = cache_dir
        self.persistent_dataset_dir = os.path.join(cache_dir, 'persistent')
        self.num_workers = num_workers

        # Create torch device
        if not cuda:
            self.pin_memory = False
            self.device = torch.device('cpu')
        else:
            self.pin_memory = True
            self.device = torch.device('cuda')

  
    def transformations(self, H, L):
        lower = L - (H / 2)
        upper = L + (H / 2)

        basic_transforms = Compose(
            [
                # Load image
                LoadImaged(keys=["image"]),

                # Segmentacija
                CTSegmentation(keys=["image"]),
                
                AddChanneld(keys=["image"]),

                # Crop foreground based on seg image.
                CropForegroundd(keys=["image"], source_key="image", margin=(50, 50, 0)),
            ]
        )

        train_transforms = Compose(
            [
                basic_transforms,

                # Normalizacija na CT okno
                # https://radiopaedia.org/articles/windowing-ct
                RandCTWindowd(keys=["image"], prob=1.0, width=(H-100, H+100), level=(L-50, L+50)),

                # Mogoče zanimiva
                RandAxisFlipd(keys=["image"], prob=0.1),

                RandAffined(
                    keys=["image"],
                    prob=0.25,
                    rotate_range=(0, 0, np.pi/8),
                    shear_range=(0.1, 0.1, 0.0),
                    translate_range=(10, 10, 0),
                    scale_range=(0.1, 0.1, 0.0),
                    spatial_size=(-1, -1, -1),
                    padding_mode="zeros"
                ),

                ToTensord(keys=["image"]),
            ]
        ).flatten()

        # NOTE: No random transforms in the validation data
        valid_transforms = Compose(
            [
                basic_transforms,

                # Normalizacija na CT okno
                # https://radiopaedia.org/articles/windowing-ct
                CTWindowd(keys=["image"], width=H, level=L),

                ToTensord(keys=["image"]),
            ]
        ).flatten()
        
        return train_transforms, valid_transforms

    def train(self, train_info, valid_info, hyperparameters, run_data_check=False): 

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        if not run_data_check:
            start_dt = datetime.datetime.now()
            start_dt_string = start_dt.strftime('%d/%m/%Y %H:%M:%S')
            print(f'Training started: {start_dt_string}')

            # 1. Create folders to save the model
            timedate_info= str(datetime.datetime.now()).split(' ')[0] + '_' + str(datetime.datetime.now().strftime("%H:%M:%S")).replace(':', '-')
            path_to_model = os.path.join(self.out_dir, 'trained_models', self.unique_name +  '_' + timedate_info)
            os.mkdir(path_to_model)

        # 2. Load hyperparameters
        learning_rate = hyperparameters['learning_rate']
        weight_decay = hyperparameters['weight_decay']
        total_epoch = hyperparameters['total_epoch']
        multiplicator = hyperparameters['multiplicator']
        batch_size = hyperparameters['batch_size']
        validation_epoch = hyperparameters['validation_epoch']
        validation_interval = hyperparameters['validation_interval']
        H = hyperparameters['H']
        L = hyperparameters['L']

        # 3. Consider class imbalance
        negative, positive = 0, 0
        for _, label in train_info:
            if int(label) == 0:
                negative += 1
            elif int(label) == 1:
                positive += 1
        
        pos_weight = torch.Tensor([(negative/positive)]).to(self.device)

        # 4. Create train and validation loaders, batch_size = 10 for validation loader (10 central slices)

        train_data = get_data_from_info(self.image_data_dir, self.seg_data_dir, train_info)
        valid_data = get_data_from_info(self.image_data_dir, self.seg_data_dir, valid_info)
        
        set_determinism(seed=0)
        train_trans, valid_trans = self.transformations(H, L)
        train_dataset = PersistentDataset(
            data=train_data[:],
            transform=train_trans,
            cache_dir=self.persistent_dataset_dir
        )
        valid_dataset = PersistentDataset(
            data=valid_data[:],
            transform=valid_trans,
            cache_dir=self.persistent_dataset_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=PadListDataCollate(Method.SYMMETRIC, NumpyPadMode.CONSTANT)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=PadListDataCollate(Method.SYMMETRIC, NumpyPadMode.CONSTANT))

        # Perform data checks
        if run_data_check:
            check_data = monai.utils.misc.first(train_loader)
            print(check_data["image"].shape, check_data["label"])
            multi_slice_viewer(check_data["image"][0, 0, :, :, :])
            plot.show()
            multi_slice_viewer(check_data["image"][2, 0, :, :, :])
            plot.show()
            multi_slice_viewer(check_data["image"][4, 0, :, :, :])
            plot.show()
            multi_slice_viewer(check_data["image"][6, 0, :, :, :])
            plot.show()
            exit()

        # 5. Prepare model
        model = ModelCT().to(self.device)

        # 6. Define loss function, optimizer and scheduler
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight) # pos_weight for class imbalance
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, multiplicator, last_epoch=-1)
        # 7. Create post validation transforms and handlers
        path_to_tensorboard = os.path.join(self.out_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=path_to_tensorboard)
        valid_post_transforms = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
            ]
        )
        valid_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(summary_writer=writer, output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=path_to_model,
                save_dict={"model": model},
                save_key_metric=True),
            MetricsSaver(save_dir=path_to_model, metrics=['Valid_AUC', 'Valid_ACC']),
        ]
        # 8. Create validatior
        discrete = AsDiscrete(threshold_values=True)
        evaluator = SupervisedEvaluator(
            device=self.device,
            val_data_loader=valid_loader,
            network=model,
            post_transform=valid_post_transforms,
            key_val_metric={"Valid_AUC": ROCAUC(output_transform=lambda x: (x["pred"], x["label"]))},
            additional_metrics={"Valid_Accuracy": Accuracy(output_transform=lambda x: (discrete(x["pred"]), x["label"]))},
            val_handlers=valid_handlers,
            amp=False,
        )
        # 9. Create trainer

        # Loss function does the last sigmoid, so we dont need it here.
        train_post_transforms = Compose(
            [
                # Empty
            ]
        )
        logger = MetricLogger(evaluator=evaluator)
        train_handlers = [
            logger,
            LrScheduleHandler(lr_scheduler=scheduler, print_lr=True),
            ValidationHandlerCT(
                validator=evaluator,
                start=validation_epoch,
                interval=validation_interval,
                epoch_level=True),
            StatsHandler(tag_name="loss", output_transform=lambda x: x["loss"]),
            TensorBoardStatsHandler(
                summary_writer=writer,
                tag_name="Train_Loss",
                output_transform=lambda x: x["loss"]),
            CheckpointSaver(
                save_dir=path_to_model,
                save_dict={"model": model, "opt": optimizer},
                save_interval=1,
                n_saved=1),
        ]

        trainer = SupervisedTrainer(
            device=self.device,
            max_epochs=total_epoch,
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=loss_function,
            post_transform=train_post_transforms,
            train_handlers=train_handlers,
            amp=False,
        )
        # 10. Run trainer
        trainer.run()
        # 11. Save results
        np.save(path_to_model + '/AUCS.npy', np.array(logger.metrics['Valid_AUC']))
        np.save(path_to_model + '/ACCS.npy', np.array(logger.metrics['Valid_ACC']))
        np.save(path_to_model + '/LOSSES.npy', np.array(logger.loss))
        np.save(path_to_model + '/PARAMETERS.npy', np.array(hyperparameters))

        return path_to_model
