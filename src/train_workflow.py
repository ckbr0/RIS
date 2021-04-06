import sys
import os
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plot
from matplotlib.widgets import Slider

import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader
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
    CacheDataset,
    Dataset,
    PersistentDataset,
    list_data_collate,
)
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    AsDiscrete,
    Activationsd,
    Compose,
    CropForegroundd,
    LoadImaged,
    SqueezeDimd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    MaskIntensityd,
    Lambdad,
    ToNumpyd,
)
from monai.metrics import compute_roc_auc
from monai.utils import set_determinism

from model import ModelCT
from utils import get_data_from_info, multi_slice_viewer
from validation_handler import ValidationHandlerCT

class TrainingWorkflow():

    def __init__(self, data_dir, hackathon_dir, out_dir, cache_dir, unique_name):
        self.data_dir = data_dir
        self.hackathon_dir = hackathon_dir
        self.out_dir = out_dir
        self.unique_name = unique_name
        self.image_data_dir = os.path.join(hackathon_dir, 'images', 'train')
        self.seg_data_dir = os.path.join(hackathon_dir, 'segmentations', 'train')
        self.cache_dir = cache_dir
        self.persistent_dataset_dir = os.path.join(cache_dir, 'persistent')
   
    def transformations(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                #LoadImaged(keys=["seg"]),
                #ToNumpyd(keys=["seg"]),
                #Lambdad(keys=["seg"], func=lambda x: np.moveaxis(x, 0, -1)),
                MaskIntensityd(keys=["image"], mask_key="seg"),
                AddChanneld(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                ToTensord(keys=["image"]),
            ]
        )

        # NOTE: No random transforms in the validation data
        valid_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                #LoadImaged(keys=["seg"]),
                #ToNumpyd(keys=["seg"]),
                #Lambdad(keys=["seg"], func=lambda x: np.moveaxis(x, 0, -1)),
                MaskIntensityd(keys=["image"], mask_key="seg"),
                AddChanneld(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                ToTensord(keys=["image"]),
            ]
        )
        
        return train_transforms, valid_transforms

    def train(self, train_info, valid_info, hyperparameters, cuda=None): 

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        start_dt = datetime.datetime.now()
        start_dt_string = start_dt.strftime('%d/%m/%Y %H:%M:%S')
        print(f'Training started: {start_dt_string}')

        # 0. Create torch device
        if not cuda:
            pin_memory = False
            device = torch.device('cpu')
        else:
            pin_memory = True
            device = torch.device('cuda')

        # 1. Create folders to save the model
        timedate_info= str(datetime.datetime.now()).split(' ')[0] + '_' + str(datetime.datetime.now().strftime("%H:%M:%S")).replace(':', '-')
        path_to_model = os.path.join(self.out_dir, 'trained_models', self.unique_name +  '_' + timedate_info)
        os.mkdir(path_to_model)

        # 2. Load hyperparameters
        learning_rate = hyperparameters['learning_rate']
        weight_decay = hyperparameters['weight_decay']
        total_epoch = hyperparameters['total_epoch']
        multiplicator = hyperparameters['multiplicator']
        validation_epoch = hyperparameters['validation_epoch']
        validation_interval = hyperparameters['validation_interval']

        # 3. Consider class imbalance
        negative, positive = 0, 0
        for _, label in train_info:
            if int(label) == 0:
                negative += 1
            elif int(label) == 1:
                positive += 1
        
        pos_weight = torch.Tensor([(negative/positive)]).to(device)

        # 4. Create train and validation loaders, batch_size = 10 for validation loader (10 central slices)

        train_data = get_data_from_info(self.image_data_dir, self.seg_data_dir, train_info)
        valid_data = get_data_from_info(self.image_data_dir, self.seg_data_dir, valid_info)

        set_determinism(seed=0)
        train_trans, valid_trans = self.transformations()
        train_dataset = PersistentDataset(data=train_data[:], transform=train_trans, cache_dir=self.persistent_dataset_dir)
        valid_dataset = PersistentDataset(data=valid_data[:], transform=valid_trans, cache_dir=self.persistent_dataset_dir)
        print('7')
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory, num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory, num_workers=2)
        print('6')
        # Perform data checks
        """check_data = {'image': np.load(train_files[0]['image']), 'label': train_files[0]['label']}
        print(check_data["image"].shape, check_data["label"])"""
        """check_data = monai.utils.misc.first(train_loader)
        #print(check_data["image"].shape, check_data["label"])
        multi_slice_viewer(check_data["image"][0, 0, :, :, :])
        plot.show()"""

        #exit()
        # 5. Prepare model
        model = ModelCT().to(device)
        print('9')
        # 6. Define loss function, optimizer and scheduler
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight) # pos_weight for class imbalance
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, multiplicator, last_epoch=-1)
        print('10')
        # 7. Create post validation transforms and handlers
        path_to_tensorboard = os.path.join(self.out_dir, 'tensorboard')
        writer = SummaryWriter(log_dir=path_to_tensorboard)
        print('11')
        valid_post_transforms = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
            ]
        )
        print('12')
        valid_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(summary_writer=writer, output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=path_to_model,
                save_dict={"model": model},
                save_key_metric=True),
            MetricsSaver(save_dir=path_to_model, metrics=['Valid_AUC', 'Valid_ACC']),
        ]
        print('13')
        # 8. Create validatior
        discrete = AsDiscrete(threshold_values=True)
        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=valid_loader,
            network=model,
            post_transform=valid_post_transforms,
            key_val_metric={"Valid_AUC": ROCAUC(output_transform=lambda x: (x["pred"], x["label"]))},
            additional_metrics={"Valid_Accuracy": Accuracy(output_transform=lambda x: (discrete(x["pred"]), x["label"]))},
            val_handlers=valid_handlers,
            amp=False,
        )
        print('14')
        # 9. Create trainer

        # Loss function does the last sigmoid, so we dont need it here.
        train_post_transforms = Compose(
            [
                # Empty
            ]
        )
        print('15')
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
            device=device,
            max_epochs=total_epoch,
            train_data_loader=train_loader,
            network=model,
            optimizer=optimizer,
            loss_function=loss_function,
            post_transform=train_post_transforms,
            train_handlers=train_handlers,
            amp=False,
        )
        print('16')
        # 10. Run trainer
        trainer.run()
        print('17')
        # 11. Save results
        np.save(path_to_model + '/AUCS.npy', np.array(logger.metrics['Valid_AUC']))
        np.save(path_to_model + '/ACCS.npy', np.array(logger.metrics['Valid_ACC']))
        np.save(path_to_model + '/LOSSES.npy', np.array(logger.loss))
        np.save(path_to_model + '/PARAMETERS.npy', np.array(hyperparameters))

        return path_to_model
