import sys
import os
import time, datetime
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
from utils import get_data_from_info, compute_acc, multi_slice_viewer
from validation_handler import ValidationHandlerCT
from transforms import CTWindowd, RandCTWindowd, CTSegmentation
from nrrd_reader import NrrdReader

class Training():

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
            self.amp = False
            self.device = torch.device('cpu')
        else:
            self.pin_memory = True
            self.amp = True
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
                    padding_mode="zeros",
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
        
        set_determinism(seed=100)
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

        """c = 1
        for d in train_loader:
            img = d["image"]
            print(c, "Size:", img.nelement()*img.element_size()/1024/1024, "shape:", img.shape)
            c += 1"""

        #exit()
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
        discrete = AsDiscrete(threshold_values=True)

        # 7. Creat lists for tracking AUC and Losses during training
        auc = 0
        aucs = []
        acc = 0
        accs = []
        losses = []
        best_auc = -np.inf
        best_epoch = -1
        nb_batches = len(train_loader)
        epoch_len = len(train_dataset) // train_loader.batch_size
        
        # 8. Run training
        for epoch in range(total_epoch):
            start = time.time()
            print('Epoch: %d/%d' % (epoch + 1, total_epoch))
            running_loss = 0
            step = 0
            # A) Train model
            model.train()  # put model in training mode
            for train_item in train_loader:
                # Extract data
                inputs, labels = train_item['image'].to(self.device), train_item['label'].to(self.device)
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Track loss change
                step += 1
                running_loss += loss.item()
                print(f"\tBatch: {step}/{epoch_len}, loss: {loss.item():.4f}")
                writer.add_scalar("Loss", loss.item(), epoch_len * epoch + step)

            # B) Validate model
            if (epoch + 1) >= validation_epoch and (epoch + 1 - validation_epoch) % validation_interval == 0:

                print("\tValidating...")

                predictions = torch.tensor([], dtype=torch.float32, device=self.device)
                trues = torch.tensor([], dtype=torch.int32)
            
                model.eval() # put model in eval mode
                with torch.no_grad():
                    for valid_item in valid_loader:
                        inputs, labels = valid_item['image'].to(self.device), valid_item["label"].to(self.device)
                        prediction = valid_post_transforms(model(inputs))
                        predictions = torch.cat([predictions, prediction], dim=0)
                        trues = torch.cat([trues, labels], dim=0)

                acc = compute_acc(discrete(predictions), trues)
                auc = compute_roc_auc(predictions, trues)

            # C) Track changes, update LR, save best model
            print(f"\tAUC: {auc}, ACC: {acc}, Average loss: {running_loss/nb_batches}, Time: {time.time()-start}")
            writer.add_scalar("auc", auc, epoch + 1)
            writer.add_scalar("acc", acc, epoch + 1)
            
            if (epoch >= total_epoch//2) and (auc > best_auc): # If over 1/2 of epochs and best AUC, save model as best model.
                torch.save(model.state_dict(), os.path.join(path_to_model, '/BEST_model.pth'))
                print("\tSaved new best metric model")
                best_auc = auc
                best_epoch = epoch+1
            else:
                pass
            
            aucs.append(auc)
            accs.append(acc)

            losses.append(running_loss/nb_batches)
            writer.add_scalar("Average loss per epoch", running_loss/nb_batches, epoch+1)
            
            scheduler.step()
            
        np.save(path_to_model + '/AUCS.npy', np.array(aucs))
        np.save(path_to_model + '/ACCS.npy', np.array(accs))
        np.save(path_to_model + '/LOSSES.npy', np.array(losses))
        np.save(path_to_model + '/PARAMETERS.npy', np.array(hyperparameters))
        torch.save(model.state_dict(), path_to_model + '/LAST_model.pth')
        
        end_dt = datetime.datetime.now()
        end_dt_string = end_dt.strftime('%d/%m/%Y %H:%M:%S')
        print(f"Train completed: {end_dt_string}, best metric: {best_auc:.4f} at epoch {best_epoch}")
       
        return aucs, accs, losses, path_to_model
