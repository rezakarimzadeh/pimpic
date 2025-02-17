import pytorch_lightning as pl
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, EnsureType, AsDiscrete
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
from .unet3d import Decoder3D, Encoder3D
from .cl_unet import *

    
class SegUNet_pl(pl.LightningModule):
    def __init__(self, config, cl_type=None, pretrained_dir=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = config['learning_rate']
        print("learning rate: ", self.lr)
        self.config = config
        self.cl_type = cl_type
        if cl_type == "pim":
            self.encoder = EncoderPiM.load_from_checkpoint(checkpoint_path=pretrained_dir, config=config)
        elif cl_type == "pic":
            self.encoder = EncoderPiC.load_from_checkpoint(checkpoint_path=pretrained_dir, config=config)
        elif cl_type =="pimpic":
            self.encoder = EncoderPiMPiC.load_from_checkpoint(checkpoint_path=pretrained_dir, config=config)
        else:
            self.encoder = Encoder3D(config)

        self.decoder = Decoder3D(config)
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cuda"), AsDiscrete(argmax=True, to_onehot=3)])
        self.post_label = Compose([EnsureType("tensor", device="cuda"), AsDiscrete(to_onehot=3)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

    def forward(self, x):
        if self.cl_type == "pim":
           x, enc_features, projected_features = self.encoder(x)
        else: 
           x, enc_features = self.encoder(x)
        out = self.decoder(x, enc_features)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        sw_batch_size = 4  # Sliding window inference batch size

        # Perform sliding window inference
        outputs = sliding_window_inference(
            inputs=images, 
            roi_size=self.config["roi_size"], 
            sw_batch_size=sw_batch_size, 
            predictor=self.forward
        )

        # Compute loss
        loss = self.loss_function(outputs, labels)

        # Post-process outputs and labels
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        # Compute Dice metric
        self.dice_metric(y_pred=outputs, y=labels)

        # Convert loss to a Python number before appending
        loss_item = loss.detach().cpu().item()

        d = {"val_loss": loss_item, "val_number": len(outputs)}

        # Log loss properly
        self.log("val_loss", loss, on_epoch=True, on_step=True, sync_dist=True)
        
        self.validation_step_outputs.append(d)

        return d
    
    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0

        for output in self.validation_step_outputs:
            val_loss += output["val_loss"]  # Summing up loss values
            num_items += output["val_number"]  # Counting total number of samples

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        # Explicitly define mean_val_loss as a PyTorch tensor
        mean_val_loss = torch.tensor(val_loss / num_items, dtype=torch.float32, device=self.device)


        # Log metrics correctly
        self.log("mean_val_dice", mean_val_dice, on_epoch=True, on_step=False)
        self.log("mean_val_loss", mean_val_loss, on_epoch=True, on_step=False)

        # Update best validation Dice score
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

            print(
                f"current epoch: {self.current_epoch} "
                f"current mean dice: {mean_val_dice:.4f}"
                f"\nbest mean dice: {self.best_val_dice:.4f} "
                f"at epoch: {self.best_val_epoch}"
            )

        self.validation_step_outputs.clear()  # Free memory
    

class SegUNet_test(pl.LightningModule):
    def __init__(self, config, cl_type=None):
        super().__init__()
        self.lr = config['learning_rate']
        self.config = config
        self.cl_type = cl_type
        if cl_type == "pim":
            self.encoder = EncoderPiM(config=config)
        elif cl_type == "pic":
            self.encoder = EncoderPiC(config=config)
        elif cl_type =="pimpic":
            self.encoder = EncoderPiMPiC(config=config)
        else:
            self.encoder = Encoder3D(config)

        self.decoder = Decoder3D(config)


    def forward(self, x):
        if self.cl_type == "pim":
           x, enc_features, projected_features = self.encoder(x)
        else: 
           x, enc_features = self.encoder(x)
        out = self.decoder(x, enc_features)
        return out
        