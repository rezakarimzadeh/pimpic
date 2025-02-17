from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import os


def contrastive_pl_trainer(model, train_dataloader, epochs, filename, checkpoint_dir):
    logger = TensorBoardLogger("logs", name=filename)
    file_path = os.path.join(checkpoint_dir, f"{filename}.ckpt")

    # Check if the file already exists and delete it
    if os.path.exists(file_path):
        print(f"Deleting existing checkpoint: {file_path}")
        os.remove(file_path)

    # save model after each epoch
    checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,  # Directory where checkpoints will be saved
    filename=filename,  # Static filename to overwrite each time
    save_top_k=1,  # Keep only the latest checkpoint
    verbose=True
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],#, EarlyStopping(monitor='val_loss', patience=10)],
        accelerator="cuda", devices=1, 
        precision='16-mixed',
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloader)
    print('best model: ', checkpoint_callback.best_model_path)


def segmentation_pl_trainer(model, train_dataloader, epochs, filename, checkpoint_dir, val_dataloader):
    logger = TensorBoardLogger("logs", name=filename)
    file_path = os.path.join(checkpoint_dir, f"{filename}.ckpt")

    # Check if the file already exists and delete it
    if os.path.exists(file_path):
        print(f"Deleting existing checkpoint: {file_path}")
        os.remove(file_path)
        
    # save the best model
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=filename,
        save_top_k=1,
        verbose=True,
        monitor='mean_val_loss',
        mode='min',
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='mean_val_loss', patience=15)],
        accelerator="cuda", devices=1,
        precision='16-mixed',
        log_every_n_steps=1
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    print('best model: ', checkpoint_callback.best_model_path)