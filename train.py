from models.seg_unet import SegUNet_pl
from models.cl_unet import EncoderPiM, EncoderPiC, EncoderPiMPiC
from dataloaders.random_crop_coordinates_dataloader import get_training_dataloaders
from dataloaders.segmentation_dataloaders import get_segmentation_dataloaders
from dataloaders import data_handler
from utils.universal_trainer import contrastive_pl_trainer, segmentation_pl_trainer
from utils.seg_test_utils import get_test_dataloader, evaluate_model
import argparse
import os 
from models.seg_unet import SegUNet_test


parser = argparse.ArgumentParser(
        description="training parser for patch-based cl and segmentation models training"
    )
parser.add_argument(
        "--config", "-c",
        type=str,
        default='./config.yaml',
        help="Path to the config file."
    )

parser.add_argument(
    "--method", "-m",
    type=str,
    required=True,
    help="method to be trained, 'contrastive learning pretraining' OR 'segmentation' (cl or seg)"
)


parser.add_argument(
    "--cl_method", "-clm",
    type=str,
    required=False,
    help="method of training contrastive learning model (pim, pic, pimpic)"
)

parser.add_argument(
    "--seg_method", "-segm",
    type=str,
    required=False,
    help="method of initialization segmentation model (random, pim, pic, pimpic)"
)

def train_cl_model(args):
    print(f"start training contrastive learning ... \n method to train: {args.cl_method} \n ")
    config = data_handler.read_yaml_file(args.config)
    epochs = config['num_epochs'] 
    train_dataloader = get_training_dataloaders(config)
    print(f"train loader len: {len(train_dataloader)}")
    if args.cl_method == "pim":
        model = EncoderPiM(config)
        filename = "pim_cl"
    elif args.cl_method == "pic":
        model = EncoderPiC(config)
        filename = "pic_cl"
    elif args.cl_method == "pimpic":
        model = EncoderPiMPiC(config)
        filename = "pimpic_cl"
    contrastive_pl_trainer(model=model, train_dataloader=train_dataloader, epochs=epochs, filename=filename, 
                           checkpoint_dir=config["checkpoint_dir_cl"])


def train_seg_model(args):
    save_dir_results = "./results"
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    config = data_handler.read_yaml_file(args.config)
    test_loader, test_datalist = get_test_dataloader(config)
    for percent in [5, 10, 20, 50, 100]:
        print(f"start training segmentation model ... \n weight initialization: {args.seg_method}")
        train_dataloader, val_dataloader = get_segmentation_dataloaders(config, percent)
        print(f"precent of data for training segmentation model on: {percent}")
        print(f"train loader len: {len(train_dataloader)}, val dataloader len:{len(val_dataloader)}")
        epochs = config['num_epochs'] 
        
        if args.seg_method == "pim":
            filename = f"seg_pim_trp_{percent}"
            model = SegUNet_pl(config, cl_type=args.seg_method, pretrained_dir=f'{config["checkpoint_dir_cl"]}/{args.seg_method}_cl.ckpt')
        elif args.seg_method == "pic":
            model = SegUNet_pl(config, cl_type=args.seg_method, pretrained_dir=f'{config["checkpoint_dir_cl"]}/{args.seg_method}_cl.ckpt')
            filename = f"seg_pic_trp_{percent}"
        elif args.seg_method == "pimpic":
            model = SegUNet_pl(config, cl_type=args.seg_method, pretrained_dir=f'{config["checkpoint_dir_cl"]}/{args.seg_method}_cl.ckpt')
            filename = f"seg_pimpic_trp_{percent}"
        elif args.seg_method == "random":
            model = SegUNet_pl(config)
            filename = f"seg_random_trp_{percent}"

        segmentation_pl_trainer(model=model, train_dataloader=train_dataloader, val_dataloader= val_dataloader, 
                                epochs=epochs, filename=filename, checkpoint_dir=config["checkpoint_dir_seg"])
        
        # evaluate trained mode
        checkpoint_dir = os.path.join(config['checkpoint_dir_seg'], f"seg_{args.seg_method}_trp_{percent}.ckpt")
        model = SegUNet_test.load_from_checkpoint(checkpoint_path=checkpoint_dir,
                                                config=config, cl_type=args.seg_method)
        
        device = model.device
        model.eval()
        evaluation_results = evaluate_model(test_loader, test_datalist, model, device, roi_size=config["roi_size"])
        file_path = os.path.join(save_dir_results, f"seg_{args.seg_method}_trp_{percent}")
        data_handler.save_json_file(evaluation_results, file_path)

def validate_method(args):
    if args.method == "cl":
        valid_methods = ['pim', 'pic', 'pimpic']
        # Check if cl_method is set and valid
        if not hasattr(args, "cl_method") or args.cl_method not in valid_methods:
            raise ValueError(
                f"Invalid value for cl_method: {getattr(args, 'cl_method', None)}. "
                f"Allowed values are: {', '.join(valid_methods)}"
            )
    elif args.method == "seg":
        valid_methods = ["random", 'pim', 'pic', 'pimpic']
        # Check if seg_method is set and valid
        if not hasattr(args, "seg_method") or args.seg_method not in valid_methods:
            raise ValueError(
                f"Invalid value for seg_method: {getattr(args, 'seg_method', None)}. "
                f"Allowed values are: {', '.join(valid_methods)}"
            )
    else:
        pass


if __name__ == "__main__":
    args = parser.parse_args()
    if args.method == 'cl':
        validate_method(args)
        train_cl_model(args)
    elif args.method == "seg":
        validate_method(args)
        train_seg_model(args)
    else:
        raise NotImplementedError(f"This method [{args.method}] is not implemented.")
    
