from dataloaders import data_handler
from models.seg_unet import SegUNet_test
import argparse
from utils.seg_test_utils import get_test_dataloader, evaluate_model
import os

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
    "--seg_method", "-segm",
    type=str,
    required=False,
    help="method of initialization segmentation model (random, pim, pic, pimpic)"
)

parser.add_argument(
    "--train_percentage", "-trp",
    type=int,
    required=False,
    help="percentage of training data"
)

def validate_method(args):
    valid_methods = ["random", 'pim', 'pic', 'pimpic']
    if not hasattr(args, "seg_method") or args.seg_method not in valid_methods:
        raise ValueError(
            f"Invalid value for seg_method: {getattr(args, 'seg_method', None)}. "
            f"Allowed values are: {', '.join(valid_methods)}"
        )


if __name__ == "__main__":  
    save_dir_results = "./results"
    if not os.path.exists(save_dir_results):
        os.makedirs(save_dir_results)

    args = parser.parse_args()
    validate_method(args)
    config = data_handler.read_yaml_file(args.config)
    train_percentage = args.train_percentage 

    test_loader, test_datalist = get_test_dataloader(config)
    seg_method = args.seg_method
    
    checkpoint_dir = os.path.join(config['checkpoint_dir_seg'],f"seg_{seg_method}_trp_{train_percentage}.ckpt")
    model = SegUNet_test.load_from_checkpoint(checkpoint_path=checkpoint_dir,
                                            config=config, cl_type=seg_method)
    device = "cuda" #model.device
    model = model.to(device)
    model.eval()

    evaluation_results = evaluate_model(test_loader, test_datalist, model, device, roi_size=config["roi_size"])
    file_path = os.path.join(save_dir_results, f"seg_{seg_method}_trp_{train_percentage}")
    data_handler.save_json_file(evaluation_results, file_path)