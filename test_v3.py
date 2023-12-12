import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Breast_dataset
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from torchvision.utils import save_image
from PIL import Image
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./data/Synapse', help='root dir for validation volume data')  # change default
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--test_epochs', '--max_epochs', type=int, 
                    default=1500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=50,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, 
                    default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true",
                    default=True, help='whether to save results during inference') #add default
parser.add_argument('--output_folder', '-o', type=str,
                    default='./output', help='saving prediction as nii!') #add default
parser.add_argument('--model_folder', type=str,
                    default='./output', help='path for model') #add default
parser.add_argument('--prefix', type=str,
                    default='', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, 
                    default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, 
                    default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, 
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str,
                    default="configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', ) # add default
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ",
                    default=None, nargs='+',)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='full', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def save_images(image, label, pred_img, output_path, filename):
    image_filename = filename + '_image.png'
    label_filename = filename + '_label.png'
    pred_filename = filename + '_pred.png'
    composite_filename = filename + '_mix.png'
    image_path = os.path.join(output_path, image_filename)
    label_path = os.path.join(output_path, label_filename)
    pred_path = os.path.join(output_path, pred_filename)
    composite_path = os.path.join(output_path, composite_filename)
    
    # Save image

    # Save label

    # Save prediction
    
    result = mask_to_image(pred_img)
    result.save(pred_path)

    # Create a composite image for visual comparison
    composite_image = np.zeros((pred_img.shape[0], pred_img.shape[1], 3), dtype=np.uint8)
    composite_image[:, :, 0] = (label == 1) * 255  # Red channel for the ground truth
    composite_image[:, :, 1] = (pred_img == 1) * 255  # Green channel for the prediction

    composite_image = Image.fromarray(composite_image)
    composite_image.save(composite_path)

    logging.info('Images saved to {}'.format(output_path))


def inference(args, model):
    model_epoch = args.test_epochs
    batch_size = args.batch_size
    output_folder = os.path.join(args.output_folder, f"predictions_{model_epoch}")
    filename_prefix = args.prefix
    os.makedirs(output_folder,exist_ok=True)

    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=12)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    param_list = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader),total=len(testloader)):
        image, label, filename = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        #save_numpy_image(image, output_folder, filename)
        metric_i, pred_img = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                       z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        mean_dice = np.mean(metric_i, axis=0)[0]
        mean_hd95 = np.mean(metric_i, axis=0)[1]
        mean_iou = np.mean(metric_i, axis=0)[2]

        logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_IoU %f' % (i_batch, filename, mean_dice, mean_hd95, mean_iou))

        param_list.append({
            "Case_Name": filename,
            "Mean_IoU": mean_iou,
            "Mean_Dice": mean_dice,
            "Mean_hd95": mean_hd95,
        })
        
        # Save Images
        #sparate by miou rate
        if mean_iou > 0.9:
            output_path = os.path.join(output_folder, '0.9+')
        elif mean_iou > 0.8:
            output_path = os.path.join(output_folder, '0.8+')
        else:
            output_path = os.path.join(output_folder, '0.8-')

        # make folder if not exist
        os.makedirs(output_path, exist_ok=True)

        # add prefix and remove file type
        if filename_prefix:
            out_filename = filename_prefix + '_' + filename
        else:
            out_filename = filename

        save_images(image, label, pred_img, output_path, out_filename)

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_IoU %f' % (i, metric_list[i-1][0], metric_list[i-1][1], metric_list[i-1][2]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    mean_iou = np.mean(metric_list, axis=0)[2]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    
    # calculate avarage Mean_IoU, mean_dice, mean_hd95
    total_iou = 0
    total_dice = 0 
    total_hd95 = 0
    for param in param_list:
        total_iou += param["Mean_IoU"]
        total_dice += param["Mean_Dice"]
        total_hd95 += param["Mean_hd95"]
    
    # calculate avarage
    total_iou /= len(param_list)
    total_dice /= len(param_list)
    total_hd95 /= len(param_list)

    logging.info(f"Average: mean_dice {total_dice:.6}, mean_hd95 {total_hd95:.6}, Mean_IoU {total_iou:.6}")

    param_list.append({
        "Case_Name": "Average",
        "Mean_IoU": total_iou,
        "Mean_Dice": total_dice,
        "Mean_hd95": total_hd95,
    })

    # Save MIoU values to a CSV file
    with open(os.path.join(output_folder, f"{model_epoch}_{filename_prefix}_miou.csv"), mode='w', newline='') as csv_file:
        # Create a DictWriter object for the CSV file
        writer = csv.DictWriter(csv_file, fieldnames=param_list[0].keys())
        
        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerows(param_list)

    return "Testing Finished!"


if __name__ == "__main__":
    
    dataset_name = args.dataset
    model_epoch = args.test_epochs
    model_folder = args.model_folder
    args.is_pretrain = True

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Breast_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 2,
            'z_spacing': 0,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True


    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(model_folder, f'epoch_{model_epoch}.pth')
    msg = net.load_state_dict(torch.load(snapshot))
    print("self trained swin unet",msg)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder,snapshot_name+".txt"), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_folder, f"predictions")
        test_save_path = args.test_save_dir 
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net)


