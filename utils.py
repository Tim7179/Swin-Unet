import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.autograd import Function
# MIoU
#from ignite.metrics import IoU


def calculate_iou(predicted_mask, ground_truth_mask):
    # Convert NumPy arrays to PyTorch tensors and move them to GPU if available
    predicted_mask = torch.tensor(predicted_mask, dtype=torch.bool, device='cuda')
    ground_truth_mask = torch.tensor(ground_truth_mask, dtype=torch.bool, device='cuda')
    
    intersection = torch.logical_and(predicted_mask, ground_truth_mask)
    union = torch.logical_or(predicted_mask, ground_truth_mask)
    
    # Calculate IoU on GPU
    iou = torch.sum(intersection) / torch.sum(union)
    
    # Move the result back to CPU (if you want to print it)
    iou = iou.cpu().item()
    
    return iou

# Function to calculate mIoU on GPU
def calculate_miou(predicted_masks, ground_truth_masks):
    class_iou = []
    num_classes = predicted_masks.shape[0]
    
    for class_idx in range(num_classes):
        class_iou.append(calculate_iou(predicted_masks[class_idx], ground_truth_masks[class_idx]))
    
    mIoU = sum(class_iou) / len(class_iou)
    
    return mIoU

def miou_coeff(predicted_masks, ground_truth_masks):
    return calculate_miou(predicted_masks, ground_truth_masks)
'''
class MIoU(Function):

    @staticmethod
    def forward(ctx, pred, target):
        # Calculate intersection and union
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        
        # Calculate MIoU with a minimum epsilon value to avoid division by zero
        epsilon = 1e-6  # You can adjust this epsilon value as needed
        iou = (intersection + epsilon) / (union + epsilon)
        
        # Save for backward pass
        ctx.save_for_backward(pred, target)

        return iou
    
    @staticmethod
    def backward(ctx, grad_output):
        #反向傳播計算梯度
        pred, target = ctx.saved_tensors
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        d_iou = grad_output * ((target * union) - (intersection * pred)) / (union ** 2)
        return d_iou, -d_iou

def miou_coeff(inputs, target):
    """Mean IoU for batches"""
    num_samples = inputs.size(0)
    s = torch.FloatTensor(1).zero_().to(inputs.device)
    
    for i in range(num_samples):
        s += MIoU.apply(inputs[i], target[i])
    
    return s / num_samples
'''


## MIoU

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-10
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            raise ValueError('Model output contains NaN or infinite values')

        if torch.isnan(target).any() or torch.isinf(target).any():
            raise ValueError('Target contains NaN or infinite values')

        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list