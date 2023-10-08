import numpy as np
def calculate_miou(pred_mask, mask, num_classes,start=0):
    #data should be finished torch
    iou_list=np.array([])
    for classes in range(start,num_classes):
        pred_binary=(pred_mask==classes).float()
        masks_binary=(mask==classes).float()
        intersection = (pred_binary * masks_binary).sum().item()
        union = (pred_binary + masks_binary).sum().item() - intersection

        iou = intersection / (union + 1e-5)# Add a small epsilon to avoid division by zero
        iou_list=np.append(iou_list,iou)
    miou=np.mean(iou_list)
    return miou