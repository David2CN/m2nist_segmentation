import numpy as np


def class_wise_metrics(y_true, y_pred, n_classes=11):
    '''
    Computes the class-wise IOU and Dice Score.

    Args:
      y_true (tensor) - ground truth label maps
      y_pred (tensor) - predicted label maps
    '''
    class_wise_iou = []
    class_wise_dice_score = []

    smoothing_factor = 0.00001

    for i in range(n_classes):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
        
        iou = (intersection) / (combined_area - intersection + smoothing_factor)
        class_wise_iou.append(iou)
        
        dice_score =  2 * ((intersection) / (combined_area + smoothing_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score
