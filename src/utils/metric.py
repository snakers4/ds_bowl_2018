import numpy as np

def calculate_ap(y_pred, gt_masks):
    height, width, num_masks = gt_masks.shape[1],gt_masks.shape[2], gt_masks.shape[0]
    
    # Make a ground truth label image (pixel value is index of object label)
    # Note that labels will contain the background label
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[gt_masks[index] > 0] = index + 1    
        
    # y_pred should also contain background labels
    # y_pred should contain it if it is taken from wt transform
        
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred)) 
    
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection 
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union   

    # Loop over IoU thresholds
    prec = []
    # print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    # print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    
    return prec, np.mean(prec)

# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn    