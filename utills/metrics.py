from tensorflow.keras.metrics import MeanIoU

def compute_mean_iou(y_true, y_pred, num_classes):
    iou = MeanIoU(num_classes=num_classes)
    iou.update_state(y_true, y_pred)
    return iou.result().numpy()
