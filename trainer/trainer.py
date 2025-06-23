# Author: Hanieh Haeri
# Created: 01/20/2023

from tensorflow.keras.metrics import MeanIoU
import segmentation_models as sm
from config import *

def build_model(input_shape):
    """
    Builds and compiles the UNet model with a given input shape.
    """
    model = sm.Unet(BACKBONE, encoder_weights='imagenet',
                    input_shape=input_shape,
                    classes=N_CLASSES,
                    activation='softmax')
    
    model.compile(optimizer='Adam',
                  loss=sm.losses.categorical_focal_jaccard_loss,
                  metrics=[sm.metrics.iou_score])
    return model

def evaluate_model(model, x_val, y_val):
    """
    Evaluates the model on validation data and computes Mean IoU.
    """
    pred_mask = model.predict(x_val)
    pred_mask_argmax = np.argmax(pred_mask, axis=3)
    true_mask_argmax = np.argmax(y_val, axis=3)

    IOU = MeanIoU(num_classes=N_CLASSES)
    IOU.update_state(pred_mask_argmax, true_mask_argmax)
    return IOU.result().numpy()
