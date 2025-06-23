import segmentation_models as sm
from config import *

def build_unet():
    model = sm.Unet(BACKBONE, encoder_weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), classes=NUM_CLASSES, activation='softmax')
    model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
    return model
