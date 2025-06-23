"""
Loads trained model and evaluates performance on validation/test data.
Includes Mean IoU calculation and visualization of predictions vs. ground truth.
Author: Hanieh Haeri
Created on: 1/20/2023
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.metrics import MeanIoU
from config import NUM_CLASSES

def run_inference(model_path, test_gen):
    model = load_model(model_path, compile=False)
    test_image, test_mask = next(test_gen)
    pred = model.predict(test_image)
    pred_argmax = np.argmax(pred, axis=-1)
    mask_argmax = np.argmax(test_mask, axis=-1)

    IOU = MeanIoU(num_classes=NUM_CLASSES)
    IOU.update_state(mask_argmax, pred_argmax)
    print("Mean IoU:", IOU.result().numpy())

    plt.imshow(pred_argmax[0])
    plt.title("Prediction")
    plt.show()
