"""
Utility functions used across the project, including plotting and image helpers.
Can be extended for logging, metrics formatting, or custom callbacks.
Author: Hanieh Haeri
Created on: 1/20/2023
"""
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_training_history(history):
    """
    Plots training and validation loss and IoU over epochs.
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    iou = history.history['iou_score']
    val_iou = history.history['val_iou_score']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'y', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, iou, 'y', label='Training IoU')
    plt.plot(epochs, val_iou, 'r', label='Validation IoU')
    plt.legend()
    plt.title('IoU')
    plt.show()

def visualize_predictions(model, image_batch, mask_batch):
    """
    Displays random test image, its ground-truth mask, and the predicted mask.
    """
    idx = random.randint(0, len(image_batch) - 1)
    pred = model.predict(image_batch)
    pred_argmax = np.argmax(pred, axis=3)

    true_mask = np.argmax(mask_batch, axis=3)

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.title("Image")
    plt.imshow(image_batch[idx])

    plt.subplot(132)
    plt.title("Ground Truth")
    plt.imshow(true_mask[idx], cmap='gray')

    plt.subplot(133)
    plt.title("Prediction")
    plt.imshow(pred_argmax[idx], cmap='gray')
    plt.show()
