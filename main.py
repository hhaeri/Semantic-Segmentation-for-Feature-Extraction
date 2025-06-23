"""
main.py

Author: Hanieh Haeri
Created: 2023-01-20

This is the entry point for training and evaluating the semantic segmentation model.
It loads the data using the data generator, builds the model using trainer.py, trains the model,
and saves the results.
"""

from data.loading import get_train_val_generators
from trainer.trainer import build_model, evaluate_model
from utils.visualize import plot_training_history
from config import *
import matplotlib.pyplot as plt

# Load training and validation generators
train_gen, val_gen, x_val, y_val = get_train_val_generators()

# Build the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = build_model(input_shape)

# Train the model
history = model.fit(
    train_gen,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_gen,
    validation_steps=VAL_STEPS_PER_EPOCH
)

# Save the model
model.save("unet_model.h5")

# Evaluate the model
iou_score = evaluate_model(model, x_val, y_val)
print(f"Mean IoU on validation set: {iou_score:.4f}")

# Plot training history
plot_history(history)
