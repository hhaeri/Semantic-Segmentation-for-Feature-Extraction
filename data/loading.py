"""
Handles data loading, preprocessing, and augmentation for images and masks.
Defines generators for training and validation datasets.
Author: Hanieh Haeri
Created on: 1/20/2023
"""
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
import segmentation_models as sm
from config import *

# Preprocessing setup
scaler = MinMaxScaler()
preprocess_input = sm.get_preprocessing(BACKBONE)

def preprocess_data(img, mask, num_class=NUM_CLASSES):
    """
    Scales the input images and converts mask to one-hot format.
    """
    img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
    img = preprocess_input(img)
    mask = to_categorical(mask, num_class)
    return (img, mask)

def train_generator(img_path, mask_path, batch_size=BATCH_SIZE):
    """
    Generates batches of image and mask pairs with augmentation.
    Uses the same seed for reproducibility and sync between images and masks.
    """
    datagen_args = dict(horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
    image_gen = ImageDataGenerator(**datagen_args).flow_from_directory(img_path, class_mode=None, batch_size=batch_size, seed=24)
    mask_gen = ImageDataGenerator(**datagen_args).flow_from_directory(mask_path, class_mode=None, color_mode='grayscale', batch_size=batch_size, seed=24)
    train_gen = zip(image_gen, mask_gen)

    for (img, mask) in train_gen:
        yield preprocess_data(img, mask)

def get_train_val_generators():
    train_gen = train_generator(TRAIN_IMG_PATH, TRAIN_MASK_PATH)
    val_gen = train_generator(VAL_IMG_PATH, VAL_MASK_PATH)
    return train_gen, val_gen
