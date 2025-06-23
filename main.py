"""
Main script to run the end-to-end semantic segmentation pipeline using U-Net.
This script ties together data loading, model training, evaluation, and inference.
Author: Hanieh Haeri
Created on: 1/20/2023
"""
from config import *
from data.loading import get_train_val_generators
from models.unet import build_unet
from trainers.trainer import train_model

if __name__ == "__main__":
    train_gen, val_gen = get_train_val_generators()
    model = build_unet()
    train_model(model, train_gen, val_gen)
