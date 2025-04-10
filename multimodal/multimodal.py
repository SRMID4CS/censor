# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 18:14:48 2025

@author: lakith
"""
import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import accuracy_score
import logging
import datetime

from run_rec import init_logger

# Define training and validation function


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in train_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images, input_ids, attention_mask)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the model

        running_loss += loss.item() * images.size(0)
        # preds = torch.argmax(outputs, dim=1)
        preds = (outputs > 0.2).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradients during validation
        for batch in val_loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images, input_ids, attention_mask)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            running_loss += loss.item() * images.size(0)
            # preds = torch.argmax(outputs, dim=1)
            preds = (outputs > 0.2).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_accuracy


# Train and validate loop with plot generation
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=3):

    current_time = datetime.datetime.now().strftime("%b.%d_%H.%M.%S")
    save_dir = os.path.join(
        "/home/ubuntu/mnt-disk/results/mm_exp_1/", "mm_ex1_gan_free_imdb"+current_time)
    os.makedirs(save_dir, exist_ok=True)
    logger = init_logger(save_dir)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        logger.info("[MM] Epoch: ", epoch+1)
        # Train the model
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate the model
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        logger.info("[MM] " + f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {
                  train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
