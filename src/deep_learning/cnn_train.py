"""
cnn_train.py

Training utilities for CNN.
"""

from pathlib import Path
import json

import torch

from tqdm import tqdm

def get_device():

    if torch.cuda.is_available():

        return torch.device("cuda")

    return torch.device("cpu")

def train_one_epoch(

    model,

    loader,

    criterion,

    optimizer,

    device

):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(

        loader,

        leave=False

    )

    for signals, labels in progress:

        signals = signals.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(signals)

        loss = criterion(

            outputs,

            labels

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predicted = outputs.argmax(1)

        total += labels.size(0)

        correct += (

            predicted == labels

        ).sum().item()

        progress.set_description(

            f"Loss {loss.item():.4f}"

        )

    epoch_loss = running_loss / len(loader)

    epoch_acc = correct / total

    return epoch_loss, epoch_acc

@torch.no_grad()

def validate(

    model,

    loader,

    criterion,

    device

):

    model.eval()

    running_loss = 0

    correct = 0

    total = 0

    for signals, labels in loader:

        signals = signals.to(device)

        labels = labels.to(device)

        outputs = model(signals)

        loss = criterion(

            outputs,

            labels

        )

        running_loss += loss.item()

        predicted = outputs.argmax(1)

        total += labels.size(0)

        correct += (

            predicted == labels

        ).sum().item()

    loss = running_loss / len(loader)

    accuracy = correct / total

    return loss, accuracy

class EarlyStopping:

    def __init__(

        self,

        patience=5,

        delta=0.0

    ):

        self.patience = patience

        self.delta = delta

        self.best_accuracy = 0.0

        self.counter = 0

        self.stop = False

    def __call__(

        self,

        accuracy

    ):

        if accuracy > self.best_accuracy + self.delta:

            self.best_accuracy = accuracy

            self.counter = 0

            return True

        self.counter += 1

        if self.counter >= self.patience:

            self.stop = True

        return False
    
def save_checkpoint(

    model,

    optimizer,

    epoch,

    accuracy,

    path

):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "accuracy": accuracy

    }

    torch.save(

        checkpoint,

        path

    )
def save_history(

    history,

    path

):

    with open(

        path,

        "w"

    ) as file:

        json.dump(

            history,

            file,

            indent=4

        )

def train_model(

    model,

    train_loader,

    val_loader,

    criterion,

    optimizer,

    scheduler,

    epochs,

    device,

    model_path,

    history_path

):

    history = {

        "train_loss": [],

        "train_accuracy": [],

        "val_loss": [],

        "val_accuracy": []

    }

    stopper = EarlyStopping(

        patience=5

    )

    best_accuracy = 0

    for epoch in range(epochs):

        print()

        print("=" * 60)

        print(

            f"Epoch {epoch+1}/{epochs}"

        )

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )

        val_loss, val_acc = validate(

            model,

            val_loader,

            criterion,

            device

        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)

        history["train_accuracy"].append(train_acc)

        history["val_loss"].append(val_loss)

        history["val_accuracy"].append(val_acc)

        print(

            f"Train Loss : {train_loss:.4f}"

        )

        print(

            f"Train Accuracy : {train_acc:.4f}"

        )

        print(

            f"Validation Loss : {val_loss:.4f}"

        )

        print(

            f"Validation Accuracy : {val_acc:.4f}"

        )

        if stopper(val_acc):

            best_accuracy = val_acc

            save_checkpoint(

                model,

                optimizer,

                epoch,

                val_acc,

                model_path

            )

            print()

            print("Best model saved.")

        if stopper.stop:

            print()

            print("Early stopping triggered.")

            break

    save_history(

        history,

        history_path

    )

    return history, best_accuracy


