import numpy as np
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

from .trainingset import PickDataset
from .model import DraftMLP

def evaluate_model(val_dataloader, network):
    """
    Evaluate model pick accuracy on validation dataset. 
    """
    # Count number correct picks. 
    num_correct, num_incorrect = 0, 0
    for pool, pack, human_pick_vector in val_dataloader: # Assumes batch size of 1. 
        # TODO: vectorize for performance. 
        human_pick_index = torch.argmax(human_pick_vector.int(), 1)
        network.eval()
        with torch.no_grad():
            bot_pick_vector = network(pool.float(), pack.float())
            bot_picks_index = torch.argmax(bot_pick_vector, 1)        
        if torch.equal(human_pick_index, bot_picks_index):
            num_correct += 1
        else:
            num_incorrect += 1

    # Return and print result. 
    percent_correct = 100 * num_correct / (num_correct + num_incorrect)
    print(f"Validation set pick accuracy = {round(percent_correct, 1)}%")
    return percent_correct

def train_model(train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                network: torch.nn.Module,
                epochs: int = 20,
                learning_rate: float = 0.01,
                experiment_name: str = "test",
                model_folder: str = "../data/models/"):    
    """
    Train and evaluate model. 
    """
    # Optimizer parameters. 
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr = learning_rate)

    # Initial evaluation. 
    print("Starting to train model")
    best_percent_correct = evaluate_model(val_dataloader, network)

    # Train model.     
    t0 = time.time()
    time_last_message = t0
    for epoch in range(epochs):
        network.train()
        epoch_training_loss = list()
        print(f"\nStarting epoch {epoch}")
        for i, (pool, pack, pick_vector) in enumerate(train_dataloader):
            optimizer.zero_grad()
            predicted_pick = network(pool.float(), pack.float())
            loss = loss_fn(predicted_pick, pick_vector.float())
            loss.backward()
            optimizer.step()
            epoch_training_loss.append(loss.item())

            # Provide updates every 10 seconds. 
            if time_last_message - time.time() > 10:
                examples_processed = (i + 1) * pool.shape[0]
                print(f"Training complete on {examples_processed} examples, time={round(time.time() - t0, 1)}")

        print(f"Training loss: {round(np.mean(epoch_training_loss), 4)}")

        # Evaluate every 2 epochs
        if epoch % 2 == 0 and epoch > 0:
            # Evaluation. 
            network = network.eval()
            percent_correct = evaluate_model(val_dataloader, network)

            # Save best model. 
            if percent_correct > best_percent_correct:
                best_percent_correct = percent_correct
                weights_path = model_folder + experiment_name + ".pt" # TODO: change this
                print(f"Saving model weights to {weights_path}")
                torch.save(network.state_dict(), weights_path)                
    print("Training completed.")
    return network