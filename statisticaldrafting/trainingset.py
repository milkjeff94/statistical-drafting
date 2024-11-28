import torch
from torch.utils.data import DataLoader, Dataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import time

# def download_file(set: str, type="Premier"):
#     # TODO: implement this. 
#     pass

def remove_basics(draft_chunk: pd.DataFrame) -> pd.DataFrame:
    # Remove basic lands from raw 17lands dataset 
    basic_names = ["Forest", "Island", "Mountain", "Plains", "Swamp"]
    columns_to_drop = ["pack_card_" + b for b in basic_names] + ["pool_" + b for b in basic_names]
    draft_chunk = draft_chunk.drop(columns=columns_to_drop)
    draft_chunk = draft_chunk[~draft_chunk["pick"].isin(basic_names)]
    return draft_chunk

class PickDataset(Dataset):
    def __init__(self, pools, packs, pick_vectors, cardnames):
        self.pools = pools
        self.packs = packs
        self.pick_vectors = pick_vectors
        self.cardnames = cardnames
                            
    def __len__(self):
        return len(self.packs)

    def __getitem__(self,index):
        return torch.from_numpy(self.pools[index]), torch.from_numpy(self.packs[index]), torch.from_numpy(self.pick_vectors[index])

def create_dataset(set_abbreviation: str, 
                   draft_mode: str = "Premier", 
                   overwrite: bool = False,
                   omit_first_days: int = 7,
                   minimum_league: str = "bronze", 
                   train_fraction: float = 0.8,
                   data_folder_17lands: str = "../data/17lands/", 
                   data_folder_training_set: str = "../data/training_sets/") -> None:
    """
    Creates clean training and validation datasets from raw 17lands data. 

    Args:
        set_abbreviation (str): Three letter abbreviation of set to create training set of. 
        draft_mode (str): Use either "Premier" or "Trad" draft data.
        overwrite (bool): If False, won't overwrite an existing dataset for the set and draft mode. 
        omit_first_days (int): Omit this many days from the beginning of the dataset. 
        minimum_league (str): For Premier draft, use only data from drafts at or above this league. 
        train_fraction (float): Fraction of dataset to use for training. 
        data_folder_17lands (str): Folder where raw 17lands files are stored. 
        data_folder_training_set (str): Folder where processed training & validation sets are stored. 
    """
    # TODO: implement download_file() here. 
    
    # Check if training set exists. 
    train_filename = f"{set_abbreviation}_{draft_mode}_{minimum_league}_train.pth"
    val_filename = f"{set_abbreviation}_{draft_mode}_{minimum_league}_val.pth"
    train_path = data_folder_training_set + train_filename
    val_path = data_folder_training_set + val_filename
    if overwrite == False and os.path.exists(train_path) and os.path.exists(val_path):
        print("Training and validation sets already exist. Skipping. Set overwrite=True to reprocess.")
        return

    # Validate input file. 
    csv_path = f"{data_folder_17lands}draft_data_public.{set_abbreviation}.{draft_mode}Draft.csv.gz"
    if os.path.exists(csv_path):
        print(f"Using input file {csv_path}")
    else:
        print(f"Did not find file {csv_path}")

    # Implement league filter. 
    leagues = ["bronze", "silver", "gold", "platinum", "diamond", "mythic"]
    if draft_mode == "Premier":
        if minimum_league not in leagues:
            raise Exception(f"Set minimum league to one of the following leagues: {leagues}")
        included_leagues = leagues[leagues.index(minimum_league):]
    else:
        minimum_league = "" # Ignored for Trad draft

    # Initialization on a single chunk. 
    for draft_chunk in pd.read_csv(csv_path, chunksize=10000, compression='gzip'):

        # Remove basics. 
        draft_chunk = remove_basics(draft_chunk)
        
        # Get date 1 week after start of draft (assumes drafts sorted by draft time). 
        first_date_str = draft_chunk["draft_time"].min()
        first_date_obj = datetime.strptime(first_date_str, "%Y-%m-%d %H:%M:%S")
        min_date_obj = first_date_obj + timedelta(days=omit_first_days)
        min_date_str = min_date_obj.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get cardnames and ids. 
        pack_cols = [col for col in draft_chunk.columns if col.startswith("pack_card")]
        cardnames = [col[10:] for col in sorted(pack_cols)]
        class_to_index = {cls: idx for idx, cls in enumerate(cardnames)}
        
        print("Completed initialization.")
        break

    # Process full input csv in chunks. 
    pack_chunks, pool_chunks, pick_chunks = [], [], []
    chunk_size = 100000 
    t0 = time.time()
    for i, draft_chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, compression='gzip')):

        # Remove basics. 
        draft_chunk = remove_basics(draft_chunk)
        
        # Filtering. 
        draft_chunk = draft_chunk[draft_chunk["draft_time"] > min_date_str] # Filter out first week. 
        draft_chunk = draft_chunk[draft_chunk["rank"].isin(included_leagues)] # Only highly ranked. 
        
        # Extract packs. 
        pack_chunk = draft_chunk[sorted(pack_cols)].astype(bool)

        # Extract pools. 
        pool_cols = [col for col in draft_chunk.columns if col.startswith("pool_")]
        pool_chunk = draft_chunk[sorted(pool_cols)].astype(np.uint8)
        
        # Extract picks. 
        pick_chunk = np.zeros((len(draft_chunk), len(cardnames)), dtype=bool)
        for j, item in enumerate(draft_chunk["pick"]):
            pick_chunk[j, class_to_index[item]] = True

        # Append data (consider multiple files for memory efficiency). 
        pick_chunks.append(pick_chunk)
        pack_chunks.append(pack_chunk)
        pool_chunks.append(pool_chunk)
        
        if i % 10 == 0:
            print(f"Loaded {chunk_size * i} picks, t=", round(time.time()-t0, 1), "s")

    print("Done loaded draft data.")

    # Concatenate all chunks into a single Dataframe
    picks = np.vstack(pick_chunks)
    packs = pd.concat(pack_chunks, ignore_index=True)
    pools = pd.concat(pool_chunks, ignore_index=True)

    # Create train and validation datasets. 
    tsize = round(len(pools) * train_fraction)
    pick_train_dataset = PickDataset(pools[:tsize].values, packs[:tsize].values, picks[:tsize], cardnames)
    pick_val_dataset = PickDataset(pools[tsize:].values, packs[tsize:].values, picks[tsize:], cardnames)

    # Serialize updated datasets. 
    if not os.path.exists(data_folder_training_set):
        os.makedirs(data_folder_training_set)

    # Write datasets. 
    torch.save(pick_train_dataset, train_path)
    print(f"Saved training set to {train_path}")

    torch.save(pick_val_dataset, val_path)
    print(f"Saved validation set to {train_path}")