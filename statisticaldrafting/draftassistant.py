from collections import Counter
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import torch
from typing import List, Union
import warnings
warnings.filterwarnings("ignore")

import statisticaldrafting as sd

# Utility functions. 
def get_card_ratings(collection_list: List[Union[str, int]],
                    network: torch.nn.Module,
                    cardnames: List[str]):
    """ Get card ratings (0-150) for current network. Used for draft assistant. """
    # Get collection vector
    collection_vector = torch.zeros([1, len(cardnames)])
    
    # Support card_id and card_names. 
    for card in cardnames: 
        if type(card) is str:
            # Validate cardname. 
            if card not in cardnames:
                raise Exception(f"{card} not in set. Please correct cardname.")
            card_index = cardnames.index(card)
        else:
            # card_id is input. 
            card_index = card
        collection_vector[0, card_index] += 1
        
    # Get raw card scores. 
    network.eval()
    with torch.no_grad():
        card_scores = network(collection_vector, torch.ones(len(cardnames)))

    # Return card ratings (0, 150).
    card_scores = card_scores.reshape(-1) # Ensure correct shape. 
    min_score = min(card_scores).item()
    max_score = max(card_scores).item()
    card_ratings = [150 * (cs - min_score) / (max_score - min_score) for cs in card_scores.tolist()]
    return [round(cr, 1) for cr in card_ratings]
