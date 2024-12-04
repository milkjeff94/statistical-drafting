import pandas as pd
pd.options.display.max_rows = 1000
import torch
from typing import List, Union
import warnings
warnings.filterwarnings("ignore")

import statisticaldrafting as sd

class DraftModel:
    def __init__(self, set: str="FDN"): 
        # Get data from set. 
        self.set = set
        self.pick_table = pd.read_csv(f"../data/cards/{self.set}.csv") # will be sorted. 
        self.cardnames = self.pick_table["name"].tolist()

        # Load model. 
        model_path = f"../data/models/{set}.pt"
        self.network = sd.DraftMLP(cardnames=self.cardnames, hidden_dims=[300, 300, 300]) # TODO: remove hidden_dims argument. 
        self.network.load_state_dict(torch.load(model_path))

        # Assign p1p1 ratings. 
        self.pick_table["p1p1_rating"] = self.get_card_ratings([])
        self.pick_table["synergy"] = [0.0] * len(self.pick_table)
        self.pick_table["rating"] = [0.0] * len(self.pick_table)

    def get_card_ratings(self, collection: List[Union[str, int]]) -> pd.Series:
        """ Get card ratings (0-150) for input collection."""
        # Create collection vector. 
        collection_vector = torch.zeros([1, len(self.cardnames)])        
        for card in collection: 
            if type(card) is str:
                # Validate cardname. 
                if card not in self.cardnames:
                    raise Exception(f"{card} not in set. Please correct cardname.")
                card_index = self.cardnames.index(card)
            else:
                # card_id is input. 
                card_index = card
            collection_vector[0, card_index] += 1
            
        # Get raw card scores. 
        self.network.eval()
        with torch.no_grad():
            card_scores = self.network(collection_vector, torch.ones(len(self.cardnames)))

        # Return card ratings (0, 150).
        card_scores = card_scores.reshape(-1) # Ensure correct shape. 
        min_score = min(card_scores).item()
        max_score = max(card_scores).item()
        card_ratings = [150 * (cs - min_score) / (max_score - min_score) for cs in card_scores.tolist()]
        rounded_card_ratings = pd.Series([round(cr, 1) for cr in card_ratings], index=[i for i in range(len(self.cardnames))], name="rating")
        return rounded_card_ratings
    
    def get_pick_order(self, collection: List[Union[str, int]]) -> pd.DataFrame:
        """Returns pick order table for collection."""
        # Get card ratings. 
        card_ratings = self.get_card_ratings(collection)

        # Update ratings. 
        self.pick_table = self.pick_table.drop("rating", axis=1)
        self.pick_table = self.pick_table.join(card_ratings)
        self.pick_table["synergy"] = self.pick_table["rating"] - self.pick_table["p1p1_rating"]

        # Return sorted values. 
        self.pick_table = self.pick_table.sort_values(by="rating", ascending=False)
        return self.pick_table
