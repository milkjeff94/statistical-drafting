import os
import pandas as pd
pd.options.display.max_rows = 1000
import torch
from typing import List, Union
import warnings
warnings.filterwarnings("ignore")

import statisticaldrafting as sd

class DraftModel:
    def __init__(self, set: str="FDN", draft_mode: str="Premier"): 
        # Get data from set. 
        self.set = set
        self.pick_table = pd.read_csv(f"../data/cards/{self.set}.csv") # will be sorted.
        self.cardnames = self.pick_table["name"].tolist()

        # Load model. 
        model_path = f"../data/models/{set}_{draft_mode}.pt"
        self.network = sd.DraftNet(cardnames=self.cardnames)
        self.network.load_state_dict(torch.load(model_path))

        # Assign p1p1 ratings. 
        self.pick_table["p1p1_rating"] = self.get_card_ratings([])
        self.pick_table["synergy"] = [0.0] * len(self.pick_table)
        self.pick_table["rating"] = [0.0] * len(self.pick_table)

    def get_collection_vector(self, collection: List[Union[str, int]]) -> torch.Tensor:
        """Get a collection vector from a list of cardnames and card_ids"""
        collection_vector = torch.zeros([1, len(self.cardnames)])        
        for card in collection: 
            if type(card) is str:
                # Validate cardname. 
                if card not in self.cardnames:
                    if card not in ["Plains", "Island", "Forest", "Swamp", "Mountain"]:
                        print(f"{card} not in draftable set.")
                    continue
                card_index = self.cardnames.index(card)
            else:
                # card_id is input. 
                card_index = card
            collection_vector[0, card_index] += 1
        return collection_vector


    def get_card_ratings(self, collection: List[Union[str, int]]) -> pd.Series:
        """ Get card ratings (0.00-5.50) for input collection."""
        # Create collection vector. 
        collection_vector = self.get_collection_vector(collection)
        
        # Get raw card scores. 
        self.network.eval()
        with torch.no_grad():
            card_scores = self.network(collection_vector, torch.ones(len(self.cardnames)))

        # Return card ratings (0-5).
        card_scores = card_scores.reshape(-1) # Ensure correct shape. 
        min_score = min(card_scores).item()
        max_score = max(card_scores).item()
        card_ratings = [5.5 * (cs - min_score) / (max_score - min_score) for cs in card_scores.tolist()]
        rounded_card_ratings = pd.Series([round(cr, 2) for cr in card_ratings], index=[i for i in range(len(self.cardnames))], name="rating")
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

    def get_deck_recommendation(self, pool_cards: List[Union[str, int]], starting_colors: str="") -> pd.DataFrame:
        """
        Constructs a deck from the cards in a pool. 

        Works by iteratively computing the "pick order" for cards in the deck. 

        The starting deck can be specified by suggesting colors in "WUBRG".

        WANRING: This is an experimental feature - expect to correct some picks in the deck. 
        """
        # Fixed parameters. 
        DECK_SIZE = 25 # Cards in deck - TODO: account for nonbasic lands precisely.
        NUM_ITERATION = 15
        
        # Initialize pool. 
        pool_cardnames = [pc if type(pc) is str else self.cardnames[pc] for pc in pool_cards]
        valid_cardnames = [cn for cn in pool_cardnames if cn in self.cardnames]
        df_pool = pd.DataFrame(valid_cardnames, columns=["name"])
        df_pool = pd.merge(df_pool, self.pick_table[["name", "color_identity"]], on='name', how='left')

        # Validate starting colors.
        if not set(starting_colors.upper()).issubset(set("WUBGR")):
            raise Exception("Starting colors should be a subset of WUBGR. Please adjust starting colors.")

        # Allow user to suggest colors for initial deck. 
        if len(starting_colors) > 0:
            colors_to_include = ["Colorless"] # TODO: account for multicolored correctly in the first iteration. 
            for color in starting_colors.upper():
                if color in "WUBGR":
                    colors_to_include.append(color)
            cur_deck = df_pool[df_pool["color_identity"].isin(colors_to_include)]["name"].tolist()
        else:
            cur_deck = []

        # Iteratively refine deck. 
        for i in range(NUM_ITERATION):
            deck_pickorder = self.get_pick_order(cur_deck)
            deck_cards_ranked = pd.merge(df_pool[["name"]], deck_pickorder[["name", "color_identity", "rating"]], on='name', how='left')
            deck_cards_ranked = deck_cards_ranked.sort_values(by="rating", ascending=False)
            cur_deck = [name for name in deck_cards_ranked["name"].head(DECK_SIZE)] 

        # Return final deck. 
        return deck_cards_ranked
    
def parse_cardnames(card_str, set="FDN"):
    """
    Get cardnames from arena deck export. 
    """
    rows = card_str.split("\n")
    pool_cardnames = []
    for row in rows:
        if f"({set})" in row:
            number = int(row.split(" ")[0])
            name = row.split("(")[0][2:-1]
            for _ in range(number):
                pool_cardnames.append(name)
    return pool_cardnames

def list_sets(model_path: str = "../data/models"):
    """
    List currently available sets.
    """
    draft_models = [dm.split(".")[0].split("_") for dm in os.listdir(model_path) if ".pt" in dm]
    draft_models = [dm.split(".")[0].split("_") for dm in os.listdir(model_path) if ".pt" in dm]
    sets = sorted(list(set([dm[0] for dm in draft_models])))
    return sets

# def evaluate_models(model_path: str = "../data/models"):
#     """
#     Get evaluation metrics for all models. Runs for some time. 
#     """
    