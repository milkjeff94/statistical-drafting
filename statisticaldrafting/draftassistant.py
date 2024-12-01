from collections import Counter
import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import torch
from typing import List
import warnings
warnings.filterwarnings("ignore")

import statisticaldrafting as sd

# Utility functions. 
def get_card_scores(collection_list: List[str],
                    network: torch.nn.Module,
                    cardnames: List[str]):
    """ Get card scores (unnormalized) for current network. Used for visualization """
    # Get collection vector
    collection_vector = torch.zeros([1, len(cardnames)])
    cnt = Counter(collection_list)
    for card in cnt:
        
        # Validate cardname. 
        if card not in cardnames:
            raise Exception(f"{card} not in set. Please correct cardname.")

        # Add to collection vector. 
        card_index = cardnames.index(card)
        collection_vector[0, card_index] = cnt[card]
        
    # Get card and collection embeddings. 
    network.eval()
    with torch.no_grad():
        card_scores = network(collection_vector, torch.ones(len(cardnames)))
    return card_scores

def get_card_ratings(card_scores, top_rating=150):
    """
    Produces a set of normalized card ratings from raw card scores. 
    """
    card_scores = card_scores.reshape(-1) # Ensure correct shape. 
    min_score = min(card_scores).item()
    max_score = max(card_scores).item()
    card_ratings = [top_rating * (cs - min_score) / (max_score - min_score) for cs in card_scores.tolist()]
    return [round(cr, 1) for cr in card_ratings]

class DraftAssistant:
    def __init__(self, set="FDN", picks_shown: int = 25):
        
        # Get data from set. 
        self.set = set
        self.pick_table = pd.read_csv(f"../data/cards/{self.set}.csv")
        self.pick_table["rating"] = [1] * len(self.pick_table)
        self.cardnames = self.pick_table["name"].tolist()
        self.collection = pd.DataFrame(columns=self.pick_table.columns)
        self.picks_shown = picks_shown

        # Load model. 
        model_path = f"../data/models/{self.set}.pt"
        self.network = sd.DraftMLP(cardnames=self.pick_table["name"].tolist(), hidden_dims=[300, 300, 300])
        self.network.load_state_dict(torch.load(model_path))

        # initialize notebook draft assistant. 
        self.rarity_options = ["All", "common", "uncommon", "common+uncommon", "rare", "mythic", "special"]
        self.color_options = ["All", "W", "G", "U", "R", "B", "Multicolor", "Colorless"]
        self.rarity_filter = widgets.Dropdown(
            options=self.rarity_options,
            value="rare",
            description="Rarity:",
        )
        self.color_filter = widgets.Dropdown(
            options=self.color_options,
            value="All",
            description="Color:",
        )
        self.rarity_filter.observe(lambda change: self.update_table(), names='value')
        self.color_filter.observe(lambda change: self.update_table(), names='value')

    def update_table(self):
        """Re-render the pick table and collection."""
        clear_output(wait=True)
        self.draft()

    def make_pick(self, card):
        """Add card to the collection and update tables."""
        # global collection
        self.collection = pd.concat([self.collection, pd.DataFrame([card])], ignore_index=True)
        self.update_table()

    def remove_card(self, index):
        """Remove a card from the collection by index and update tables."""
        # global collection
        self.collection = self.collection.drop(index).reset_index(drop=True)
        self.update_table()
        
    # Function to reset the collection
    def reset_collection(self, change=None):
        """Reset the collection (clear all cards)."""
        # global collection
        self.collection = pd.DataFrame(columns=self.pick_table.columns)  # Empty collection
        self.update_table()

    def draft(self):
        """
        Display pick table and collection with interactive buttons.
        """
        collection_list = [n for n in self.collection["name"]]
        card_scores = get_card_scores(collection_list, self.network, self.cardnames)
        ratings = get_card_ratings(card_scores) 
        self.pick_table["rating"] = ratings # Use percentiles for now. 
        
        if "p1p1_rating" not in self.pick_table.columns:
            p1p1_scores = get_card_scores([], self.network, self.cardnames)
            p1p1_ratings = get_card_ratings(p1p1_scores)
            self.pick_table["p1p1_rating"] = p1p1_ratings
            
        self.pick_table["synergy"] = (self.pick_table["rating"] - self.pick_table["p1p1_rating"]).round(1)
            
        
        # Hide scores in collection table. 
        self.collection["rating"] = [""] * len(self.collection)
        
        # Apply filters to the pick table
        self.filtered_table = self.pick_table.copy()

        # If the rarity filter is "All", exclude cards with "Basic" rarity
        if self.rarity_filter.value == "All":
            pass 
        elif self.rarity_filter.value == "common+uncommon":
            self.filtered_table = self.filtered_table[self.filtered_table['rarity'].isin(["common", "uncommon"])]
        else:
            self.filtered_table = self.filtered_table[self.filtered_table['rarity'] == self.rarity_filter.value]
        
        if self.color_filter.value != "All":
            self.filtered_table = self.filtered_table[self.filtered_table['color_identity'] == self.color_filter.value]

        # Sort the filtered pick table by rating (descending order)
        self.filtered_table = self.filtered_table.sort_values(by="rating", ascending=False)
        
        # Add the "New Draft" button to reset the collection
        new_draft_button = widgets.Button(description="New Draft", button_style="warning")
        new_draft_button.on_click(self.reset_collection)
        display(new_draft_button)
        
        # Display the filters
        filter_box = widgets.HBox([self.rarity_filter, self.color_filter])
        display(filter_box)

        # Get the maximum length of card names to align them
        max_name_length = self.filtered_table['name'].apply(len).max()
        max_name_length = max(max_name_length, 20)  # Width of name column
        
        # Formatting function to align columns and display as text
        def format_pick_row(row):
            return f"{row['name']:<{max_name_length}} | {row['rarity']:<9} | {row['color_identity']:<12} | {row['synergy']:>+7}| {row['rating']:>6}"
        
        def format_collection_row(row):
            return f"{row['name']:<{max_name_length}} | {row['rarity']:<9} | {row['color_identity']:<12} | {'':>7}| {row['rating']:>6}"


        # Display the filtered pick table with buttons
        print(f'{" Card Name":<{max_name_length}} | {"Rarity":<9} | {"Color":<12} | {"Synergy":>7}| {"Rating":>6}')
        for row_count, (index, row) in enumerate(self.filtered_table.iterrows()):
            row_widget = widgets.Output()
            with row_widget:
                print(format_pick_row(row))
            pick_button = widgets.Button(description=f"Pick: {row['name']}", button_style="success")
            pick_button.on_click(lambda btn, card=row: self.make_pick(card.to_dict()))
            display(widgets.HBox([row_widget, pick_button]))

            if row_count >= (self.picks_shown - 1):
                break

        # Display the collection with remove buttons (same format as pick table)
        print("\nCollection:")
        if not self.collection.empty:
            collection_widget = widgets.Output()
            with collection_widget:
                # Use the same format_row for collection as for pick table
                for _, row in self.collection.iterrows():
                    print(format_collection_row(row))

            remove_buttons = []
            for idx, row in self.collection.iterrows():
                remove_button = widgets.Button(description=f"Remove: {row['name']}", button_style="danger")
                remove_button.on_click(lambda btn, index=idx: self.remove_card(index))

                # Align text and remove button together in the same layout
                row_widget = widgets.Output()
                with row_widget:
                    print(format_collection_row(row))
                
                remove_buttons.append(widgets.HBox([row_widget, remove_button]))

            remove_buttons_box = widgets.VBox(remove_buttons)
            display(remove_buttons_box)
        else:
            print("Collection is empty.")

# Run the draft assistant. 
# DraftAssistant(set="FDN", picks_shown=10).draft()