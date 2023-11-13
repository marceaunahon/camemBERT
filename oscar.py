from typing import Any
from datasets import load_dataset
import random

class Oscar():
    def __init__(self, language="fr", split="train"):
        self.language = language
        self.split = split
        self.dataset = load_dataset("nthngdy/oscar-mini",
                        use_auth_token="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", # required
                        language=language,
                        split=split) # optional, but the dataset only has a train split

    # return the dataset
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
    
    # return the length of the dataset
    def __len__(self) -> int:
        return len(self.dataset)

    # return a random sample from the dataset
    def get_random_sample(self) -> Any:
        return random.choice(self.dataset)
    
    def search_by_keyword(self, keyword: str) -> Any:
        # Convert keyword to lowercase for case-insensitive search
        keyword_lower = keyword.lower()

        # Use a set to avoid duplicates
        results = []

        for d in self.dataset:
            # Convert text to lowercase for case-insensitive search
            text_lower = d["text"].lower()

            # Check if the keyword is present in the lowercase text
            if keyword_lower in text_lower:
                results.append(d)

        return results