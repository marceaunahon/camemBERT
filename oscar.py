from typing import Any
from datasets import load_dataset
import random
import re


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
        # Use a list comprehension to filter examples
        results = [example for example in self.dataset if keyword in example["text"]]

        return results

    def tokenize_text(self, text: str) -> Any:
        # Tokenize the sentence
        words = re.findall(r'\b\w+\b', text)

        # Apply whole-word masking (WWM)
        masked_sentence = ' '.join(['[MASK]' if random.random() < 0.15 else word for word in words])

        return masked_sentence
