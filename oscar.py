from typing import Any
from datasets import load_dataset
import random
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter


class Oscar():
    
    def __init__(self, language="fr", split="train"):
        """ initialization of  the object Oscar with the language and the split
        charge the dataset with the language and the split """
        self.language = language
        self.split = split
        self.dataset = load_dataset("nthngdy/oscar-mini",
                        use_auth_token="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", # required
                        language=language,
                        split=split) # optional, but the dataset only has a train split
        self.tokenized_dataset = self.tokenize_dataset()

    # return the element at the index of the dataset
    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]
    
    # return the length of the dataset
    def __len__(self) -> int:
        return len(self.dataset)

    # return a random sample from the dataset
    def get_random_sample(self) -> Any:
        return random.choice(self.dataset)
    
    # search the dataset with a keyword and return the results as a list
    def search_by_keyword(self, keyword: str) -> Any:
        # Use a list comprehension to filter examples
        results = [example for example in self.dataset if keyword in example["text"]]

        return results

    # tokenize the text with the WWM method 
    """     def tokenize_text(self, text: str) -> Any:
        # Tokenize the sentence
        words = re.findall(r'\b\w+\b', text)

        # Apply whole-word masking (WWM)
        masked_sentence = ' '.join(['[MASK]' if random.random() < 0.15 else word for word in words])

        return masked_sentence """

    def tokenize_text(self, text: str) -> Any:
        # Tokenize the sentence
        tokens = word_tokenize(text)

        return tokens
    
    def tokenize_dataset(self) -> Any:
        try:
            # Try to find the 'punkt' package
            nltk.data.find('tokenizers/punkt')
            print("'punkt' is already installed.")
        except LookupError:
            # If not found, download it
            nltk.download('punkt')
            print("'punkt' has been downloaded.")
        # Tokenize the dataset
        # create an empty list of the same size as the dataset
        tokenized_dataset = [None] * len(self.dataset)
        for i,sentence in enumerate(self.dataset):
            if i % 1000 == 0:
                print(f"\rTokenizing {i}/{len(self.dataset)}", end="")
            tokenized_dataset[i] = self.tokenize_text(sentence["text"])

        print(f"\rTokenized {len(self.dataset)}/{len(self.dataset)} sentences.")

        return tokenized_dataset
    
    def get_vocab(self) -> Any:
        vocab = Counter()

        for sentence in self.tokenized_dataset:
            vocab.update(sentence)

        return vocab
