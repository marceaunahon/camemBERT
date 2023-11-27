from typing import Any
from datasets import load_dataset
import random
import sentencepiece as spm
import os


class Oscar():
    
    def __init__(self, language="fr", split="train", init_tokenizer=False):
        """ initialization of  the object Oscar with the language and the split
        charge the dataset with the language and the split """
        self.language = language
        self.split = split
        self.dataset = load_dataset("nthngdy/oscar-mini",
                        use_auth_token="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", # required
                        language=language,
                        split=split) # optional, but the dataset only has a train split
        
        if init_tokenizer:
            self.init_tokenizer()

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

    def write_to_file(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                f.write(example["text"] + '\n')

    def init_tokenizer(self, vocab_size=32000, txt_file="Tokenization/oscar_text.txt", model_prefix="Tokenization/oscar_tokenizer"):
        # check if the file exists
        if not os.path.exists(txt_file):
            print(f"Creating file {txt_file}...")
            self.write_to_file(txt_file)
            print(f"File {txt_file} has been created.")
        else:
            print(f"File {txt_file} found.")

        # check if the model exists
        if not os.path.exists(f'{model_prefix}.model'):
            print(f"Creating model {model_prefix}...")
            # Train the SentencePiece model
            spm.SentencePieceTrainer.train(f'--input={txt_file} --model_prefix={model_prefix} --vocab_size={vocab_size}')

        # Load the model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f'{model_prefix}.model')

        self.vocab = {self.tokenizer.id_to_piece(id): id for id in range(self.tokenizer.get_piece_size())}

    def tokenize_text(self, text: str) -> Any:
        return self.tokenizer.encode_as_pieces(text)
    
    def get_vocab(self) -> Any:
        return self.vocab



    # tokenize the text with the WWM method 
    """     def tokenize_text(self, text: str) -> Any:
        # Tokenize the sentence
        words = re.findall(r'\b\w+\b', text)

        # Apply whole-word masking (WWM)
        masked_sentence = ' '.join(['[MASK]' if random.random() < 0.15 else word for word in words])

        return masked_sentence """

    """    def tokenize_text(self, text: str) -> Any:
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

        return vocab """
    

