from typing import Any
from datasets import load_dataset
import random
import sentencepiece as spm
import os
import torch
from torch.utils.data import Dataset
import numpy as np


class Oscar(Dataset):

    def __init__(self, language="fr", split="train", init_tokenizer=True, padding=True, max_length=200, masked_ratio=0.15, random_ratio=0.1, keep_ratio=0.1):
        """ initialization of  the object Oscar with the language and the split
        charge the dataset with the language and the split

        Args:
            language (str, optional): language of the dataset. Defaults to 'fr'.
            split (str, optional): split of the dataset. Defaults to 'train'.

        Raises:
            Exception: if the split is not in ['train', 'test', 'validation']

        Returns:
            None
        """
        self.language = language
        self.split = split
        self.padding = padding
        self.max_length = max_length
        self.masked_ratio = masked_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio
        self.dataset = load_dataset("nthngdy/oscar-mini",
                                    use_auth_token="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",  # required
                                    language=language,
                                    split=split)  # optional, but the dataset only has a train split
        self.tokenizer = None

        # initialize the tokenizer
        if init_tokenizer:
            self.init_tokenizer()

    def __getitem__(self, index: int) -> Any:
        """
        Get the element at the index of the dataset with tokenization and 15% of possible masking (WWM)

        Args:
            index (int): index of the element

        Returns:
            tensor_masked_text, tensor_tokenized_text: two tensors of the masked text and the tokenized text (ids)
        """
        # check if the tokenizer has been initialized
        if self.tokenizer is None:
            raise Exception(
                "You must initialize the tokenizer before tokenizing the dataset.")
        #  tokenize the text
        tokenized_text = self.tokenize_text(self.dataset[index]["text"])
        masked_text = self.mask_text(tokenized_text)

        tensor_masked_text = torch.tensor(masked_text)
        tensor_tokenized_text = torch.tensor(tokenized_text)

        return tensor_masked_text, tensor_tokenized_text

    def tokenize_text(self, text: str) -> Any:
        """
        Tokenize a text

        Args:
            text (str): text to tokenize

        Returns:
            tokenized_text (list): int list of the tokens ids
        """
        #  use the tokenizer to tokenize the text
        tokenized_text = self.tokenizer.encode_as_ids(text)
        #  add the special tokens at the beginning and the end of the sentence
        tokenized_text = [self.start_token_id] + \
            tokenized_text + [self.end_token_id]
        if self.padding:
            if len(tokenized_text) > self.max_length:
                #  if the sentence is too long, cut it
                tokenized_text = tokenized_text[:self.max_length]
            #  pad the sentence if it is too short
            else:
                tokenized_text += [self.pad_tokken_id] * \
                    (self.max_length - len(tokenized_text))
        return tokenized_text

    def tokens_to_ids(self, tokens: list) -> Any:
        """
        Convert the tokens to their ids in the vocabulary

        Args:
            tokens (list): str list of tokens

        Returns:
            ids (list): int list of tokens ids
        """
        #  convert the tokens to their ids in the vocabulary
        return [self.vocab[token] if token in self.vocab else self.vocab["<unk>"] for token in tokens]

    def mask_text(self, tokens: list) -> Any:
        """
        Mask 15% of the tokens

        Args:
            tokens (list): list of tokens

        Returns:
            masked_tokens (list): list of ids of the masked tokens
        """
        #  create a copy of the tokens
        masked_tokens = tokens.copy()
        # mask 15% of the tokens
        for i in range(len(tokens)):
            if random.random() < self.masked_ratio:
                # randomize 10% of the 15% of the tokens
                if random.random() < self.random_ratio:
                    # replace the token with a random token from the vocabulary
                    masked_tokens[i] = random.randint(0, self.get_vocab_size()-1)
                elif random.random() < self.keep_ratio:
                    # keep the token
                    masked_tokens[i] = tokens[i]
                else:
                    # replace the token with <mask>
                    masked_tokens[i] = self.mask_token_id
            else:
                # keep the token
                masked_tokens[i] = tokens[i]
        return masked_tokens

    # return the length of the dataset
    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            length (int): length of the dataset
        """
        return len(self.dataset)

    def get_raw_text(self, index: int) -> Any:
        """
        Get the raw text element at the index of the dataset (without tokenization)

        Args:
            index (int): index of the element

        Returns:
            element (tuple): {id: id of the element, text: text of the element}
        """
        return self.dataset[index]

    #  return a random sample from the dataset
    def get_random_sample(self) -> Any:
        """
        Get a random sample from the dataset

        Returns:
            random_sample (tuple): {id: id of the element, text: text of the element}
        """
        return random.choice(self.dataset)

    # search the dataset with a keyword and return the results as a list
    def search_by_keyword(self, keyword: str) -> Any:
        """
        Search the dataset with a keyword and return the results as a list

        Args:
            keyword (str): keyword to search

        Returns:
            results (list): list of the results
        """
        # Use a list comprehension to filter examples
        results = [
            example for example in self.dataset if keyword in example["text"]]
        return results

    def write_to_file(self, filename):
        """
        Write the dataset to a texte file (used to train the tokenizer)

        Args:
            filename (str): name of the file to write

        Returns:
            None        
        """
        # Write the dataset to a texte file (used to train the tokenizer)
        with open(filename, 'w', encoding='utf-8') as f:
            for example in self.dataset:
                f.write(example["text"] + '\n')

    def init_tokenizer(self, vocab_size=32000, txt_file="Tokenization/oscar_text.txt", model_prefix="Tokenization/oscar_tokenizer"):
        """
        Initialize the tokenizer

        Args:
            vocab_size (int, optional): size of the vocabulary. Defaults to 32000.
            txt_file (str, optional): name of the file to write. Defaults to "Tokenization/oscar_text.txt".
            model_prefix (str, optional): name of the model. Defaults to "Tokenization/oscar_tokenizer".

        Returns:
            None
        """

        #  check if the file exists
        if not os.path.exists(txt_file):
            #  if not create it and write the dataset in it
            print(f"Creating file {txt_file}...")
            self.write_to_file(txt_file)
            print(f"File {txt_file} has been created.")
        else:
            print(f"File {txt_file} found.")

        #  check if the model already exists
        if not os.path.exists(f'{model_prefix}.model'):
            print(f"Creating model {model_prefix}...")
            # if not train the model with the file and the set vocabulary size
            spm.SentencePieceTrainer.train(
                f'--input={txt_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --user_defined_symbols=<pad>,<mask>')

        # Load the trained model
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f'{model_prefix}.model')

        #  create the vocabulary
        self.vocab = {self.tokenizer.id_to_piece(
            id): id for id in range(self.tokenizer.get_piece_size())}

        # get the ids of the special tokens
        self.start_token_id = self.vocab["<s>"]
        self.end_token_id = self.vocab["</s>"]
        self.mask_token_id = self.vocab["<mask>"]
        self.pad_tokken_id = self.vocab["<pad>"]

    def ids_to_tokens(self, ids: list) -> Any:
        """
        Convert the ids to their tokens in the vocabulary

        Args:
            ids (list): int list of tokens ids

        Returns:
            tokens (list): str list of tokens
        """
        #  convert the ids to their tokens in the vocabulary
        return [self.tokenizer.id_to_piece(id) for id in ids]

    def get_vocab(self) -> Any:
        """
        Get the vocabulary of the tokenizer

        Returns:
            vocab (dict): vocabulary of the tokenizer
        """
        return self.vocab

    def get_vocab_size(self) -> int:
        """
        Get the size of the vocabulary of the tokenizer

        Returns:
            size (int): size of the vocabulary of the tokenizer
        """
        return len(self.vocab)
