from datasets import load_dataset

class Oscar():
    def __init__(self, language="fr", streaming=True, split="train"):
        self.language = language
        self.streaming = streaming
        self.split = split
        self.dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        use_auth_token="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", # required
                        language="fr", 
                        streaming=True, # optional
                        split="train") # optional, but the dataset only has a train split