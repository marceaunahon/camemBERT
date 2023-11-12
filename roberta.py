import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class Roberta:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(model_name)

    def predict(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0).detach().numpy()
        return self.model.config.id2label[scores.argmax()]
    