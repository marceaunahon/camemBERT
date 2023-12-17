import conllu
import torch
def read_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
        sentences = conllu.parse(data) 
    return sentences

def extract_information(sentence):
    words = [token['form'] for token in sentence]
    pos_tags = [token['upostag'] for token in sentence]
    head_indices = [token['head'] for token in sentence]
    dependencies = [token['deprel'] for token in sentence]

    return words, pos_tags, head_indices, dependencies

def word2sentence(words,cond = False):
    l = len(words)
    out = ''
    for idx, w in enumerate(words):
        if idx < (l-1):
            if words[idx+1].isalpha() or cond:
                out += w + ' '
            else:
                out += w
        else :
            out += w
    return out

def preprocess_conllu(file_path,cond = False):

    ud_treebank = read_conllu(file_path)
    sentences,words_acc, pos_tags_acc, head_indices_acc, dependencies_acc =[], [], [] , [], []
    # Example: Embed Sentences and Compare
    for sentence in ud_treebank:
        # Extract words, POS tags, head indices, dependencies
        words_acc.append([token['form'] for token in sentence])
        pos_tags_acc.append([token['upostag'] for token in sentence])
        head_indices_acc.append([token['head'] for token in sentence])
        dependencies_acc.append([token['deprel'] for token in sentence])
        #Extract sentences
        sentences.append(word2sentence(words_acc[-1],cond))
    return sentences,words_acc, pos_tags_acc, head_indices_acc, dependencies_acc

def make_prediction(sentence,pos):
    labels = [l['entity'] for l in pos(sentence)]
    return list(zip(sentence.split(" "), labels))

def specific_to_general_pos(tag):
    specific_to_general = {
        "PREP": "ADP",
        "AUX": "AUX",
        "ADV": "ADV",
        "COSUB": "SCONJ",
        "COCO": "CCONJ",
        "PART": "PART",
        "PRON": "PRON",
        "PDEMMS": "PRON",
        "PDEMMP": "PRON",
        "PDEMFS": "PRON",
        "PDEMFP": "PRON",
        "PINDMS": "PRON",
        "PINDMP": "PRON",
        "PINDFS": "PRON",
        "PINDFP": "PRON",
        "PROPN": "PROPN",
        "XFAMIL": "PROPN",
        "NUM": "NUM",
        "DINTMS": "NUM",
        "DINTFS": "NUM",
        "PPOBJMS": "PRON",
        "PPOBJMP": "PRON",
        "PPOBJFS": "PRON",
        "PPOBJFP": "PRON",
        "PPER1S": "PRON",
        "PPER2S": "PRON",
        "PPER3MS": "PRON",
        "PPER3MP": "PRON",
        "PPER3FS": "PRON",
        "PPER3FP": "PRON",
        "PREFS": "PRON",
        "PREF": "PRON",
        "PREFP": "PRON",
        "VERB": "VERB",
        "VPPMS": "VERB",
        "VPPMP": "VERB",
        "VPPFS": "VERB",
        "VPPFP": "VERB",
        "DET": "DET",
        "DETMS": "DET",
        "DETFS": "DET",
        "ADJ": "ADJ",
        "ADJMS": "ADJ",
        "ADJMP": "ADJ",
        "ADJFS": "ADJ",
        "ADJFP": "ADJ",
        "NOUN": "NOUN",
        "NMS": "NOUN",
        "NMP": "NOUN",
        "NFS": "NOUN",
        "NFP": "NOUN",
        "PREL": "SCONJ",
        "PRELMS": "SCONJ",
        "PRELMP": "SCONJ",
        "PRELFS": "SCONJ",
        "PRELFP": "SCONJ",
        "INTJ": "INTJ",
        "CHIF": "NUM",
        "SYM": "SYM",
        "YPFOR": "PUNCT",
        "PUNCT": "PUNCT",
        "MOTINC": "X",
        "X": "X",
    }

    return specific_to_general.get(tag, tag)


from transformers import CamembertTokenizer, CamembertForTokenClassification, CamembertModel
import conllu
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
model_name = "camembert-base"

# Load the token classification model for POS tagging
tokenizer = CamembertTokenizer.from_pretrained(model_name)
#model = CamembertForTokenClassification.from_pretrained(model_name)
model  = CamembertModel.from_pretrained("camembert-base")
#the number of possible tags represent the output dim of the layer we will add

class CustomDataSet(Dataset):
    def __init__(self,sentences,pos_tags,tokenizer,max_len=64):
        self.sentences = sentences
        self.pos_tags = pos_tags
        self.tokenizer = tokenizer
        self.max_length = max_len
        self.pos_tag_labels = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'SCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SYM', 'VERB', 'X','_']

        # Create a mapping from label to index
        self.tag2index = {tag: idx for idx, tag in enumerate(self.pos_tag_labels)}


    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        pos_tag = self.pos_tags[idx]

        # Tokenize and encode the sentence
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Convert pos_tag to tensor (adjust based on your data type)
        pos_tag_tensor = torch.tensor([self.tag2index[tag] for tag in pos_tag])

        return {
            'inputs' : encoding,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pos_tags': pos_tag_tensor
        }


class CamembertPOS(torch.nn.Module):
    def __init__(self, model):
        super(CamembertPOS, self).__init__()
        nb_tags = 18
        pos_layer = torch.nn.Linear(model.config.hidden_size, nb_tags)
        self.camembert_pos = model
        self.pos_layer = pos_layer

    def forward(self, inputs):
        outputs = self.camembert_pos(**inputs)
        # Use the output embeddings for POS tagging
        pos_tag_logits = self.pos_layer(outputs.last_hidden_state)
        return pos_tag_logits



camembertPOS = CamembertPOS(model)

sentences,all_words, pos_tags_sentence, head_indices, dependencies = preprocess_conllu('fr_gsd-ud-train.conllu',True)
dataset = CustomDataSet(sentences, pos_tags_sentence, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(camembertPOS.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
num_epochs = 10
best_loss = float('inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
camembertPOS.to(device)
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(dataset, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs = batch['inputs']
        inputs = inputs.to(device)
        pos_tags = batch['pos_tags']

        tags = torch.nn.functional.pad(pos_tags, (0, 64 - len(pos_tags)), value=-100)

        optimizer.zero_grad()
        pos_tag_logits = camembertPOS(inputs)

        # Calculate loss
        loss = criterion(pos_tag_logits.view(-1, pos_tag_logits.shape[-1]), tags.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(camembertPOS.state_dict(), f"camembertPOS{avg_loss}.pth")
