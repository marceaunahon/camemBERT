import conllu
from transformers import CamembertTokenizer, CamembertForTokenClassification, TokenClassificationPipeline
import numpy as np
from tqdm import tqdm


#some utility functions
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

def preprocess_conllu(file_path):

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
        sentences.append(word2sentence(words_acc[-1],True))
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

def evaluate_dependency_parsing(model, tokenizer, sentences, gold_dependencies):
    correct, total = 0, 0

    for sent, gold_deps in tqdm(zip(sentences, gold_dependencies)):
        inputs = tokenizer(sent, return_tensors="pt")
        outputs = model(**inputs)
        predicted_dependencies = outputs.logits.argmax(dim=-1).squeeze().tolist()

        for pred_dep, gold_dep in zip(predicted_dependencies, gold_deps):
            if pred_dep == gold_dep:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":

    model_name = 'qanastek/pos-french-camembert'
    data_path = 'C:/Users/User/Documents/GitHub/camemBERT/Eval/fr_gsd-ud-dev.conllu'

    tokenizer = CamembertTokenizer.from_pretrained(model_name)
    model = CamembertForTokenClassification.from_pretrained(model_name)
    pos = TokenClassificationPipeline(model=model, tokenizer=tokenizer)


    sentences,all_words_acc, pos_tags_acc, head_indices_acc, dependencies_acc = preprocess_conllu(data_path)
    
    #accuracy_depency_parcing = evaluate_dependency_parsing(model, tokenizer, sentences, dependencies_acc)
    
    same, truth = 0, 0
    for acc_sentence_words,acc_sentence_pos_tags in tqdm(zip(all_words_acc,pos_tags_acc)):
        s,t = 0,0
        for acc_word,acc_pos_tag in zip(acc_sentence_words,acc_sentence_pos_tags):
            tag = specific_to_general_pos( make_prediction(acc_word,pos)[0][1] )
            res = 1 if acc_pos_tag == tag else 0
            s += res
            t += 1
        same += s
        truth += t
    accuracy = same/truth

    with open("accuracy_log.txt", "w") as file:
        log = f"Model : {model_name}\n"
        log += f"Database path (HuggingFace) : {data_path}"
        log += f"Accuracy : {accuracy*100}%"
        file.write(log)

