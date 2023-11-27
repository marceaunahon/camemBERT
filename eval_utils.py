import conllu

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

def preprocess_conllu(file_path):
    sentences = read_conllu(file_path)

    all_words, all_pos_tags, all_head_indices, all_dependencies = [], [], [], []

    for sentence in sentences:
        words, pos_tags, head_indices, dependencies = extract_information(sentence)
        all_words.append(words)
        all_pos_tags.append(pos_tags)
        all_head_indices.append(head_indices)
        all_dependencies.append(dependencies)

    return all_words, all_pos_tags, all_head_indices, all_dependencies

