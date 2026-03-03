import numpy as np
import json
import yaml

def build_vocab(train_df):
    # collect every unique character from all training texts
    all_text = " ".join(train_df['text'].tolist())
    vocab = sorted(set(all_text))
    
    # add special tokens at the front
    # PAD = index 0 → used to fill short sequences
    # UNK = index 1 → used for characters never seen in training
    vocab = ['<PAD>', '<UNK>'] + vocab
    
    # create two mappings
    char2idx = {ch: idx for idx, ch in enumerate(vocab)}
    idx2char = {idx: ch for idx, ch in enumerate(vocab)}
    
    return char2idx, idx2char

def build_label_encoder(train_df):
    # get all unique language codes sorted alphabetically
    languages = sorted(train_df['labels'].unique())
    
    # map each language to an integer
    # e.g {'ar':0, 'bg':1, 'de':2, 'el':3, 'en':4 ...}
    lang2idx = {lang: idx for idx, lang in enumerate(languages)}
    idx2lang = {idx: lang for idx, lang in enumerate(languages)}
    
    return lang2idx, idx2lang

def encode_text(text, char2idx, max_len):
    # convert each character to its index
    # if character not in vocab → use UNK index (1)
    encoded = [char2idx.get(ch, 1) for ch in text[:max_len]]
    
    # pad with zeros if text shorter than max_len
    padded = encoded + [0] * (max_len - len(encoded))
    
    return padded

def encode_dataset(df, char2idx, lang2idx, max_len):
    # encode every text sample in the dataframe
    X = np.array([encode_text(t, char2idx, max_len) for t in df['text']])
    
    # encode every label
    y = np.array([lang2idx[l] for l in df['labels']])
    
    return X, y

def save_vocab(char2idx, lang2idx, idx2lang, path):
    # save vocab to disk so we can use it later during inference
    # without this, we can't decode predictions back to language names
    vocab_data = {
        'char2idx': char2idx,
        'lang2idx': lang2idx,
        'idx2lang': idx2lang
    }
    with open(path, 'w') as f:
        json.dump(vocab_data, f)
    print(f"Vocab saved to {path}")

def load_vocab(path):
    with open(path, 'r') as f:
        vocab_data = json.load(f)
    return vocab_data['char2idx'], vocab_data['lang2idx'], vocab_data['idx2lang']