import torch
import torch.nn.functional as F
from src.data_loader import load_config
from src.preprocess import encode_text, load_vocab
from src.model import LangIDModel

def load_model(cfg, device):
    # load saved vocab
    char2idx, lang2idx, idx2lang = load_vocab(cfg['paths']['vocab_save'])
    
    # rebuild model architecture
    model = LangIDModel(
        vocab_size    = len(char2idx),
        embedding_dim = cfg['model']['embedding_dim'],
        gru_units     = cfg['model']['gru_units'],
        num_classes   = cfg['model']['num_classes'],
        dropout       = cfg['model']['dropout'],
        num_layers    = cfg['model']['num_layers']
    ).to(device)
    
    # load saved weights
    model.load_state_dict(torch.load(cfg['paths']['model_save'], 
                                     map_location=device,
                                     weights_only=True))
    model.eval()  # disable dropout for inference
    
    return model, char2idx, idx2lang

def predict(text, model, char2idx, idx2lang, cfg, device):
    # encode raw text exactly like training time
    max_len = cfg['data']['max_len']
    encoded = encode_text(text, char2idx, max_len)
    
    # convert to tensor and add batch dimension
    # model expects shape (batch_size, max_len)
    # unsqueeze(0) adds batch dimension → (1, max_len)
    x = torch.LongTensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x)                         # raw scores for 20 languages
        probs  = F.softmax(output, dim=1)         # convert to probabilities (sum to 1)
        confidence, pred_idx = probs.max(dim=1)   # get highest probability
    
    language   = idx2lang[str(pred_idx.item())]
    confidence = confidence.item() * 100
    
    # top 3 predictions with confidence
    top3_probs, top3_idxs = probs.topk(3, dim=1)
    top3 = [(idx2lang[str(i.item())], round(p.item()*100, 2)) 
            for i, p in zip(top3_idxs[0], top3_probs[0])]
    
    return language, confidence, top3

if __name__ == "__main__":
    cfg    = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, char2idx, idx2lang = load_model(cfg, device)
    
    # test sentences in different languages
    test_inputs = [
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous?",
        "यह एक हिंदी वाक्य है",
        "これは日本語のテキストです",
        "Hola, ¿cómo estás?",
        "Yaar tu kya kar raha hai",       # Hinglish edge case
        "😂😭🔥",                          # emoji edge case
    ]
    
    print(f"\n{'Input':<40} {'Predicted':>10} {'Confidence':>12}")
    print("-" * 65)
    for text in test_inputs:
        lang, conf, top3 = predict(text, model, char2idx, idx2lang, cfg, device)
        print(f"{text:<40} {lang:>10} {conf:>10.2f}%")
        print(f"  Top 3: {top3}")
        print()