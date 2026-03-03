import torch
from src.data_loader import load_config
from src.preprocess import load_vocab, encode_text
from src.model import LangIDModel
import torch.nn.functional as F

def load_model(cfg, device):
    char2idx, lang2idx, idx2lang = load_vocab(cfg['paths']['vocab_save'])
    model = LangIDModel(
        vocab_size    = len(char2idx),
        embedding_dim = cfg['model']['embedding_dim'],
        gru_units     = cfg['model']['gru_units'],
        num_classes   = cfg['model']['num_classes'],
        dropout       = cfg['model']['dropout'],
        num_layers    = cfg['model']['num_layers']
    ).to(device)
    model.load_state_dict(torch.load(cfg['paths']['model_save'],
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    return model, char2idx, idx2lang

def predict(text, model, char2idx, idx2lang, cfg, device):
    encoded = encode_text(text, char2idx, cfg['data']['max_len'])
    x = torch.LongTensor(encoded).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        probs  = F.softmax(output, dim=1)
        conf, pred_idx = probs.max(dim=1)
    return idx2lang[str(pred_idx.item())], round(conf.item() * 100, 2)

def run_tests():
    cfg    = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, char2idx, idx2lang = load_model(cfg, device)

    # each test: (input_text, expected_language, description)
    test_cases = [
        # normal cases
        ("यह एक हिंदी वाक्य है",           "hi", "Hindi normal"),
        ("これは日本語です",                  "ja", "Japanese normal"),
        ("Bonjour comment allez vous",      "fr", "French normal"),
        ("Hola cómo estás hoy",             "es", "Spanish normal"),

        # edge cases
        ("😂😭🔥💀",                         None, "Emojis only"),
        ("Yaar tu kya kar raha hai bhai",   None, "Hinglish"),
        ("Hello مرحبا Bonjour",             None, "Mixed languages"),
        ("a",                               None, "Single character"),
        ("123 456 789",                     None, "Numbers only"),
        ("!!!???###",                       None, "Special characters only"),
    ]

    print(f"\n{'Test':<30} {'Predicted':<12} {'Confidence':<12} {'Status'}")
    print("-" * 70)

    for text, expected, description in test_cases:
        lang, conf = predict(text, model, char2idx, idx2lang, cfg, device)

        if expected is None:
            # edge cases — we just observe, no pass/fail
            status = "⚪ OBSERVE"
        elif lang == expected:
            status = "✅ PASS"
        else:
            status = f"❌ FAIL (expected {expected})"

        print(f"{description:<30} {lang:<12} {conf:<12} {status}")
        print(f"  Input: {text[:50]}")
        print()

if __name__ == "__main__":
    run_tests()