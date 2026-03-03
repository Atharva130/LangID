from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from src.data_loader import load_config
from src.preprocess import load_vocab, encode_text
from src.model import LangIDModel

# ── App setup ──────────────────────────────────────────────────────
app = FastAPI(title="Language Identifier API")

# allows frontend to talk to backend (cross origin requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once when server starts ─────────────────────────────
# we don't reload model on every request — that would be very slow
cfg    = load_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ── Request schema ──────────────────────────────────────────────────
# defines what the API expects as input
class TextInput(BaseModel):
    text: str

# ── Routes ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Language Identifier API is running"}

@app.post("/predict")
def predict(input: TextInput):
    text = input.text
    
    # handle empty input
    if not text.strip():
        return {"error": "empty input"}
    
    # encode text same way as training
    encoded = encode_text(text, char2idx, cfg['data']['max_len'])
    x = torch.LongTensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x)
        probs  = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    
    # top 3 predictions
    top3_probs, top3_idxs = probs.topk(3, dim=1)
    top3 = [
        {
            "language": idx2lang[str(i.item())],
            "confidence": round(p.item() * 100, 2)
        }
        for i, p in zip(top3_idxs[0], top3_probs[0])
    ]
    
    return {
        "text"      : text,
        "language"  : idx2lang[str(pred_idx.item())],
        "confidence": round(confidence.item() * 100, 2),
        "top3"      : top3
    }