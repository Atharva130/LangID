# 🌍 Language Identifier — Real-Time NLP with GRU

> A production-style NLP system that identifies 20 languages in real time using a Bidirectional GRU neural network. Built with PyTorch, served via FastAPI, with a live interactive frontend.

---

## 🎯 What It Does

Type any text → get the language identified instantly with confidence scores.
Supports 20 languages including Arabic, Chinese, Japanese, Hindi, Urdu and more.

## 🎥 Demo
```
![Animation](https://github.com/user-attachments/assets/ff6d7286-66cd-4b2f-854b-dbbbd64f5f90)
```

## 🏗️ Architecture
```
Raw Text Input
      ↓
Character-level Tokenization
      ↓
Embedding Layer (64-dim)
      ↓
Bidirectional GRU (128 units × 2 layers)
      ↓
Dropout (0.4)
      ↓
Linear Layer → Softmax (20 classes)
      ↓
Language + Confidence Score
```

**Why Character-level?**  
Word-level models need separate vocabularies per language. Character-level handles Arabic, Chinese, Thai, Devanagari — all with one unified vocab of 4400 characters. Also robust to typos and informal text by design.

**Why Bidirectional GRU?**  
GRU is faster than LSTM with comparable accuracy on short sequences. Bidirectional means the model reads text both left→right and right→left — capturing context from both directions simultaneously.

---

## 📊 Results

### Final Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | 96% |
| Macro F1 Score | 0.96 |
| Vocab Size | 4,400 characters |
| Total Parameters | 732,180 |
| Training Time | ~5 mins on RTX 4050 |
| Dataset | papluca/language-identification (70k samples) |

### Per-Language F1 Scores
| Language | F1 | Language | F1 |
|----------|----|----------|----|
| Arabic | 1.00 | Japanese | 1.00 |
| Greek | 1.00 | Thai | 1.00 |
| Vietnamese | 1.00 | Chinese | 1.00 |
| French | 0.97 | German | 0.98 |
| Hindi | 0.98 | Turkish | 0.98 |
| Urdu | 0.98 | Spanish | 0.96 |
| Bulgarian | 0.80 | Russian | 0.85 |

### Experiment Tracking
| Experiment | GRU Units | Dropout | Patience | Val Accuracy | Decision |
|------------|-----------|---------|----------|-------------|----------|
| Baseline | 128 | 0.3 | 3 | 96.22% | Good start |
| Larger model | 256 | 0.3 | 5 | 94.79% | Overfit — rejected |
| **Final** | **128** | **0.4** | **5** | **96.63%** | ✅ Best model |

---

## 🔍 Confusion Analysis

The hardest language pair was **Bulgarian vs Russian** — both use Cyrillic script so the model must rely on subtle character frequency differences rather than script identity. 154 out of 500 Bulgarian samples were misclassified as Russian.

Languages with unique scripts (Arabic, Japanese, Thai, Chinese) achieved perfect F1 of 1.00 — confirming that character-level modeling is naturally script-aware.

---

## 🧪 Edge Case Analysis

| Input | Predicted | Observation |
|-------|-----------|-------------|
| Emojis only 😂🔥 | Uncertain | Never seen in training — expected failure |
| Hinglish "Yaar kya kar raha hai" | Swahili | Romanized Hindi resembles Swahili's Latin vowel patterns |
| Mixed "Hello مرحبا Bonjour" | Turkish | Model picks dominant character pattern |
| Single character "a" | Polish | Insufficient context for reliable prediction |
| Numbers "123 456" | Polish | No language signal in digits |

---

## 🗂️ Project Structure
```
LangID/
│
├── data/                        ← saved model and vocab
├── notebooks/
│   ├── 01_eda.ipynb             ← data exploration
│   └── plots/                   ← saved visualizations
│
├── src/
│   ├── data_loader.py           ← loads dataset from HuggingFace
│   ├── preprocess.py            ← tokenization, encoding, vocab
│   ├── model.py                 ← Bidirectional GRU architecture
│   ├── train.py                 ← training loop with early stopping
│   ├── evaluate.py              ← metrics and confusion matrix
│   ├── predict.py               ← inference on raw text
│   └── api.py                   ← FastAPI backend
│
├── tests/
│   └── test_edge_cases.py       ← robustness testing
│
├── index.html                   ← frontend UI
├── config.yaml                  ← all hyperparameters
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 How To Run

**1. Clone and install**
```bash
git clone https://github.com/Atharva130/LangID.git
cd LangID
pip install -r requirements.txt
```

**2. Train the model**
```bash
make train
```

**3. Evaluate**
```bash
make evaluate
```

**4. Start the API**
```bash
make run
```

**5. Open frontend**  
Open `index.html` in your browser and start typing.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Model training and inference |
| HuggingFace Datasets | Dataset loading |
| FastAPI | REST API backend |
| Uvicorn | ASGI server |
| Scikit-learn | Evaluation metrics |
| Matplotlib / Seaborn | Visualizations |
| TensorBoard | Training curves |

---

## 💡 What I Would Improve

- Add confidence threshold — return "Unknown" below 50% confidence
- Train on Hinglish data to handle code-switching
- Add WebSocket endpoint for true keystroke-level real-time detection
- Quantize model with TFLite for faster CPU inference
- Deploy on Render with Docker for public access

---

