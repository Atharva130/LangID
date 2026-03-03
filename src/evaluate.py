import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
import os

from src.data_loader import load_config, load_data
from src.preprocess import build_vocab, build_label_encoder, encode_dataset, load_vocab
from src.model import LangIDModel

def evaluate():
    # ── 1. Load config ──────────────────────────────────────────────
    cfg = load_config()

    # ── 2. Load data ────────────────────────────────────────────────
    print("Loading data...")
    train_df, val_df, test_df = load_data(cfg)

    # ── 3. Load saved vocab ─────────────────────────────────────────
    # we use the SAVED vocab, not rebuild it
    # this is critical — must be same mapping as training time
    print("Loading vocab...")
    char2idx, lang2idx, idx2lang = load_vocab(cfg['paths']['vocab_save'])

    # ── 4. Encode test set ──────────────────────────────────────────
    print("Encoding test set...")
    X_test, y_test = encode_dataset(test_df, char2idx, lang2idx, cfg['data']['max_len'])

    # ── 5. Create DataLoader ─────────────────────────────────────────
    X_test_t  = torch.LongTensor(X_test)
    y_test_t  = torch.LongTensor(y_test)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader  = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    # ── 6. Load trained model ────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LangIDModel(
        vocab_size    = len(char2idx),
        embedding_dim = cfg['model']['embedding_dim'],
        gru_units     = cfg['model']['gru_units'],
        num_classes   = cfg['model']['num_classes'],
        dropout       = cfg['model']['dropout'],
        num_layers    = cfg['model']['num_layers']
    ).to(device)

    # load the saved weights into the model
    model.load_state_dict(torch.load(cfg['paths']['model_save'], map_location=device,weights_only=True))
    model.eval()
    print(f"Model loaded from {cfg['paths']['model_save']}")

    # ── 7. Run inference on test set ─────────────────────────────────
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── 8. Overall accuracy ──────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # ── 9. Per-language classification report ────────────────────────
    # shows precision, recall, f1 for each language individually
    lang_names = [idx2lang[str(i)] for i in range(len(idx2lang))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=lang_names))

    # ── 10. Confusion matrix ─────────────────────────────────────────
    os.makedirs('notebooks/plots', exist_ok=True)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=lang_names,
                yticklabels=lang_names,
                cmap='Blues')
    plt.title('Confusion Matrix — Language Identification')
    plt.ylabel('True Language')
    plt.xlabel('Predicted Language')
    plt.tight_layout()
    plt.savefig('notebooks/plots/confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved to notebooks/plots/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()