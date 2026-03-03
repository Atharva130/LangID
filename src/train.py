import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import yaml
from src.data_loader import load_config, load_data
from src.preprocess import (build_vocab, build_label_encoder,
                             encode_dataset, save_vocab)
from src.model import LangIDModel

def train():
    # ── 1. Load config ──────────────────────────────────────────────
    cfg = load_config()

    # ── 2. Load raw data ────────────────────────────────────────────
    print("Loading data...")
    train_df, val_df, test_df = load_data(cfg)

    # ── 3. Preprocess ───────────────────────────────────────────────
    print("Building vocab...")
    char2idx, idx2char   = build_vocab(train_df)
    lang2idx, idx2lang   = build_label_encoder(train_df)
    max_len              = cfg['data']['max_len']

    print("Encoding dataset...")
    X_train, y_train = encode_dataset(train_df, char2idx, lang2idx, max_len)
    X_val,   y_val   = encode_dataset(val_df,   char2idx, lang2idx, max_len)

    # ── 4. Save vocab (needed for inference later) ──────────────────
    os.makedirs('data', exist_ok=True)
    save_vocab(char2idx, lang2idx, idx2lang, cfg['paths']['vocab_save'])

    # ── 5. Convert numpy arrays → PyTorch tensors ───────────────────
    # LongTensor for integer inputs and labels
    X_train_t = torch.LongTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t   = torch.LongTensor(X_val)
    y_val_t   = torch.LongTensor(y_val)

    # ── 6. Create DataLoaders ────────────────────────────────────────
    # DataLoader batches your data automatically + shuffles each epoch
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=cfg['training']['batch_size'],
                              shuffle=False)

    # ── 7. Setup device (GPU if available) ──────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # ── 8. Build model and move to GPU ──────────────────────────────
    model = LangIDModel(
        vocab_size    = len(char2idx),
        embedding_dim = cfg['model']['embedding_dim'],
        gru_units     = cfg['model']['gru_units'],
        num_classes   = cfg['model']['num_classes'],
        dropout       = cfg['model']['dropout'],
        num_layers    = cfg['model']['num_layers']
    ).to(device)

    # ── 9. Loss function and optimizer ──────────────────────────────
    # CrossEntropyLoss = standard for multi-class classification
    # Adam = adaptive learning rate optimizer, works well out of the box
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg['training']['learning_rate'])

    # ── 10. TensorBoard writer ──────────────────────────────────────
    os.makedirs(cfg['paths']['logs'], exist_ok=True)
    writer = SummaryWriter(cfg['paths']['logs'])

    # ── 11. Training loop ───────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    epochs = cfg['training']['epochs']
    patience = cfg['training']['patience']

    for epoch in range(epochs):
        # ── Training phase ──
        model.train()   # tells model we are training (enables dropout)
        train_loss, train_correct = 0, 0

        for X_batch, y_batch in train_loader:
            # move batch to GPU
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()        # reset gradients from last step
            outputs = model(X_batch)     # forward pass
            loss = criterion(outputs, y_batch)  # compute loss
            loss.backward()              # backpropagation
            optimizer.step()             # update weights

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        # ── Validation phase ──
        model.eval()    # tells model we are evaluating (disables dropout)
        val_loss, val_correct = 0, 0

        with torch.no_grad():   # no gradient computation needed
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y_batch).sum().item()

        # ── Compute epoch metrics ──
        train_acc = train_correct / len(train_dataset)
        val_acc   = val_correct   / len(val_dataset)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        # ── Log to TensorBoard ──
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val',   avg_val_loss,   epoch)
        writer.add_scalar('Accuracy/train', train_acc,  epoch)
        writer.add_scalar('Accuracy/val',   val_acc,    epoch)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # ── Early stopping + save best model ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg['paths']['model_save'])
            print(f"  ✓ Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    train()