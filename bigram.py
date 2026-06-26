"""
bigram.py — Bigram Language Model
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    load_text, build_vocab, encode,
    NextTokenDataset, train_one_epoch, estimate_loss, sample_model,
)
from config import ModelConfig, TrainConfig

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
model_config = ModelConfig(block_size=8)
train_config = TrainConfig(
    batch_size=32,
    epochs=5,
    learning_rate=1e-2
    )
# ---------------------------------------------------------------------------

torch.manual_seed(1337)


# ===========================================================================
# Model
# ===========================================================================

class BigramLanguageModel(nn.Module):
    """
    Bigram 언어 모델.
    각 토큰이 임베딩 테이블에서 직접 다음 토큰의 logits를 읽어온다.
    컨텍스트를 사용하지 않는 가장 단순한 형태.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        # idx is (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
        return logits


# ===========================================================================
# Main — 학습 및 생성
# ===========================================================================

if __name__ == '__main__':
    input_path = Path(__file__).parent / 'input.txt'
    text = load_text(input_path)
    
    chars, stoi, itos, vocab_size = build_vocab(text)

    model_config.vocab_size = vocab_size
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = NextTokenDataset(train_data, model_config.block_size)
    val_dataset = NextTokenDataset(val_data, model_config.block_size)

    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=False)

    model = BigramLanguageModel(model_config.vocab_size)
    m = model.to(train_config.device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    print("Training Bigram Model...")
    for epoch in range(train_config.epochs):
        train_loss = train_one_epoch(
            m,
            train_loader,
            optimizer,
            train_config.device,
            max_steps=train_config.max_steps,
        )
        val_loss = estimate_loss(
            m,
            val_loader,
            train_config.device,
            eval_iters=train_config.eval_iters,
        )
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    print("\n--- Sample generation ---")
    print(sample_model(
        m,
        model_config.block_size,
        stoi,
        itos,
        train_config.device,
        start_text=train_config.sample_start_text,
        max_new_tokens=train_config.sample_max_new_tokens,
    ))