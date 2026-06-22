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

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size = 32        # 병렬 처리할 독립 시퀀스 수
block_size = 8         # 최대 컨텍스트 길이
max_epochs = 5         # 학습 에포크 수
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = NextTokenDataset(train_data, block_size)
    val_dataset = NextTokenDataset(val_data, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BigramLanguageModel(vocab_size)
    m = model.to(device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Training Bigram Model...")
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(m, train_loader, optimizer, device, max_steps=600)
        val_loss = estimate_loss(m, val_loader, device, eval_iters=200)
        print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    print("\n--- Sample generation ---")
    print(sample_model(m, block_size, stoi, itos, device,
                       start_text="\n", max_new_tokens=500))