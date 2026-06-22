"""
common.py — 공통 유틸리티 모듈
bigram.py와 gpt.py에서 공유하는 데이터 파이프라인, 학습, 평가, 생성 코드를 모아둔 파일.
"""

import math
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data loading & tokenization
# ---------------------------------------------------------------------------

def load_text(path):
    """텍스트 파일을 읽어 문자열로 반환한다."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_vocab(text):
    """
    문자 수준 어휘를 만든다.

    Returns:
        chars: 정렬된 고유 문자 리스트
        stoi: 문자 → 정수 매핑
        itos: 정수 → 문자 매핑
        vocab_size: 어휘 크기
    """
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos, len(chars)


def encode(s, stoi):
    """문자열 → 정수 리스트"""
    return [stoi[c] for c in s]


def decode(l, itos):
    """정수 리스트 → 문자열"""
    return ''.join([itos[i] for i in l])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def get_batch(data, block_size, batch_size, device):
    """
    DataLoader 병목을 우회하기 위한 빠른 무작위 배치 추출 유틸리티.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class NextTokenDataset(Dataset):
    """
    연속된 문자 시퀀스에서 (x, y) 쌍을 생성하는 데이터셋.
    x = data[idx : idx+block_size]
    y = data[idx+1 : idx+block_size+1]
    """

    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sequence_cross_entropy(logits, targets):
    """시퀀스 형태 logits (B, T, C)에 대한 cross-entropy loss."""
    return F.cross_entropy(logits.transpose(1, 2), targets)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, max_steps=None, max_grad_norm=1.0, scaler=None):
    """
    1 에포크 학습을 수행한다.
    """
    model.train()
    total_loss, total_count = 0.0, 0
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                logits = model(xb)
                loss = sequence_cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = sequence_cross_entropy(logits, yb)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break
    return total_loss / total_count


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, val_loader, device, eval_iters=200):
    """검증 데이터에 대한 평균 loss를 계산한다."""
    model.eval()
    total_loss, total_count = 0.0, 0
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    for step, (xb, yb) in enumerate(val_loader):
        xb, yb = xb.to(device), yb.to(device)
        # AMP 컨텍스트 적용
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type=='cuda')):
            logits = model(xb)
            loss = sequence_cross_entropy(logits, yb)
            
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if step + 1 >= eval_iters:
            break
    model.train()
    return total_loss / total_count if total_count > 0 else 0.0


# ---------------------------------------------------------------------------
# Training Utilities
# ---------------------------------------------------------------------------

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """Cosine Annealing with Warmup learning rate scheduler"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)



# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_model(model, block_size, stoi, itos, device,
                 start_text="\n", max_new_tokens=300,
                 temperature=1.0, top_k=None):
    """
    학습된 모델로 텍스트를 생성한다.
    """
    model.eval()
    if start_text:
        context = torch.tensor([encode(start_text, stoi)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    idx = context
    out = list(start_text) if start_text else []

    for _ in range(max_new_tokens):
        # 컨텍스트를 block_size로 자르기
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # (B, C) — 마지막 위치의 logits만

        # Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        out.append(itos[idx_next.item()])
        idx = torch.cat((idx, idx_next), dim=1)

    return "".join(out)