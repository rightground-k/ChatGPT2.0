"""
gpt.py — Generative Pre-trained Transformer
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from common import (
    load_text, build_vocab, encode, get_batch,
    sequence_cross_entropy, sample_model, get_lr
)
from config import ModelConfig, TrainConfig

# ---------------------------------------------------------------------------
# GPT Model Definition
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    단일 행렬 연산으로 최적화된 Multi-Head Attention
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # causal mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """
    MLP Block (ReLU 사용)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer Block
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    GPT Language Model (Weight Tying, 설정 시스템 적용)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        if config.weight_tying:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        device = idx.device
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

# ---------------------------------------------------------------------------
# Training Script
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, model_config):
    """빠른 데이터 배치를 활용하는 Loss 계산"""
    out = {}
    model.eval()
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, model_config.block_size, config.batch_size, config.device)
            # AMP 적용
            if config.mixed_precision:
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = model(X)
                    loss = sequence_cross_entropy(logits, Y)
            else:
                logits = model(X)
                loss = sequence_cross_entropy(logits, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

if __name__ == '__main__':
    # 설정 로드
    train_config = TrainConfig()
    
    print("Loading data...")
    text = load_text('input.txt')
    chars, stoi, itos, vocab_size = build_vocab(text)
    
    model_config = ModelConfig(vocab_size=vocab_size)
    
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print("Initializing model...")
    model = GPTLanguageModel(model_config)
    m = model.to(train_config.device)
    print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    
    # PyTorch 1.12+ (or older via torch.cuda.amp)
    scaler = None
    if train_config.mixed_precision and 'cuda' in train_config.device:
        try:
            scaler = torch.amp.GradScaler('cuda')
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler()
    
    print("Training GPU-optimized GPT...")
    for iter_num in range(train_config.max_iters):
        
        # 주기적 평가 및 로깅
        if iter_num % train_config.eval_interval == 0 or iter_num == train_config.max_iters - 1:
            losses = estimate_loss(m, train_data, val_data, train_config, model_config)
            print(f"step {iter_num:4d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # LR 스케줄링 적용
        lr = get_lr(iter_num, train_config.learning_rate, warmup_iters=train_config.max_iters//10, 
                    lr_decay_iters=train_config.max_iters, min_lr=train_config.learning_rate / 10.0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 학습 스텝
        xb, yb = get_batch(train_data, model_config.block_size, train_config.batch_size, train_config.device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            device_type = 'cuda' if 'cuda' in train_config.device else 'cpu'
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                logits = m(xb)
                loss = sequence_cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = m(xb)
            loss = sequence_cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()

    print("\n--- Generating sample text ---")
    generated_text = sample_model(m, model_config.block_size, stoi, itos, train_config.device, 
                                  start_text="\n", max_new_tokens=500, temperature=0.8, top_k=40)
    print(generated_text)