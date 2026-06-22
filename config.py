from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    """GPT 모델의 아키텍처 하이퍼파라미터"""
    vocab_size: int = 65    # 기본값 (shakespear char 기준)
    block_size: int = 256   # 최대 문맥 길이
    n_embd: int = 384       # 임베딩 차원
    n_head: int = 6         # 어텐션 헤드 수
    n_layer: int = 6        # 트랜스포머 블록 수
    dropout: float = 0.2    # 드롭아웃 확률
    weight_tying: bool = True # 임베딩과 lm_head 가중치 공유 여부

@dataclass
class TrainConfig:
    """학습 관련 하이퍼파라미터"""
    batch_size: int = 64
    max_iters: int = 2000   # 총 학습 스텝 수
    eval_interval: int = 200
    eval_iters: int = 100
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    mixed_precision: bool = True # AMP 사용 여부
