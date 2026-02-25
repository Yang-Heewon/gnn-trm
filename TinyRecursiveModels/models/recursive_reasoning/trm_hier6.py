from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
import os
import numpy as np

# Import layers (Assuming these exist in your models folder)
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    """Carries the hidden states for the hierarchical model (1 H + 6 Ls)"""
    z_H: torch.Tensor
    z_L1: torch.Tensor
    z_L2: torch.Tensor
    z_L3: torch.Tensor
    z_L4: torch.Tensor
    z_L5: torch.Tensor
    z_L6: torch.Tensor

@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    """Wrapper carry that includes metadata like steps and halted status"""
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    num_relation_identifiers: int = 0 
    relation_emb_ndim: int = 0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"
    mlp_t: bool = False 
    puzzle_emb_len: int = 16 
    no_ACT_continue: bool = True 

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(hidden_size=self.config.seq_len + self.puzzle_emb_len, expansion=config.expansion)
        else:
            self.self_attn = Attention(hidden_size=config.hidden_size, head_dim=config.hidden_size // config.num_heads, num_heads=config.num_heads, num_key_value_heads=config.num_heads, causal=False)
        self.mlp = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states, attention_mask=attention_mask), variance_epsilon=self.norm_eps)
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1 / self.embed_scale
        
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.node_cls_head = CastedLinear(self.config.hidden_size, 1, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len

        # [수정 1] 데이터 그릇 초기화 (파일 로딩 삭제)
        self.puzzle_emb_data = None   # 학습 스크립트에서 주입됨
        self.relation_emb_data = None # 학습 스크립트에서 주입됨

        # [수정 2] Entity Projector (384 -> 512)
        if self.config.puzzle_emb_ndim > 0 and self.config.puzzle_emb_ndim != self.config.hidden_size:
            self.puzzle_projector = nn.Linear(self.config.puzzle_emb_ndim, self.config.hidden_size, bias=False).to(self.forward_dtype)
            nn.init.xavier_uniform_(self.puzzle_projector.weight)
            #이 부분을 통해서 relation만 학습하게 설정
            for param in self.puzzle_projector.parameters():
                param.requires_grad = False
        else: 
            self.puzzle_projector = None

        # [수정 3] Relation Projector (384 -> 512)
        if hasattr(self.config, 'relation_emb_ndim') and self.config.relation_emb_ndim > 0:
            if self.config.relation_emb_ndim != self.config.hidden_size:
                self.relation_projector = nn.Linear(self.config.relation_emb_ndim, self.config.hidden_size, bias=False).to(self.forward_dtype)
                nn.init.xavier_uniform_(self.relation_projector.weight)
            else: 
                self.relation_projector = None

        # [수정 4] Score Head 추가 (Ranking 점수 계산용)
        self.score_proj = None

        # --- RoPE ---
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads, max_position_embeddings=self.config.seq_len + self.puzzle_emb_len + 128, base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len + 128, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # --- Layers & Init ---
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L1_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L2_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L3_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L4_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L5_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)
        self.L6_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=0.02), persistent=True)

        with torch.no_grad():
            self.node_cls_head.weight.zero_()
            self.node_cls_head.bias.fill_(-5)

    # trm_hier6.py 내 TinyRecursiveReasoningModel_ACTV1_Inner 클래스 내부

    def puzzle_emb_lookup(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        # 데이터가 없으면 0 반환 (안전장치)
        if self.puzzle_emb_data is None:
            return torch.zeros((*puzzle_identifiers.shape, self.config.hidden_size), 
                               device=puzzle_identifiers.device, dtype=self.forward_dtype)
        
        # 외부 데이터에서 Lookup
        ids_cpu = puzzle_identifiers.cpu()
        if isinstance(self.puzzle_emb_data, np.ndarray):
            emb = torch.from_numpy(self.puzzle_emb_data[ids_cpu.numpy()])
        else:
            # Tensor인 경우
            dev = self.puzzle_emb_data.device
            emb = F.embedding(puzzle_identifiers.to(dev), self.puzzle_emb_data)
        
        emb = emb.to(device=puzzle_identifiers.device, dtype=self.forward_dtype)
        
        # Projection (384 -> 512)
        if hasattr(self, 'puzzle_projector') and self.puzzle_projector is not None:
            emb = self.puzzle_projector(emb)
        return emb

    def relation_emb_lookup(self, relation_identifiers: torch.Tensor) -> torch.Tensor:
        if self.relation_emb_data is None:
            return torch.zeros((*relation_identifiers.shape, self.config.hidden_size),
                            device=relation_identifiers.device, dtype=self.forward_dtype)

        ids_cpu = relation_identifiers.cpu()

        if isinstance(self.relation_emb_data, np.ndarray):
            emb = torch.from_numpy(self.relation_emb_data[ids_cpu.numpy()])
        else:
            dev = self.relation_emb_data.device
            emb = F.embedding(relation_identifiers.to(dev), self.relation_emb_data)

        emb = emb.to(device=relation_identifiers.device, dtype=self.forward_dtype)

        if hasattr(self, 'relation_projector') and self.relation_projector is not None:
            emb = self.relation_projector(emb)
        return emb

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor, relation_identifiers: torch.Tensor = None):
        B, L = input.shape
        embedding = self.embed_tokens(input.to(torch.int32))
        
        if self.config.puzzle_emb_ndim > 0:
            # lookup 함수들이 이미 512차원(hidden_size)으로 맞춰서 줍니다.
            puzzle_embedding = self.puzzle_emb_lookup(puzzle_identifiers)
            
            if relation_identifiers is not None:
                relation_vecs = self.relation_emb_lookup(relation_identifiers)
                if relation_vecs is not None:
                    # 차원이 이미 512이므로 바로 더하기 가능
                    puzzle_embedding = puzzle_embedding + relation_vecs
            
            puzzle_embedding = puzzle_embedding.view(B, -1, self.config.hidden_size)
            embedding = torch.cat((puzzle_embedding, embedding), dim=1)
            
        if self.config.pos_encodings == "learned":
            current_seq_len = embedding.size(1)
            pos_emb = self.embed_pos.embedding_weight[:current_seq_len]
            embedding = 0.707106781 * (embedding + pos_emb.to(self.forward_dtype))
            
        return  embedding
    # trm_hier6.py 파일 내 TinyRecursiveReasoningModel_ACTV1_Inner 클래스 내부

    def empty_carry(self, batch_size: int, seq_len: int = None):
        """
        학습 시작 시 6개의 L 레이어와 1개의 H 레이어에 대한 빈 텐서를 생성합니다.
        """
        # 모델의 기본 장치(GPU)와 데이터 타입을 확인합니다.
        device = self.H_init.device
        
        # 시퀀스 길이를 지정하지 않으면 기본 설정값(시퀀스+퍼즐 임베딩)을 사용합니다.
        if seq_len is None:
            seq_len = self.config.seq_len + self.puzzle_emb_len

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L1=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L2=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L3=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L4=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L5=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
            z_L6=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype, device=device),
        )
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, target_seq_len: int = None):
        device = self.H_init.device
        reset_flag = reset_flag.to(device)
        
        # [🔥 Critical Fix] Use clone() to break in-place graph connections
        def _safe_get(t):
            return t.to(device).clone()

        # Dynamic Resizing Logic
        if target_seq_len is not None and carry.z_H.shape[1] != target_seq_len:
            def _resize(t):
                new_t = torch.zeros(t.size(0), target_seq_len, t.size(2), dtype=t.dtype, device=t.device)
                copy_len = min(t.size(1), target_seq_len)
                new_t[:, :copy_len, :] = t[:, :copy_len, :]
                return new_t
            
            carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=_resize(carry.z_H), z_L1=_resize(carry.z_L1), z_L2=_resize(carry.z_L2),
                z_L3=_resize(carry.z_L3), z_L4=_resize(carry.z_L4), z_L5=_resize(carry.z_L5), z_L6=_resize(carry.z_L6)
            )

        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H = torch.where(reset_flag.view(-1, 1, 1), self.H_init, _safe_get(carry.z_H)),
            z_L1= torch.where(reset_flag.view(-1, 1, 1), self.L1_init, _safe_get(carry.z_L1)),
            z_L2= torch.where(reset_flag.view(-1, 1, 1), self.L2_init, _safe_get(carry.z_L2)),
            z_L3= torch.where(reset_flag.view(-1, 1, 1), self.L3_init, _safe_get(carry.z_L3)),
            z_L4= torch.where(reset_flag.view(-1, 1, 1), self.L4_init, _safe_get(carry.z_L4)),
            z_L5= torch.where(reset_flag.view(-1, 1, 1), self.L5_init, _safe_get(carry.z_L5)),
            z_L6= torch.where(reset_flag.view(-1, 1, 1), self.L6_init, _safe_get(carry.z_L6)),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]):
        input_ids = batch.get("inputs", batch.get("input_ids"))
        rel_ids = batch.get("relation_identifiers", None)
        
        # Embeddings & Dimensions
        input_embeddings = self._input_embeddings(input_ids, batch["puzzle_identifiers"], relation_identifiers=rel_ids)
        target_len = input_embeddings.shape[1]
        device = input_embeddings.device

        # [🔥 Critical Fix] Force resize carry to match input length every step
        current_len = carry.z_H.shape[1]
        if current_len != target_len:
            def _force_resize(t):
                new_t = torch.zeros(t.size(0), target_len, t.size(2), dtype=t.dtype, device=device)
                copy_len = min(current_len, target_len)
                new_t[:, :copy_len, :] = t[:, :copy_len, :]
                return new_t

            carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
                z_H=_force_resize(carry.z_H),
                z_L1=_force_resize(carry.z_L1), z_L2=_force_resize(carry.z_L2),
                z_L3=_force_resize(carry.z_L3), z_L4=_force_resize(carry.z_L4),
                z_L5=_force_resize(carry.z_L5), z_L6=_force_resize(carry.z_L6),
            )

        # RoPE
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb()
            if isinstance(cos_sin, tuple):
                cos, sin = cos_sin
                if cos.shape[0] > target_len:
                    cos, sin = cos[:target_len], sin[:target_len]
                elif cos.shape[0] < target_len:
                    pad = target_len - cos.shape[0]
                    cos = F.pad(cos, (0,0,0,0,0,pad))
                    sin = F.pad(sin, (0,0,0,0,0,pad))
                # IMPORTANT: break potential alias/version sharing with cached RoPE buffers.
                cos_sin = (cos.contiguous().clone(), sin.contiguous().clone())
        seq_info = dict(cos_sin=cos_sin)

        # Recursive Pass
        z_H = carry.z_H
        z_L = [carry.z_L1, carry.z_L2, carry.z_L3, carry.z_L4, carry.z_L5, carry.z_L6]

        # H-1 cycles (no grad)
        # H-1 cycles: train에서는 grad ON, eval에서만 no_grad
        if self.training:
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L_ = sum(z_L)
                    z_L[_L_step] = self.L_level(z_L_, z_H + input_embeddings, **seq_info)
                z_L_ = sum(z_L)
                z_H = self.L_level(z_H, z_L_, **seq_info)
        else:
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles - 1):
                    for _L_step in range(self.config.L_cycles):
                        z_L_ = sum(z_L)
                        z_L[_L_step] = self.L_level(z_L_, z_H + input_embeddings, **seq_info)
                    z_L_ = sum(z_L)
                    z_H = self.L_level(z_H, z_L_, **seq_info)
        
        # Final cycle (with grad)
        for _L_step in range(self.config.L_cycles):
            z_L_ = sum(z_L)
            z_L[_L_step] = self.L_level(z_L_, z_H + input_embeddings, **seq_info)
        z_L_ = sum(z_L)
        z_H = self.L_level(z_H, z_L_, **seq_info)

        # Final cycle 끝난 직후
        z_H_last = z_H  # ✅ 이건 detach 하지 말고 이번 step scorer용으로 쓸 것

        # 다음 step carry는 detach해서 그래프 끊기(Truncated BPTT)
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=z_H.detach(),
            z_L1=z_L[0].detach(), z_L2=z_L[1].detach(), z_L3=z_L[2].detach(),
            z_L4=z_L[3].detach(), z_L5=z_L[4].detach(), z_L6=z_L[5].detach(),
        )

        output = self.lm_head(z_H)
        node_logits = self.node_cls_head(z_H).squeeze(-1).to(torch.float32)

        # ✅ z_H_last 추가로 반환
        return new_carry, output, node_logits, z_H_last

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb_lookup

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch.get("inputs", batch.get("input_ids")).shape[0]
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        device = next(iter(batch.values())).device
        halted_sync = carry.halted.to(device)
        steps_sync = carry.steps.to(device)
        current_data_sync = {k: v.to(device) for k, v in carry.current_data.items()}

        # 현재 seq_len 계산
        P_len = batch["puzzle_identifiers"].shape[1] if "puzzle_identifiers" in batch else self.inner.puzzle_emb_len
        if "input_ids" in batch:
            current_seq_len = batch["input_ids"].shape[1] + P_len
        elif "inputs" in batch:
            current_seq_len = batch["inputs"].shape[1] + P_len
        else:
            current_seq_len = self.config.seq_len + P_len

        # ✅ 1) new_inner_carry 먼저 정의
        new_inner_carry = self.inner.reset_carry(halted_sync, carry.inner_carry, target_seq_len=current_seq_len)
        new_steps = torch.where(halted_sync, torch.tensor(0, device=device), steps_sync)

        # ✅ 2) current data 구성
        dynamic_keys = ["puzzle_identifiers", "relation_identifiers", "candidate_mask"]
        new_current_data = {
            k: torch.where(halted_sync.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in current_data_sync.items()
            if (k in batch) and (k not in dynamic_keys)
        }

        # ✅ 3) 이번 step batch data 주입(필수)
        if "puzzle_identifiers" in batch: new_current_data["puzzle_identifiers"] = batch["puzzle_identifiers"]
        if "relation_identifiers" in batch: new_current_data["relation_identifiers"] = batch["relation_identifiers"]
        if "candidate_mask" in batch: new_current_data["candidate_mask"] = batch["candidate_mask"]
        if "attention_mask" in batch: new_current_data["attention_mask"] = batch["attention_mask"]
        if "input_ids" in batch: new_current_data["input_ids"] = batch["input_ids"]
        elif "inputs" in batch: new_current_data["inputs"] = batch["inputs"]

        # ✅ 4) 이제 inner 호출 (여기서 z_H_last도 받음)
        new_inner_carry, logits, node_logits, z_H_last = self.inner(new_inner_carry, new_current_data)

        # 서브그래프 토큰들에 대한 독립적인 BCE 예측
        # Nodes are concatenated BEFORE text queries in _input_embeddings
        P_len = new_current_data["puzzle_identifiers"].shape[1] if "puzzle_identifiers" in new_current_data else self.inner.puzzle_emb_len
        outputs = {
            "logits": logits,
            "scores": node_logits[:, :P_len],
        }

        # ✅ 6) step update (1-step forward only)
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = torch.ones_like(halted_sync)

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
