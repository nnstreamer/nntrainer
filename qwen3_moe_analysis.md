# Qwen3 MoE 모델 분석 및 nntrainer 구현 가이드

## Qwen3 MoE 모델 개요

Qwen3 MoE는 Mixture-of-Experts (MoE) 아키텍처를 사용하는 Transformer 기반 언어 모델입니다.

### 주요 특징
- **총 파라미터**: 30.5B (Qwen3-30B-A3B 기준)
- **활성화 파라미터**: 3.3B
- **레이어 수**: 48
- **어텐션 헤드**: Query 32개, Key-Value 4개 (Grouped Query Attention)
- **전문가(Expert) 수**: 128개
- **활성화 전문가 수**: 8개 (토큰당)
- **컨텍스트 길이**: 32,768 토큰
- **숨김 차원**: 2048
- **중간 차원**: 6144
- **MoE 중간 차원**: 768

## 핵심 컴포넌트 분석

### 1. Qwen3MoeModel (메인 모델)
```
- Token Embedding
- 48개의 Decoder Layer (Qwen3MoeDecoderLayer)
- Final Layer Normalization (RMSNorm)
```

### 2. Qwen3MoeDecoderLayer
```
- Self Attention (Qwen3MoeAttention)
- Input Layer Norm (RMSNorm)
- MLP 또는 MoE Block (조건부)
- Post Attention Layer Norm (RMSNorm)
```

### 3. Qwen3MoeAttention (Multi-Head Attention)
```
- Query, Key, Value Projection (Linear)
- Query/Key Normalization (RMSNorm)
- Rotary Position Embedding (RoPE)
- Grouped Query Attention (GQA)
- Output Projection (Linear)
```

### 4. Qwen3MoeSparseMoeBlock (핵심 MoE 컴포넌트)
```
- Router/Gate (Linear layer: hidden_size → num_experts)
- 128개의 Expert MLP
- Top-K Selection (상위 8개 expert 선택)
- Load Balancing Loss
```

### 5. Qwen3MoeMLP (Expert Network)
```
- Gate Projection (Linear: hidden_size → moe_intermediate_size)
- Up Projection (Linear: hidden_size → moe_intermediate_size)  
- SiLU Activation
- Down Projection (Linear: moe_intermediate_size → hidden_size)
```

### 6. Qwen3MoeRMSNorm
```
- Root Mean Square Layer Normalization
- learnable weight parameter
```

## nntrainer에서 필요한 레이어 기능들

### 이미 사용 가능한 레이어들 ✅

1. **Embedding Layer** (`embedding.h`)
   - Token embedding을 위한 기본 임베딩 레이어

2. **Multi-Head Attention Layer** (`multi_head_attention_layer.h`)
   - 기본 멀티헤드 어텐션 (커스텀 필요)

3. **FC Layer** (`fc_layer.h`)
   - Linear transformation을 위한 완전연결 레이어

4. **Layer Normalization** (`layer_normalization_layer.h`)
   - 정규화 레이어 (RMSNorm으로 수정 필요)

5. **Rotary Embedding** (LLaMA 앱에서 구현됨)
   - `rotary_embedding.h`에서 이미 구현됨

### 새로 구현해야 할 레이어들 ❌

1. **Qwen3MoeRMSNorm Layer**
   - Root Mean Square Normalization
   - LayerNorm과 다른 정규화 방식

2. **MoE Router Layer**
   - 전문가 선택을 위한 게이팅 네트워크
   - Top-K 전문가 선택 로직
   - Load balancing loss 계산

3. **Sparse MoE Block Layer**
   - 여러 전문가(Expert) 네트워크 관리
   - 조건부 계산 (일부 전문가만 활성화)
   - Expert 병렬 처리

4. **Grouped Query Attention (GQA)**
   - Key-Value 헤드 수가 Query 헤드 수보다 적은 어텐션
   - Key-Value 공유 메커니즘

5. **SiLU Activation Layer**
   - Swish/SiLU 활성화 함수
   - `x * sigmoid(x)`

6. **Qwen3MoeMLP Layer**
   - Gate와 Up projection을 가진 MLP
   - Element-wise multiplication

## 구현 우선순위

### Phase 1: 기본 컴포넌트
1. **SiLU Activation Layer** - 가장 간단함
2. **RMSNorm Layer** - LayerNorm 수정으로 구현 가능
3. **GQA Layer** - 기존 Multi-Head Attention 수정

### Phase 2: MLP 컴포넌트  
4. **Qwen3MoeMLP Layer** - SiLU와 FC 레이어 조합
5. **Basic MoE Router** - 단순 라우터부터 시작

### Phase 3: 고급 MoE
6. **Sparse MoE Block** - 완전한 MoE 시스템
7. **Load Balancing** - 효율성 최적화

## 권장 구현 방법

1. **기존 LLaMA 애플리케이션 확장**
   - LLaMA의 Rotary Embedding과 Attention 구조 재사용
   - Multi-Head Attention을 GQA로 수정

2. **점진적 구현**
   - Dense 버전부터 시작 (MoE 없이)
   - 단일 전문가로 MoE 동작 검증
   - 점진적으로 전문가 수 증가

3. **성능 최적화**
   - Expert 병렬 처리
   - 메모리 효율적인 구현
   - Top-K 선택 최적화

## 예상 구현 시간

- **Phase 1**: 2-3주 (기본 레이어들)
- **Phase 2**: 3-4주 (MLP와 기본 MoE)  
- **Phase 3**: 4-6주 (완전한 MoE 시스템)
- **총 예상 시간**: 9-13주

이 분석을 바탕으로 Qwen3 MoE 모델을 nntrainer에서 구현할 수 있는 Application 코드를 작성하겠습니다.