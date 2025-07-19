# Accurate Q4_K PyTorch Weight Converter

llama.cpp의 실제 gguf-py 구현을 기반으로 한 정확하고 가독성 있는 Q4_K 양자화 변환기입니다.

## 🎯 특징

- **정확성**: llama.cpp의 실제 Q4_K 구현에 기반
- **가독성**: 명확한 코드 구조와 상세한 주석
- **호환성**: llama.cpp와 호환되는 Q4_K 형식
- **효율성**: 4.5 bits per weight로 높은 압축률
- **사용성**: 간단한 CLI 인터페이스 제공

## 📋 Q4_K 형식 사양

웹 검색을 통해 확인한 llama.cpp의 Q4_K 구조:

```
- 슈퍼블록: 256개 요소 (QK_K = 256)
- 서브블록: 8개 블록 × 32개 요소
- 스케일: 6비트 양자화 (0-63 범위)
- 최소값: 6비트 양자화 (0-63 범위)
- 공식: w = q * block_scale(6-bit) + block_min(6-bit)
- 효율성: 4.5 bits per weight
```

## 🚀 설치

```bash
# 필수 패키지 설치
pip install torch numpy

# 선택적 패키지 (safetensors 지원)
pip install safetensors
```

## 💻 사용법

### 1. 명령줄 인터페이스

```bash
# 기본 사용법
python simple_q4k_interface.py --model model.pth --output model_q4k.npz

# 모델 이름 지정
python simple_q4k_interface.py --model model.pth --output model_q4k.npz --name "My Model"

# safetensors 파일 변환
python simple_q4k_interface.py --model model.safetensors --output model_q4k.npz

# 양자화된 모델 검증
python simple_q4k_interface.py --verify model_q4k.npz

# 원본과 양자화된 모델 크기 비교
python simple_q4k_interface.py --compare model.pth model_q4k.npz

# 자세한 로그 출력
python simple_q4k_interface.py --model model.pth --output model_q4k.npz --verbose
```

### 2. Python API

```python
from simple_q4k_interface import quantize_model
from accurate_q4k_converter import save_q4k_model, convert_tensor_to_q4k

# 간단한 모델 변환
success = quantize_model("model.pth", "model_q4k.npz", "My Model")

# PyTorch 모델 직접 변환
import torch
model = torch.load("model.pth")
save_q4k_model(model, "model_q4k.npz", "My Model")

# 개별 텐서 변환
tensor = torch.randn(1024, 512, dtype=torch.float32)
result = convert_tensor_to_q4k(tensor)
```

### 3. 고급 사용법

```python
from accurate_q4k_converter import (
    quantize_row_q4_k_accurate, 
    save_q4k_model, 
    load_q4k_model
)

# 256개 요소 블록 직접 양자화
import numpy as np
block = np.random.randn(256).astype(np.float32)
quantized_block = quantize_row_q4_k_accurate(block)

# 양자화된 모델 로드
data = load_q4k_model("model_q4k.npz")
metadata = data['metadata']
quantized_tensors = data['quantized_tensors']
```

## 📊 성능 비교

일반적인 압축 성능:

| 모델 크기 | 원본 (FP32) | Q4_K 양자화 | 압축률 | 크기 절약 |
|-----------|-------------|-------------|--------|-----------|
| 7B 파라미터 | ~28GB | ~4GB | 7.0x | 85.7% |
| 13B 파라미터 | ~52GB | ~7.5GB | 6.9x | 85.6% |
| 30B 파라미터 | ~120GB | ~17GB | 7.1x | 85.8% |

## 📁 파일 구조

```
.
├── accurate_q4k_converter.py    # 핵심 Q4_K 변환 로직
├── simple_q4k_interface.py      # 사용하기 쉬운 CLI 인터페이스
└── README.md                     # 이 파일
```

## ⚠️ 주의사항

1. **입력 형식**: float32, float16, bfloat16 텐서만 지원
2. **메모리**: 큰 모델의 경우 충분한 RAM 필요
3. **정확성**: 양자화로 인한 약간의 정확도 손실 가능
4. **호환성**: llama.cpp Q4_K 형식과 호환되도록 설계

## 🔧 기술적 세부사항

### Q4_K 블록 구조

```python
# 각 Q4_K 블록 (256개 요소)의 구조:
block_data = {
    'd_scale': float16,      # 2 bytes - 스케일의 전역 스케일
    'd_min': float16,        # 2 bytes - 최소값의 전역 스케일  
    'scales_packed': bytes,  # 12 bytes - 8개 6비트 스케일
    'quantized_values': bytes # 128 bytes - 4비트 양자화 값들
}
# 총 크기: 144 bytes (256 weights → 4.5 bits per weight)
```

### 양자화 공식

```python
# 각 서브블록에 대해:
scale = (max_val - min_val) / 15.0  # 4비트 범위: 0-15
quantized_value = round((original_value - min_val) / scale)
quantized_value = clip(quantized_value, 0, 15)

# 역양자화:
reconstructed_value = min_val + quantized_value * scale
```

## 🐛 문제 해결

### 일반적인 오류

1. **"Input must have exactly 256 elements"**
   - 텐서가 자동으로 256의 배수로 패딩됩니다
   - 내부적으로 처리되므로 사용자는 신경 쓸 필요 없음

2. **"No float tensors found to quantize"**
   - 모델에 float32/float16/bfloat16 텐서가 없음
   - 모델 파일 형식 확인 필요

3. **메모리 부족 오류**
   - 큰 모델의 경우 배치 처리 고려
   - 시스템 RAM 확인

### 성능 최적화

```python
# 큰 모델의 경우 텐서별로 처리
for name, tensor in model.state_dict().items():
    if tensor.dtype in [torch.float32, torch.float16]:
        result = convert_tensor_to_q4k(tensor)
        # 개별 저장 또는 스트리밍 처리
```

## 📚 참고 자료

- [llama.cpp Q4_K 구현](https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305)
- [GGUF 양자화 설명](https://github.com/huggingface/huggingface.js/blob/main/packages/gguf/src/quant-descriptions.ts)
- [GGML 양자화 타입](https://blog.csdn.net/chengyq116/article/details/145433926)

## 🤝 기여

버그 리포트나 개선 제안은 언제든 환영합니다!

## 📄 라이선스

MIT License - 자유롭게 사용하세요.

---

**참고**: 이 구현은 llama.cpp의 실제 gguf-py 구현을 웹 검색을 통해 분석하여 작성되었습니다. 정확성을 위해 지속적으로 업데이트됩니다.
