# FloatTensor::apply 함수 OpenMP 병렬화 분석

## 현재 상황 분석

### 1. OpenMP 지원 현황
- ✅ 프로젝트에서 OpenMP 이미 사용 중
- ✅ `meson_options.txt`에서 `enable-openmp` 기본값 true
- ✅ `float_tensor.cpp`의 `topK` 함수에서 이미 OpenMP 사용
- ✅ ARM NEON 구현에서도 OpenMP 활용

### 2. apply 함수 구조 분석

```cpp
Tensor &FloatTensor::apply(std::function<float(float)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  // 케이스 1: 연속 메모리 (contiguous)
  if (contiguous && output.getContiguous()) {
    const float *data = (float *)getData();
    float *rdata = output.getData<float>();
    std::transform(data, data + size(), rdata, f);
  }
  
  // 케이스 2: stride[3] == 1 (3중 루프)
  else if (strides[3] == 1 && output.getStrides()[3] == 1) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          // width 단위로 std::transform 적용
        }
      }
    }
  }
  
  // 케이스 3: 일반 케이스 (4중 루프)
  else {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            // 개별 요소 처리
          }
        }
      }
    }
  }
}
```

## OpenMP 병렬화 가능성 분석

### 케이스 1: 연속 메모리 (Contiguous)

#### 현재 구현
```cpp
std::transform(data, data + size(), rdata, f);
```

#### OpenMP 병렬화 적용
```cpp
const size_t total_size = size();
#pragma omp parallel for schedule(static)
for (size_t i = 0; i < total_size; ++i) {
  rdata[i] = f(data[i]);
}
```

**성능 향상 예상도: ⭐⭐⭐⭐⭐ (매우 높음)**
- 완전히 독립적인 연산
- 메모리 접근 패턴이 순차적
- 캐시 지역성 우수

### 케이스 2: stride[3] == 1 (3중 루프)

#### 현재 구현
```cpp
for (unsigned int b = 0; b < batch(); ++b) {
  for (unsigned int c = 0; c < channel(); ++c) {
    for (unsigned int h = 0; h < height(); ++h) {
      std::transform(in_data, in_data + width(), out_data, f);
    }
  }
}
```

#### OpenMP 병렬화 적용 (옵션 1: 외부 루프)
```cpp
#pragma omp parallel for collapse(3) schedule(static)
for (unsigned int b = 0; b < batch(); ++b) {
  for (unsigned int c = 0; c < channel(); ++c) {
    for (unsigned int h = 0; h < height(); ++h) {
      std::transform(in_data, in_data + width(), out_data, f);
    }
  }
}
```

#### OpenMP 병렬화 적용 (옵션 2: 내부 루프)
```cpp
for (unsigned int b = 0; b < batch(); ++b) {
  for (unsigned int c = 0; c < channel(); ++c) {
    for (unsigned int h = 0; h < height(); ++h) {
      #pragma omp parallel for schedule(static)
      for (unsigned int w = 0; w < width(); ++w) {
        out_data[w] = f(in_data[w]);
      }
    }
  }
}
```

**성능 향상 예상도: ⭐⭐⭐⭐ (높음)**
- 독립적인 연산 
- 메모리 접근 패턴 양호
- collapse(3) 사용으로 더 많은 병렬성 확보

### 케이스 3: 일반 케이스 (4중 루프)

#### 현재 구현
```cpp
for (unsigned int b = 0; b < batch(); ++b) {
  for (unsigned int c = 0; c < channel(); ++c) {
    for (unsigned int h = 0; h < height(); ++h) {
      for (unsigned int w = 0; w < width(); ++w) {
        output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
      }
    }
  }
}
```

#### OpenMP 병렬화 적용
```cpp
#pragma omp parallel for collapse(4) schedule(static)
for (unsigned int b = 0; b < batch(); ++b) {
  for (unsigned int c = 0; c < channel(); ++c) {
    for (unsigned int h = 0; h < height(); ++h) {
      for (unsigned int w = 0; w < width(); ++w) {
        output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
      }
    }
  }
}
```

**성능 향상 예상도: ⭐⭐⭐ (중간)**
- 독립적인 연산이지만 getValue/setValue 오버헤드
- 메모리 접근 패턴이 불규칙할 수 있음
- collapse(4)로 최대 병렬성 확보

## apply 함수 사용 현황

### 주요 사용 함수들
- `multiply(float)`, `divide(float)`, `add(float)`, `subtract(float)` - 스칼라 연산
- `abs()`, `pow()`, `sqrt()`, `erf()` - 수학 함수
- `sin()`, `cos()`, `tan()` - 삼각함수
- `inv_sqrt()` - 역제곱근
- 각종 활성화 함수들 (softmax 등)

### 사용 빈도 분석
- 거의 모든 요소별(element-wise) 연산에서 사용
- 활성화 함수 적용 시 대량 사용
- 수학 함수 적용 시 집중적 사용

## 예상 성능 향상 효과

### 1. 연산 특성에 따른 효과
- **간단한 연산** (add, multiply): 3-5배 성능 향상
- **복잡한 연산** (sin, cos, sqrt): 2-4배 성능 향상  
- **매우 복잡한 연산** (사용자 정의 함수): 1.5-3배 성능 향상

### 2. 텐서 크기에 따른 효과
- **큰 텐서** (1M+ elements): 최대 성능 향상
- **중간 텐서** (10K-1M elements): 좋은 성능 향상
- **작은 텐서** (< 10K elements): 미미한 성능 향상 (오버헤드 가능)

### 3. 하드웨어에 따른 효과
- **멀티코어 CPU**: 코어 수에 비례한 성능 향상
- **하이퍼스레딩**: 추가 10-30% 성능 향상
- **NUMA 시스템**: 메모리 지역성 고려 필요

## 병렬화 시 고려사항

### 1. 스레드 안전성
- `std::function<float(float)>` 함수 객체의 스레드 안전성 확인 필요
- 대부분의 수학 함수는 스레드 안전
- 사용자 정의 함수는 주의 필요

### 2. 메모리 대역폭
- 병렬 처리 시 메모리 대역폭이 병목이 될 수 있음
- 간단한 연산일수록 메모리 바운드 가능성 높음

### 3. 최적화 전략
- 작은 텐서에 대해서는 OpenMP 비활성화
- 임계값 설정 (`size() < threshold`)
- 스케줄링 정책 최적화

## 구현 권장사항

### 1. 적응형 병렬화
```cpp
const size_t PARALLEL_THRESHOLD = 10000;
if (size() >= PARALLEL_THRESHOLD) {
  // OpenMP 병렬화 적용
} else {
  // 기존 순차 처리
}
```

### 2. 스케줄링 최적화
- **간단한 연산**: `schedule(static)`
- **복잡한 연산**: `schedule(dynamic)` 또는 `schedule(guided)`

### 3. 중첩 수준 최적화
- **큰 외부 차원**: `collapse(2)` 또는 `collapse(3)`
- **작은 외부 차원**: 내부 루프만 병렬화

## 결론

**OpenMP 병렬화로 인한 성능 향상 가능성: 매우 높음**

### 주요 이점
1. **대폭적인 성능 향상**: 2-5배 성능 향상 예상
2. **광범위한 영향**: 거의 모든 요소별 연산에 적용
3. **구현 용이성**: 이미 OpenMP 인프라 구축됨
4. **안정성**: 독립적인 연산으로 안전한 병렬화

### 추천 우선순위
1. **1순위**: 케이스 1 (contiguous) - 가장 높은 성능 향상
2. **2순위**: 케이스 2 (stride[3] == 1) - 좋은 성능 향상
3. **3순위**: 케이스 3 (일반) - 중간 성능 향상

**결론적으로, apply 함수의 OpenMP 병렬화는 매우 가치 있는 최적화이며, 특히 딥러닝 워크로드에서 큰 성능 향상을 가져올 것으로 예상됩니다.**