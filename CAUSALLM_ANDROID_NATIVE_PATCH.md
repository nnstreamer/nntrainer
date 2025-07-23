# CausalLM Android Native Build Patch

## 요약

PR #3344의 CausalLM Application을 Android 네이티브 실행 파일로 빌드할 수 있는 패치를 생성했습니다.

**핵심 목표**: Gradle이나 Android Studio 없이 CausalLM의 `main.cpp`를 Android에서 직접 실행 가능한 네이티브 바이너리로 빌드

## 생성된 파일들

```
Applications/CausalLM/android/
├── build_android_native.sh          # 기본 빌드 스크립트
├── build_android_native_advanced.sh # 고급 빌드 스크립트 (권장)
└── README.md                        # 상세 사용법
```

## 주요 특징

### ✅ 간단한 사용법
```bash
export ANDROID_NDK=/path/to/ndk
cd Applications/CausalLM/android
./build_android_native_advanced.sh
cd deploy
./deploy_to_device.sh
```

### ✅ 완전 자동화
- `tools/package_android.sh` 자동 호출
- 모든 dependency 자동 추출 및 링크
- 배포 패키지 자동 생성
- Android 디바이스 배포 스크립트 포함

### ✅ 네이티브 실행
- APK 설치 불필요
- Android 디바이스에서 직접 실행
- `/data/local/tmp/causallm/` 경로에 설치
- 모든 필요한 라이브러리 포함

## 빌드 과정

1. **nntrainer 빌드**: 기존 `package_android.sh` 사용하여 Android용 nntrainer + CausalLM 빌드
2. **Dependency 추출**: `nntrainer_for_android.tar.gz`에서 헤더와 라이브러리 추출
3. **소스 파일 검색**: CausalLM의 모든 필요한 소스 파일 자동 검색
4. **NDK 컴파일**: Android NDK로 ARM64 바이너리 컴파일
5. **배포 패키지**: 실행 파일, 라이브러리, 스크립트 패키징

## 지원 기능

### 모델 지원
- **Llama**: 기본 Llama 모델
- **Qwen3**: Qwen3 1.7b/4b/7b/14b
- **Qwen3-MoE**: Qwen3-MoE 30b-A3b
- **확장 가능**: 새로운 모델 타입 쉽게 추가

### 자동 소스 검색
스크립트가 자동으로 다음 파일들을 찾아 컴파일:
- Core: `main.cpp`, `causal_lm.cpp`, `llm_util.cpp`
- Models: `qwen3_causallm.cpp`, `qwen3_moe_causallm.cpp`
- Layers: `embedding_layer.cpp`, `rms_norm.cpp`, `swiglu.cpp` 등

### 배포 자동화
생성되는 배포 패키지:
- `nntr_causallm_android` - 실행 파일
- `*.so` - 필요한 모든 라이브러리
- `deploy_to_device.sh` - 디바이스 배포 스크립트
- `run_causallm.sh` - 디바이스에서 실행 스크립트
- 상세한 사용법 문서

## 사용 시나리오

### 1. 개발자용
```bash
# 빌드
./build_android_native_advanced.sh

# 테스트
cd deploy
./deploy_to_device.sh
adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/models/qwen3-4b'
```

### 2. 연구자용
- 다양한 모델로 실험
- 성능 측정 및 비교
- 커스텀 설정으로 테스트

### 3. 프로덕션용
- 서버에서 자동 빌드
- CI/CD 파이프라인 통합
- 대량 디바이스 배포

## 기술적 특징

### 컴파일러 설정
- **Target**: `aarch64-linux-android` (ARM64)
- **API Level**: 30 (Android 11+)
- **Optimization**: `-O3` 최적화
- **OpenMP**: 멀티스레딩 지원
- **Static Linking**: 의존성 최소화

### 에러 처리
- 환경 변수 검증
- 소스 파일 존재 확인
- 컴파일 실패 시 대안 시도
- 상세한 에러 메시지

### 호환성
- Android 7.0+ (API 24+)
- ARM64 아키텍처
- USB 디버깅 지원 디바이스

## 장점

1. **단순함**: 복잡한 Android 앱 구조 불필요
2. **직접성**: APK 없이 바로 실행
3. **자동화**: 모든 과정 스크립트로 자동화
4. **완전성**: 실행에 필요한 모든 것 포함
5. **확장성**: 새로운 모델/기능 쉽게 추가

## 사용 예시

```bash
# 환경 설정
export ANDROID_NDK=/opt/android-ndk-r25c

# 빌드
cd Applications/CausalLM/android
./build_android_native_advanced.sh

# 결과 확인
ls deploy/
# nntr_causallm_android  deploy_to_device.sh  README.md  *.so

# 배포
cd deploy
./deploy_to_device.sh

# 모델 복사
adb push ~/models/qwen3-4b /sdcard/causallm_models/

# 실행
adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/qwen3-4b'
```

## 결론

이 패치는 PR #3344의 CausalLM을 Android에서 간단하게 빌드하고 실행할 수 있는 완전한 솔루션을 제공합니다. 

- **기존 `package_android.sh` 활용**: 검증된 빌드 시스템 재사용
- **최소한의 복잡성**: Gradle/Android Studio 불필요
- **최대한의 자동화**: 원클릭 빌드 & 배포
- **완전한 문서화**: 상세한 사용법과 문제 해결 가이드

개발자들이 CausalLM을 Android 환경에서 쉽게 테스트하고 배포할 수 있도록 하는 것이 이 패치의 목표입니다.