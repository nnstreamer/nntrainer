# CausalLM Android Native Build (Meson-Powered)

PR #3344의 CausalLM Application을 Android 네이티브 실행 파일로 빌드하는 스크립트입니다.

## 개요

- **목적**: CausalLM main.cpp를 Android에서 직접 실행 가능한 네이티브 바이너리로 빌드
- **방식**: **Meson 빌드 시스템**을 최대한 활용하여 통합 빌드
- **특징**: 크로스 컴파일, 다중 아키텍처 지원, 고급 빌드 옵션
- **결과**: Android 디바이스에서 직접 실행 가능한 `nntr_causallm_android` 바이너리

## 사용법

### 1. 환경 설정
```bash
export ANDROID_NDK=/path/to/your/android-ndk
```

### 2. 빌드 실행
```bash
cd Applications/CausalLM/android

# 🚀 Meson 기본 빌드 (권장)
./build_android_meson.sh

# 🔧 Meson 고급 빌드 (다양한 옵션 지원)
./build_android_meson_advanced.sh

# 📦 기존 NDK 빌드 (호환성용)
./build_android_native.sh
./build_android_native_advanced.sh
```

### 3. Meson 고급 빌드 옵션
```bash
# 다양한 아키텍처 지원
./build_android_meson_advanced.sh --arch aarch64    # ARM64 (기본)
./build_android_meson_advanced.sh --arch armv7      # ARM32
./build_android_meson_advanced.sh --arch x86_64     # Intel 64-bit

# 빌드 타입 및 옵션
./build_android_meson_advanced.sh --debug           # 디버그 빌드
./build_android_meson_advanced.sh --clean           # 클린 빌드
./build_android_meson_advanced.sh --verbose         # 상세 출력

# API 레벨 및 병렬 빌드
./build_android_meson_advanced.sh --api-level 29 --jobs 8

# Meson 옵션 직접 전달
./build_android_meson_advanced.sh -Denable-opencl=false -Denable-fp16=true
```

### 4. 디바이스에 배포
```bash
# Meson 빌드 결과물 배포
cd deploy_meson          # 기본 Meson 빌드
cd deploy_meson_aarch64   # 고급 Meson 빌드 (아키텍처별)

# 기존 빌드 결과물 배포
cd deploy                 # NDK 빌드

./deploy_to_device.sh
```

### 4. 모델 파일 복사
```bash
adb push /path/to/your/model /sdcard/causallm_models/model_name
```

### 5. 실행
```bash
adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/model_name'
```

## 빌드 과정

### 🚀 Meson 빌드 (권장)
1. **Meson 설정**: Android용 크로스 컴파일 설정 자동 생성
2. **통합 빌드**: CausalLM을 포함한 전체 nntrainer 프로젝트를 Meson으로 빌드
3. **설치 및 패키징**: Meson install로 Android용 바이너리 생성
4. **배포 패키지**: 아키텍처별 최적화된 배포 패키지 생성

### 📦 기존 NDK 빌드 (호환성)
1. **nntrainer 빌드**: `tools/package_android.sh` 호출하여 Android용 nntrainer 빌드
2. **dependency 추출**: 빌드된 라이브러리들을 추출
3. **CausalLM 컴파일**: NDK를 사용하여 main.cpp와 모든 dependency를 컴파일
4. **배포 패키지 생성**: 실행 파일, 라이브러리, 배포 스크립트 생성

## 파일 구조

```
android/
├── 🚀 Meson 빌드 스크립트
│   ├── build_android_meson.sh              # Meson 기본 빌드
│   ├── build_android_meson_advanced.sh     # Meson 고급 빌드 (권장)
│   └── android-cross-file.txt              # Meson 크로스 컴파일 템플릿
├── 📦 기존 NDK 빌드 스크립트 (호환성)
│   ├── build_android_native.sh             # NDK 기본 빌드
│   └── build_android_native_advanced.sh    # NDK 고급 빌드
├── README.md                               # 이 파일
├── 🏗️ 빌드 결과물 (빌드 후 생성)
│   ├── android_deps/                       # NDK 빌드용 dependency
│   ├── build/                              # NDK 빌드 결과물
│   ├── deploy/                             # NDK 배포 패키지
│   ├── deploy_meson/                       # Meson 기본 배포 패키지
│   └── deploy_meson_aarch64/               # Meson 아키텍처별 배포 패키지
└── 📱 배포 패키지 구조 (각 deploy 디렉토리)
    ├── nntr_causallm_android               # 실행 파일
    ├── *.so                                # 필요한 라이브러리들
    ├── deploy_to_device.sh                 # 디바이스 배포 스크립트
    ├── build_info.txt                      # 빌드 정보 (Meson만)
    └── README.md                           # 배포 가이드
```

## 필요 조건

- **Android NDK**: 환경변수 `ANDROID_NDK` 설정 필요
- **ADB**: Android 디바이스 연결을 위해 필요
- **CausalLM 소스**: PR #3344의 CausalLM 구현이 필요
- **모델 파일**: config.json, nntr_config.json, tokenizer.json, *.bin 등

## 장점

### 🚀 Meson 빌드의 장점
- ✅ **통합 빌드**: nntrainer 프로젝트와 완전 통합
- ✅ **크로스 컴파일**: 자동 크로스 컴파일 설정 생성
- ✅ **다중 아키텍처**: ARM64, ARM32, x86_64, x86 지원
- ✅ **빌드 최적화**: 병렬 빌드, 캐싱, 증분 빌드
- ✅ **의존성 관리**: Meson의 강력한 의존성 해결
- ✅ **디버깅 지원**: 디버그/릴리즈 빌드 쉽게 전환

### 📦 공통 장점
- ✅ **간단함**: Gradle, Android Studio 불필요
- ✅ **직접 실행**: APK 설치 없이 바로 실행
- ✅ **자동화**: 모든 dependency 자동 처리
- ✅ **완전한 패키지**: 실행에 필요한 모든 파일 포함

## 사용 예시

### 🚀 Meson 빌드 (권장)
```bash
# 1. 환경 설정
export ANDROID_NDK=/opt/android-ndk-r25c

# 2. Meson 고급 빌드
cd Applications/CausalLM/android
./build_android_meson_advanced.sh --clean --arch aarch64

# 3. 배포
cd deploy_meson_aarch64
./deploy_to_device.sh

# 4. 모델 복사
adb push ~/models/qwen3-4b /sdcard/causallm_models/

# 5. 실행
adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/qwen3-4b'

# 6. 빌드 정보 확인
adb shell '/data/local/tmp/causallm/run_causallm.sh --info'
```

### 📦 기존 NDK 빌드 (호환성)
```bash
# 1. 빌드
export ANDROID_NDK=/opt/android-ndk-r25c
cd Applications/CausalLM/android
./build_android_native_advanced.sh

# 2. 배포
cd deploy
./deploy_to_device.sh

# 3. 모델 복사
adb push ~/models/qwen3-4b /sdcard/causallm_models/

# 4. 실행
adb shell '/data/local/tmp/causallm/run_causallm.sh /sdcard/causallm_models/qwen3-4b'
```

## 문제 해결

### 빌드 실패
- NDK 경로 확인: `echo $ANDROID_NDK`
- CausalLM 소스 확인: `ls ../main.cpp`

### 실행 실패
- 디바이스 연결 확인: `adb devices`
- 권한 문제: `adb shell chmod +x /data/local/tmp/causallm/*`
- 라이브러리 문제: 모든 .so 파일이 배포되었는지 확인

### 성능 이슈
- 더 작은 모델 사용
- `nntr_config.json`에서 `batch_size`, `max_seq_len` 조정
- 디바이스 메모리 확인

이 방식으로 CausalLM을 Android에서 네이티브 실행 파일로 간단하게 빌드하고 실행할 수 있습니다.