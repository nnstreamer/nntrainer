# CausalLM Android Native Build

PR #3344의 CausalLM Application을 Android 네이티브 실행 파일로 빌드하는 스크립트입니다.

## 개요

- **목적**: CausalLM main.cpp를 Android에서 직접 실행 가능한 네이티브 바이너리로 빌드
- **방식**: Gradle/Android Studio 없이 순수 NDK 사용
- **결과**: Android 디바이스에서 직접 실행 가능한 `nntr_causallm_android` 바이너리

## 사용법

### 1. 환경 설정
```bash
export ANDROID_NDK=/path/to/your/android-ndk
```

### 2. 빌드 실행
```bash
cd Applications/CausalLM/android

# 기본 빌드
./build_android_native.sh

# 또는 고급 빌드 (모든 dependency 자동 처리)
./build_android_native_advanced.sh
```

### 3. 디바이스에 배포
```bash
cd deploy
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

1. **nntrainer 빌드**: `tools/package_android.sh` 호출하여 Android용 nntrainer 빌드
2. **dependency 추출**: 빌드된 라이브러리들을 추출
3. **CausalLM 컴파일**: NDK를 사용하여 main.cpp와 모든 dependency를 컴파일
4. **배포 패키지 생성**: 실행 파일, 라이브러리, 배포 스크립트 생성

## 파일 구조

```
android/
├── build_android_native.sh          # 기본 빌드 스크립트
├── build_android_native_advanced.sh # 고급 빌드 스크립트 (권장)
├── README.md                        # 이 파일
├── android_deps/                    # 추출된 dependency (빌드 후 생성)
├── build/                           # 빌드 결과물 (빌드 후 생성)
└── deploy/                          # 배포 패키지 (빌드 후 생성)
    ├── nntr_causallm_android        # 실행 파일
    ├── *.so                         # 필요한 라이브러리들
    ├── deploy_to_device.sh          # 디바이스 배포 스크립트
    └── README.md                    # 배포 가이드
```

## 필요 조건

- **Android NDK**: 환경변수 `ANDROID_NDK` 설정 필요
- **ADB**: Android 디바이스 연결을 위해 필요
- **CausalLM 소스**: PR #3344의 CausalLM 구현이 필요
- **모델 파일**: config.json, nntr_config.json, tokenizer.json, *.bin 등

## 장점

- ✅ **간단함**: Gradle, Android Studio 불필요
- ✅ **직접 실행**: APK 설치 없이 바로 실행
- ✅ **자동화**: 모든 dependency 자동 처리
- ✅ **완전한 패키지**: 실행에 필요한 모든 파일 포함

## 사용 예시

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