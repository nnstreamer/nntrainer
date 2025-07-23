# Android Build Support Patch for Applications/CausalLM

이 패치는 nntrainer의 CausalLM 애플리케이션에 대한 Android 빌드 지원을 활성화합니다. 참조 패치 스타일 `ae24db6e9c018a819841f5884defb2c9c1fc3a14`을 따라 **간단한 executable 기반 접근법**을 사용합니다.

## 주요 설계 원칙

### 1. 전체 빌드 시스템과 분리
- 메인 meson 빌드에서는 Android 앱을 빌드하지 않음 (기존 warning 유지)
- CausalLM은 별도의 Android 빌드 스크립트로 독립적으로 빌드
- 각 애플리케이션이 개별적으로 Android 빌드 가능

### 2. 단순한 Executable 구조  
- **JNI 불필요**: 복잡한 JNI wrapper 대신 단순한 C++ executable
- **main.cpp 중심**: 하나의 실행 파일로 네이티브와 Android 모두 지원
- **크로스 플랫폼**: 동일한 main.cpp가 Linux와 Android에서 모두 동작

## 구현된 변경사항

### 1. 핵심 파일들

#### `Applications/CausalLM/main.cpp`
- 네이티브와 Android 모두에서 동작하는 단일 executable
- Android 로깅 지원 (`#ifdef ANDROID`)
- 명령행 인자 처리 및 모델 경로 지원
- TODO 주석으로 실제 CausalLM 구현 연결 지점 표시

#### `Applications/CausalLM/meson_android.build`
- Android 전용 meson 빌드 설정
- 독립적인 프로젝트로 구성
- nntrainer 의존성 선택적 처리
- Android 특화 컴파일러 플래그

#### `Applications/CausalLM/build_android.sh`
- 완전 자동화된 Android 빌드 스크립트
- NDK 설정 및 크로스 컴파일 파일 자동 생성
- 다양한 ABI 지원 (arm64-v8a, armeabi-v7a, x86, x86_64)
- 빌드 완료 후 Android 디바이스 배포 가이드 제공

### 2. 지원 파일들

#### `Applications/CausalLM/test_build.sh`
- 빌드 구조 검증용 테스트 스크립트
- CausalLM 구현 없이도 빌드 테스트 가능

#### `Applications/CausalLM/README.md`
- 완전한 Android 빌드 가이드
- 네이티브 빌드와 Android 빌드 사용법
- 트러블슈팅 가이드

## 사용법

### Android 빌드
```bash
# 환경 변수 설정
export ANDROID_NDK_ROOT=/path/to/android-ndk
export ANDROID_ABI=arm64-v8a
export ANDROID_API_LEVEL=21

# 빌드 실행
cd Applications/CausalLM
./build_android.sh

# Android 디바이스에 배포
adb push build_android/package/bin/nntr_causallm_android /data/local/tmp/
adb shell chmod +x /data/local/tmp/nntr_causallm_android
adb shell /data/local/tmp/nntr_causallm_android
```

### 네이티브 빌드
```bash
# 메인 nntrainer 빌드에 포함됨
meson setup builddir -Dplatform=none
meson compile -C builddir
./builddir/Applications/CausalLM/nntr_causallm
```

## 장점

### 1. 단순성
- **JNI 불필요**: 복잡한 Java-C++ 바인딩 없음
- **단일 소스**: main.cpp 하나로 모든 플랫폼 지원
- **최소 의존성**: Android NDK만 있으면 빌드 가능

### 2. 독립성
- **분리된 빌드**: 메인 빌드 시스템에 영향 없음
- **개별 관리**: CausalLM만 독립적으로 Android 빌드 가능
- **확장성**: 다른 애플리케이션도 동일한 패턴 적용 가능

### 3. 유지보수성
- **참조 스타일 준수**: 기존 패치 스타일과 일관성
- **명확한 구조**: 빌드 스크립트와 설정 파일이 명확히 분리
- **문서화**: 완전한 사용법 및 트러블슈팅 가이드

## 디렉토리 구조

```
Applications/CausalLM/
├── main.cpp                  # 크로스플랫폼 executable 소스
├── meson.build              # 네이티브 빌드용
├── meson_android.build      # Android 빌드용  
├── build_android.sh         # Android 빌드 스크립트
├── test_build.sh           # 테스트 빌드 스크립트
├── README.md               # 완전한 사용 가이드
├── layers/                 # 커스텀 레이어 (향후)
├── lib/android/           # Android 전용 라이브러리
└── ANDROID_BUILD_PATCH_SUMMARY.md # 이 문서
```

## 향후 확장

1. **다른 애플리케이션**: 동일한 패턴으로 다른 Applications도 Android 지원 추가
2. **iOS 지원**: 유사한 구조로 iOS 빌드 스크립트 추가 가능
3. **UI 통합**: 필요시 Android UI와 연동 가능
4. **성능 최적화**: Android 특화 최적화 추가 가능

이 패치는 복잡한 JNI 구조 없이도 CausalLM을 Android에서 실행할 수 있게 하며, 참조 패치의 스타일을 충실히 따라 간단하고 유지보수하기 쉬운 구조를 제공합니다.