# Meson Subprojects 패치 적용 문제 해결 방법

## 문제 상황
수동으로 `patch` 명령으로는 패치가 잘 적용되지만, `meson subprojects update`를 할 때 패치가 적용되지 않는 문제가 발생합니다.

## 문제 원인 분석

### 1. 패치 파일 경로 문제
- **현재 설정**: `subprojects/ggml.wrap`에서 `patch_directory = ggml`, `diff_files = ggml/0001-nntrainer-ggml-patch.patch`
- **실제 파일 위치**: `subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch`
- **결론**: 경로 설정은 올바름 (meson이 `subprojects/packagefiles/`를 기준으로 찾음)

### 2. 패치 적용 메커니즘 차이
- **수동 패치**: 현재 소스 코드 상태를 기준으로 패치 적용
- **meson 패치**: git에서 받은 clean한 소스 코드를 기준으로 패치 적용

## 해결 방법

### 방법 1: 패치 파일 형식 검증

```bash
# 1. 패치 파일 헤더 확인
head -10 subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 2. 패치 파일 구조 확인
grep -n "^diff --git" subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 방법 2: 패치 적용 테스트

```bash
# 1. 임시 디렉토리에서 테스트
mkdir -p test_patch
cd test_patch

# 2. 원본 소스 받기
git clone https://github.com/ggml-org/ggml.git
cd ggml
git checkout 489716ba99ecd51164f79e8c6fec0b5bf634eac9

# 3. 패치 적용 테스트
patch -p1 --dry-run < ../../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 4. 실제 적용
patch -p1 < ../../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 방법 3: 패치 파일 재생성

```bash
# 1. 기존 패치 파일 백업
cp subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch \
   subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch.backup

# 2. 새로운 패치 생성
./create_ggml_patch.sh

# 3. 생성된 임시 디렉토리에서 수정 후 패치 생성
cd temp_ggml_patch
# 필요한 수정 작업...
git add .
git commit -m "nntrainer ggml modifications"
git format-patch HEAD~1 --stdout > ../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 방법 4: wrap 파일 형식 수정

현재 `subprojects/ggml.wrap`:
```ini
[wrap-git]
url = https://github.com/ggml-org/ggml.git
directory = ggml
revision = 489716ba99ecd51164f79e8c6fec0b5bf634eac9
patch_directory = ggml
diff_files = ggml/0001-nntrainer-ggml-patch.patch
method = cmake
```

**권장 수정사항**:
```ini
[wrap-git]
url = https://github.com/ggml-org/ggml.git
directory = ggml
revision = 489716ba99ecd51164f79e8c6fec0b5bf634eac9
patch_directory = ggml
diff_files = ggml/0001-nntrainer-ggml-patch.patch
method = cmake
depth = 1
```

### 방법 5: 패치 파일 내용 검증

```bash
# 1. 패치 파일의 경로 확인
grep "^diff --git" subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 2. 패치 파일의 모든 경로가 올바른지 확인
grep "^@@" subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 방법 6: 강제 재적용

```bash
# 1. 서브프로젝트 완전 정리
rm -rf subprojects/ggml
rm -rf subprojects/packagecache

# 2. 캐시 정리
meson subprojects purge --confirm

# 3. 다시 다운로드 및 적용
meson subprojects download ggml
meson subprojects update ggml
```

### 방법 7: 디버깅 모드로 실행

```bash
# verbose 모드로 실행
meson subprojects update ggml --verbose

# 또는 더 자세한 로그
MESON_LOG_LEVEL=debug meson subprojects update ggml
```

## 일반적인 문제와 해결책

### 1. 패치 파일 경로 문제
- **문제**: 패치 파일 내부의 경로가 맞지 않음
- **해결**: 패치 파일을 다시 생성하거나 경로를 수정

### 2. 패치 적용 순서 문제
- **문제**: 여러 패치가 있을 때 적용 순서가 잘못됨
- **해결**: `diff_files`에서 순서를 올바르게 설정

### 3. 패치 파일 형식 문제
- **문제**: 패치 파일이 올바른 git format-patch 형식이 아님
- **해결**: `git format-patch`로 다시 생성

### 4. 소스 코드 버전 불일치
- **문제**: 패치가 만들어진 소스 버전과 현재 revision이 다름
- **해결**: 올바른 revision으로 패치 재생성

## 자동화된 해결 스크립트

```bash
#!/bin/bash
# fix_ggml_patch.sh

set -e

echo "=== GGML 패치 문제 해결 스크립트 ==="

# 1. 현재 설정 확인
echo "현재 wrap 파일 설정:"
cat subprojects/ggml.wrap

# 2. 패치 파일 검증
echo "패치 파일 검증 중..."
if ! head -10 subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch | grep -q "^From"; then
    echo "⚠️  패치 파일 형식이 올바르지 않습니다"
fi

# 3. 패치 적용 테스트
echo "패치 적용 테스트 중..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

git clone https://github.com/ggml-org/ggml.git
cd ggml
git checkout 489716ba99ecd51164f79e8c6fec0b5bf634eac9

if patch -p1 --dry-run < "$OLDPWD/subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch"; then
    echo "✅ 패치 적용 테스트 성공"
else
    echo "❌ 패치 적용 테스트 실패"
    echo "패치 파일을 다시 생성해야 합니다"
fi

# 4. 정리
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

# 5. 재적용 시도
echo "서브프로젝트 재적용 중..."
rm -rf subprojects/ggml
meson subprojects download ggml
meson subprojects update ggml

echo "완료!"
```

이 방법들을 순서대로 시도해보시면 meson subprojects의 패치 적용 문제를 해결할 수 있습니다.