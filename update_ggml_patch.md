# GGML 서브프로젝트 패치 업데이트 방법

## 1. 기존 패치 파일 수정 (권장)

### 단계 1: 기존 패치 파일 확인
```bash
# 현재 패치 파일 위치
ls -la subprojects/packagefiles/ggml/
cat subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 단계 2: 수정된 코드를 기존 패치 파일에 반영
```bash
# 1. 기존 패치 파일을 백업
cp subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch \
   subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch.backup

# 2. 패치 파일을 수정 (텍스트 에디터로 직접 편집)
# - 기존 패치 내용에 새로운 변경사항 추가
# - 혹은 완전히 새로운 패치로 대체
```

### 단계 3: 서브프로젝트 재빌드
```bash
# 기존 서브프로젝트 정리
rm -rf subprojects/ggml/

# 서브프로젝트 다시 설정
meson subprojects download ggml
meson subprojects update ggml
```

## 2. 새로운 패치 파일 생성

### 단계 1: 원본 ggml 소스 받기
```bash
# 원본 ggml 클론
git clone https://github.com/ggml-org/ggml.git temp_ggml
cd temp_ggml

# 특정 리비전으로 체크아웃 (ggml.wrap에 있는 revision 사용)
git checkout 489716ba99ecd51164f79e8c6fec0b5bf634eac9
```

### 단계 2: 수정사항 적용
```bash
# 필요한 수정사항을 적용
# 예: 파일 편집, 새로운 기능 추가 등
```

### 단계 3: 패치 파일 생성
```bash
# 변경사항 커밋
git add .
git commit -m "nntrainer ggml modifications"

# 패치 파일 생성
git format-patch HEAD~1 --stdout > ../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 임시 디렉토리 정리
cd ..
rm -rf temp_ggml
```

## 3. 다중 패치 파일 사용

### ggml.wrap 파일 수정
```ini
[wrap-git]
url = https://github.com/ggml-org/ggml.git
directory = ggml
revision = 489716ba99ecd51164f79e8c6fec0b5bf634eac9
patch_directory = ggml
diff_files = ggml/0001-nntrainer-ggml-patch.patch, ggml/0002-additional-changes.patch
method = cmake
```

### 추가 패치 파일 생성
```bash
# 두 번째 패치 파일 생성
# (첫 번째 패치가 적용된 상태에서 추가 변경사항에 대한 패치)
```

## 4. 패치 적용 확인

### 단계 1: 서브프로젝트 재설정
```bash
# 기존 서브프로젝트 삭제
rm -rf subprojects/ggml/

# 서브프로젝트 다운로드 및 패치 적용
meson subprojects download ggml
meson subprojects update ggml
```

### 단계 2: 패치 적용 확인
```bash
# 패치가 올바르게 적용되었는지 확인
ls -la subprojects/ggml/
# 수정된 파일들이 올바르게 패치되었는지 확인
```

### 단계 3: 빌드 테스트
```bash
# 빌드 테스트
meson setup build
meson compile -C build
```

## 5. 문제 해결

### 패치 적용 실패 시
```bash
# 수동으로 패치 적용 테스트
cd subprojects/ggml/
patch -p1 < ../packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

### 패치 파일 형식 확인
```bash
# 패치 파일의 형식이 올바른지 확인
head -20 subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch
```

## 6. 자동화 스크립트 (선택사항)

### update_ggml_patch.sh
```bash
#!/bin/bash

# 기존 서브프로젝트 정리
rm -rf subprojects/ggml/

# 서브프로젝트 재설정
meson subprojects download ggml
meson subprojects update ggml

# 빌드 테스트
if [ -d "build" ]; then
    meson compile -C build
else
    meson setup build
    meson compile -C build
fi

echo "GGML 패치 업데이트 완료"
```

## 주의사항

1. **백업**: 기존 패치 파일을 항상 백업해 두세요
2. **테스트**: 패치 적용 후 반드시 빌드 테스트를 수행하세요
3. **버전 관리**: 패치 파일도 git으로 관리하여 변경사항을 추적하세요
4. **의존성**: 새로운 패치가 기존 코드와 호환되는지 확인하세요

이 방법을 통해 ggml 서브프로젝트의 수정사항을 패치로 적용하고 meson git wrapping 설정을 업데이트할 수 있습니다.