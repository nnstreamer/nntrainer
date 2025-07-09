# GGML 서브프로젝트 패치 문제 해결 가이드

## 문제 상황
- 수동으로 `patch` 명령은 잘 작동하지만 `meson subprojects update`시 패치가 적용되지 않는 문제

## 빠른 해결 방법

### 1. 자동 진단 및 해결
```bash
# 문제 진단 및 자동 해결
./fix_ggml_patch.sh
```

### 2. 수동 해결 (문제가 계속될 경우)
```bash
# 서브프로젝트 정리
rm -rf subprojects/ggml
rm -rf subprojects/packagecache

# 재다운로드 및 패치 적용
meson subprojects download ggml
meson subprojects update ggml
```

### 3. 패치 파일 재생성 (패치 파일 자체에 문제가 있을 경우)
```bash
# 패치 작업 환경 준비
./create_ggml_patch.sh

# 생성된 임시 디렉토리에서 수정 작업
cd temp_ggml_patch
# ... 필요한 수정 작업 수행 ...

# 패치 파일 생성
git add .
git commit -m "nntrainer ggml modifications"
git format-patch HEAD~1 --stdout > ../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 패치 적용
cd ..
./update_ggml_patch.sh

# 정리
rm -rf temp_ggml_patch
```

## 제공된 스크립트 설명

| 스크립트 | 용도 |
|----------|------|
| `fix_ggml_patch.sh` | 패치 문제 자동 진단 및 해결 |
| `create_ggml_patch.sh` | 새로운 패치 파일 생성을 위한 환경 준비 |
| `update_ggml_patch.sh` | 패치 파일 적용 및 빌드 자동화 |

## 상세 문제 해결 방법

자세한 문제 해결 방법은 다음 문서를 참조하세요:
- `meson_patch_troubleshooting.md` - 상세한 문제 분석 및 해결 방법
- `update_ggml_patch.md` - 패치 업데이트 방법론

## 일반적인 오류와 해결책

### 1. 패치 적용 실패
```bash
# 패치 파일 검증
patch -p1 --dry-run < subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch

# 문제가 있다면 패치 파일 재생성
./create_ggml_patch.sh
```

### 2. meson 명령 실행 오류
```bash
# verbose 모드로 실행하여 상세 오류 확인
meson subprojects update ggml --verbose

# 또는 디버그 모드
MESON_LOG_LEVEL=debug meson subprojects update ggml
```

### 3. 캐시 문제
```bash
# 모든 캐시 정리
rm -rf subprojects/ggml
rm -rf subprojects/packagecache
meson subprojects purge --confirm
```

## 현재 프로젝트 설정

- **GGML 버전**: 489716ba99ecd51164f79e8c6fec0b5bf634eac9
- **패치 파일**: `subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch`
- **Wrap 파일**: `subprojects/ggml.wrap`

## 추가 도움말

문제가 계속 발생하면 다음을 확인하세요:

1. **패치 파일 형식**: `git format-patch` 형식인지 확인
2. **파일 경로**: 패치 파일 내부의 경로가 올바른지 확인
3. **소스 버전**: 패치가 현재 revision에 맞는지 확인
4. **meson 버전**: 최신 meson 버전 사용 권장

## 문제 신고

위 방법들로도 해결되지 않으면 다음 정보와 함께 문제를 신고하세요:
- `./fix_ggml_patch.sh` 출력 결과
- 사용 중인 meson 버전 (`meson --version`)
- 운영체제 및 환경 정보