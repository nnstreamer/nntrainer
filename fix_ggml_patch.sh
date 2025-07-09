#!/bin/bash

# GGML 패치 문제 해결 스크립트
# 사용법: ./fix_ggml_patch.sh

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
GGML_WRAP_FILE="$PROJECT_ROOT/subprojects/ggml.wrap"
PATCH_DIR="$PROJECT_ROOT/subprojects/packagefiles/ggml"
PATCH_FILE="$PATCH_DIR/0001-nntrainer-ggml-patch.patch"

echo "=== GGML 패치 문제 해결 스크립트 ==="

# 1. 현재 설정 확인
echo "🔍 현재 wrap 파일 설정:"
if [ -f "$GGML_WRAP_FILE" ]; then
    cat "$GGML_WRAP_FILE"
else
    echo "❌ ggml.wrap 파일을 찾을 수 없습니다: $GGML_WRAP_FILE"
    exit 1
fi

echo ""

# 2. 패치 파일 존재 확인
echo "📋 패치 파일 상태 확인:"
if [ -f "$PATCH_FILE" ]; then
    echo "✅ 패치 파일 존재: $PATCH_FILE"
    echo "   크기: $(stat -c%s "$PATCH_FILE") bytes"
    echo "   수정 시간: $(stat -c%y "$PATCH_FILE")"
else
    echo "❌ 패치 파일을 찾을 수 없습니다: $PATCH_FILE"
    exit 1
fi

echo ""

# 3. 패치 파일 형식 검증
echo "🔍 패치 파일 형식 검증:"
if head -10 "$PATCH_FILE" | grep -q "^From"; then
    echo "✅ 패치 파일 형식이 올바릅니다 (git format-patch 형식)"
else
    echo "⚠️  패치 파일 형식이 올바르지 않을 수 있습니다"
    echo "   첫 10줄:"
    head -10 "$PATCH_FILE"
fi

echo ""

# 4. 패치 파일 내용 분석
echo "🔍 패치 파일 내용 분석:"
DIFF_COUNT=$(grep -c "^diff --git" "$PATCH_FILE" || echo "0")
HUNK_COUNT=$(grep -c "^@@" "$PATCH_FILE" || echo "0")
echo "   수정된 파일 수: $DIFF_COUNT"
echo "   변경 블록 수: $HUNK_COUNT"

echo "   수정된 파일 목록:"
grep "^diff --git" "$PATCH_FILE" | sed 's/^diff --git a\/// ; s/ b\/.*//' | sed 's/^/     - /'

echo ""

# 5. 패치 적용 테스트
echo "🧪 패치 적용 테스트:"
TEMP_DIR=$(mktemp -d)
echo "   임시 디렉토리: $TEMP_DIR"

cd "$TEMP_DIR"

# revision 읽기
REVISION=$(grep "^revision" "$GGML_WRAP_FILE" | cut -d'=' -f2 | xargs)
echo "   테스트 revision: $REVISION"

# git clone
echo "   원본 소스 다운로드 중..."
if git clone --quiet https://github.com/ggml-org/ggml.git; then
    echo "   ✅ 소스 다운로드 완료"
else
    echo "   ❌ 소스 다운로드 실패"
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
    exit 1
fi

cd ggml

# revision 체크아웃
if git checkout --quiet "$REVISION"; then
    echo "   ✅ revision 체크아웃 완료"
else
    echo "   ❌ revision 체크아웃 실패"
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 패치 적용 테스트 (dry-run)
echo "   패치 적용 테스트 (dry-run):"
if patch -p1 --dry-run --silent < "$PATCH_FILE"; then
    echo "   ✅ 패치 적용 테스트 성공"
    PATCH_OK=true
else
    echo "   ❌ 패치 적용 테스트 실패"
    echo "   상세 오류:"
    patch -p1 --dry-run < "$PATCH_FILE" 2>&1 | head -20 | sed 's/^/     /'
    PATCH_OK=false
fi

# 정리
cd "$PROJECT_ROOT"
rm -rf "$TEMP_DIR"

echo ""

# 6. 해결 방안 제시
if [ "$PATCH_OK" = true ]; then
    echo "✅ 패치 파일이 올바르게 작동합니다."
    echo ""
    echo "🔧 권장 해결 방법:"
    echo "1. 서브프로젝트 완전 정리 및 재적용"
    echo "   rm -rf subprojects/ggml"
    echo "   rm -rf subprojects/packagecache"
    echo "   meson subprojects download ggml"
    echo "   meson subprojects update ggml"
    echo ""
    echo "2. 또는 자동화 스크립트 사용:"
    echo "   ./update_ggml_patch.sh"
    
    # 자동 적용 여부 확인
    echo ""
    echo "🤖 자동으로 해결을 시도하시겠습니까? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "📦 서브프로젝트 정리 중..."
        rm -rf subprojects/ggml
        if [ -d subprojects/packagecache ]; then
            rm -rf subprojects/packagecache
        fi
        
        echo "⬇️ 서브프로젝트 다운로드 중..."
        if command -v meson >/dev/null 2>&1; then
            meson subprojects download ggml
            meson subprojects update ggml
            echo "✅ 완료!"
        else
            echo "❌ meson 명령을 찾을 수 없습니다."
            echo "   수동으로 다음 명령을 실행하세요:"
            echo "   meson subprojects download ggml"
            echo "   meson subprojects update ggml"
        fi
    fi
else
    echo "❌ 패치 파일에 문제가 있습니다."
    echo ""
    echo "🔧 권장 해결 방법:"
    echo "1. 패치 파일 재생성:"
    echo "   ./create_ggml_patch.sh"
    echo "   # 임시 디렉토리에서 수정 작업 수행"
    echo "   # git add . && git commit -m 'nntrainer ggml modifications'"
    echo "   # git format-patch HEAD~1 --stdout > ../subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch"
    echo ""
    echo "2. 또는 기존 패치 파일 수정:"
    echo "   # 패치 파일을 직접 편집하여 경로나 내용 수정"
    echo "   vim subprojects/packagefiles/ggml/0001-nntrainer-ggml-patch.patch"
    echo ""
    echo "3. 패치 파일 백업 복원 (있는 경우):"
    echo "   # 백업 파일이 있다면 복원"
    echo "   ls -la subprojects/packagefiles/ggml/*.backup*"
fi

echo ""
echo "🔍 추가 디버깅 정보:"
echo "   - 패치 파일 위치: $PATCH_FILE"
echo "   - wrap 파일 위치: $GGML_WRAP_FILE"
echo "   - 사용할 수 있는 명령어:"
echo "     * meson subprojects download ggml --verbose"
echo "     * meson subprojects update ggml --verbose"
echo "     * MESON_LOG_LEVEL=debug meson subprojects update ggml"