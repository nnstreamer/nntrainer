#!/bin/bash

# GGML 패치 생성 도우미 스크립트
# 사용법: ./create_ggml_patch.sh

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
GGML_WRAP_FILE="$PROJECT_ROOT/subprojects/ggml.wrap"
PATCH_DIR="$PROJECT_ROOT/subprojects/packagefiles/ggml"
PATCH_FILE="$PATCH_DIR/0001-nntrainer-ggml-patch.patch"
TEMP_DIR="$PROJECT_ROOT/temp_ggml_patch"

echo "=== GGML 패치 생성 도우미 스크립트 ==="

# ggml.wrap 파일에서 revision 읽기
if [ ! -f "$GGML_WRAP_FILE" ]; then
    echo "❌ 오류: ggml.wrap 파일을 찾을 수 없습니다: $GGML_WRAP_FILE"
    exit 1
fi

REVISION=$(grep "^revision" "$GGML_WRAP_FILE" | cut -d'=' -f2 | xargs)
if [ -z "$REVISION" ]; then
    echo "❌ 오류: ggml.wrap 파일에서 revision을 찾을 수 없습니다"
    exit 1
fi

echo "📋 현재 GGML revision: $REVISION"

# 임시 디렉토리 정리
if [ -d "$TEMP_DIR" ]; then
    echo "🧹 기존 임시 디렉토리 정리: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
fi

# 원본 GGML 클론
echo "⬇️ 원본 GGML 클론 중..."
if ! git clone https://github.com/ggml-org/ggml.git "$TEMP_DIR"; then
    echo "❌ GGML 클론 실패"
    exit 1
fi

cd "$TEMP_DIR"

# 특정 revision으로 체크아웃
echo "🔄 revision $REVISION으로 체크아웃 중..."
if ! git checkout "$REVISION"; then
    echo "❌ revision 체크아웃 실패"
    exit 1
fi

echo ""
echo "✅ GGML 소스 코드 준비 완료!"
echo "📂 작업 디렉토리: $TEMP_DIR"
echo ""
echo "🔧 다음 단계:"
echo "1. 필요한 파일들을 수정하세요"
echo "2. 수정 완료 후 다음 명령어를 실행하세요:"
echo ""
echo "   cd '$TEMP_DIR'"
echo "   git add ."
echo "   git commit -m 'nntrainer ggml modifications'"
echo "   git format-patch HEAD~1 --stdout > '$PATCH_FILE'"
echo ""
echo "3. 패치 파일이 생성되면 다음 명령어로 적용하세요:"
echo "   cd '$PROJECT_ROOT'"
echo "   ./update_ggml_patch.sh"
echo ""
echo "4. 임시 디렉토리 정리:"
echo "   rm -rf '$TEMP_DIR'"
echo ""

# 기존 패치 파일이 있는 경우 적용 제안
if [ -f "$PATCH_FILE" ]; then
    echo "💡 팁: 기존 패치 파일이 있습니다. 이를 먼저 적용하고 추가 수정하시겠습니까? (y/n)"
    read -r response
    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo "🔄 기존 패치 적용 중..."
        if patch -p1 < "$PATCH_FILE"; then
            echo "✅ 기존 패치 적용 완료"
        else
            echo "❌ 기존 패치 적용 실패 (수동으로 해결 필요)"
        fi
    fi
fi

echo ""
echo "🎯 현재 위치: $(pwd)"
echo "📝 에디터를 사용하여 필요한 파일들을 수정하세요!"
echo ""
echo "주요 수정 대상 파일들:"
find . -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "CMakeLists.txt" | head -10
echo ""
echo "수정 완료 후 위의 git 명령어들을 실행하세요."