#!/bin/bash

# GGML 서브프로젝트 패치 업데이트 스크립트
# 사용법: ./update_ggml_patch.sh [patch_file]

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR"
GGML_WRAP_FILE="$PROJECT_ROOT/subprojects/ggml.wrap"
PATCH_DIR="$PROJECT_ROOT/subprojects/packagefiles/ggml"
PATCH_FILE="$PATCH_DIR/0001-nntrainer-ggml-patch.patch"
SUBPROJECT_DIR="$PROJECT_ROOT/subprojects/ggml"

echo "=== GGML 서브프로젝트 패치 업데이트 스크립트 ==="

# 인수 처리
if [ "$#" -eq 1 ]; then
    NEW_PATCH_FILE="$1"
    if [ ! -f "$NEW_PATCH_FILE" ]; then
        echo "❌ 오류: 패치 파일 '$NEW_PATCH_FILE'을 찾을 수 없습니다."
        exit 1
    fi
    echo "📋 새로운 패치 파일: $NEW_PATCH_FILE"
fi

# 필요한 파일들이 존재하는지 확인
if [ ! -f "$GGML_WRAP_FILE" ]; then
    echo "❌ 오류: ggml.wrap 파일을 찾을 수 없습니다: $GGML_WRAP_FILE"
    exit 1
fi

# 패치 디렉토리 확인 및 생성
if [ ! -d "$PATCH_DIR" ]; then
    echo "📁 패치 디렉토리 생성: $PATCH_DIR"
    mkdir -p "$PATCH_DIR"
fi

# 기존 패치 파일 백업
if [ -f "$PATCH_FILE" ]; then
    BACKUP_FILE="${PATCH_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "💾 기존 패치 파일 백업: $BACKUP_FILE"
    cp "$PATCH_FILE" "$BACKUP_FILE"
fi

# 새로운 패치 파일이 제공된 경우 복사
if [ -n "$NEW_PATCH_FILE" ]; then
    echo "📝 새로운 패치 파일 적용: $NEW_PATCH_FILE -> $PATCH_FILE"
    cp "$NEW_PATCH_FILE" "$PATCH_FILE"
fi

# 기존 서브프로젝트 정리
if [ -d "$SUBPROJECT_DIR" ]; then
    echo "🧹 기존 서브프로젝트 정리: $SUBPROJECT_DIR"
    rm -rf "$SUBPROJECT_DIR"
fi

# 서브프로젝트 다운로드 및 패치 적용
echo "⬇️ 서브프로젝트 다운로드 중..."
if ! meson subprojects download ggml; then
    echo "❌ 서브프로젝트 다운로드 실패"
    exit 1
fi

echo "🔄 서브프로젝트 업데이트 중..."
if ! meson subprojects update ggml; then
    echo "❌ 서브프로젝트 업데이트 실패"
    exit 1
fi

# 패치 적용 확인
echo "✅ 패치 적용 확인 중..."
if [ -d "$SUBPROJECT_DIR" ]; then
    echo "📂 서브프로젝트 디렉토리: $SUBPROJECT_DIR"
    ls -la "$SUBPROJECT_DIR" | head -10
else
    echo "❌ 서브프로젝트 디렉토리가 생성되지 않았습니다"
    exit 1
fi

# 빌드 테스트
echo "🔨 빌드 테스트 중..."
BUILD_DIR="$PROJECT_ROOT/build"

if [ -d "$BUILD_DIR" ]; then
    echo "🔧 기존 빌드 디렉토리에서 컴파일 중..."
    if ! meson compile -C "$BUILD_DIR"; then
        echo "❌ 빌드 실패"
        exit 1
    fi
else
    echo "🏗️ 새로운 빌드 설정 중..."
    if ! meson setup "$BUILD_DIR"; then
        echo "❌ 빌드 설정 실패"
        exit 1
    fi
    
    echo "🔧 컴파일 중..."
    if ! meson compile -C "$BUILD_DIR"; then
        echo "❌ 빌드 실패"
        exit 1
    fi
fi

echo ""
echo "✅ GGML 패치 업데이트 완료!"
echo "📋 적용된 패치 파일: $PATCH_FILE"
echo "📂 서브프로젝트 위치: $SUBPROJECT_DIR"
echo "🔨 빌드 디렉토리: $BUILD_DIR"
echo ""
echo "🔍 패치 내용 확인:"
echo "   head -20 $PATCH_FILE"
echo ""
echo "🧪 테스트 권장사항:"
echo "   - 수정된 기능이 올바르게 작동하는지 확인하세요"
echo "   - 기존 테스트가 여전히 통과하는지 확인하세요"
echo "   - 새로운 기능에 대한 테스트를 추가하세요"