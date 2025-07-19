#!/usr/bin/env python3
"""
Simple PyTorch to Q4_K Converter
=================================

이 스크립트는 PyTorch 모델을 Q4_K 형식으로 간단하게 변환할 수 있는 도구입니다.
llama.cpp와 호환되는 형식으로 변환하여 메모리 사용량을 크게 줄일 수 있습니다.

사용법:
    python simple_q4k_converter.py --input model.pth --output model_q4k.bin
    
또는 Python 코드에서:
    from simple_q4k_converter import convert_model_to_q4k
    convert_model_to_q4k(model, "output.bin")
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

# 앞서 작성한 코드들을 임포트 (실제 사용시에는 별도 파일로 분리)
from correct_q4k_converter import convert_to_q4k, save_q4k_weights

def convert_model_to_q4k(model_or_path, output_path: str, model_name: str = "converted_model"):
    """
    PyTorch 모델을 Q4_K 형식으로 변환하는 간단한 함수
    
    Args:
        model_or_path: torch.nn.Module 객체 또는 모델 파일 경로
        output_path: 출력 파일 경로
        model_name: 모델 이름 (메타데이터용)
    """
    
    print("🔄 Starting Q4_K conversion...")
    
    # 입력이 경로인 경우 모델 로드
    if isinstance(model_or_path, (str, Path)):
        print(f"📂 Loading model from: {model_or_path}")
        
        if not os.path.exists(model_or_path):
            raise FileNotFoundError(f"Model file not found: {model_or_path}")
        
        # 다양한 형식 지원
        if str(model_or_path).endswith('.pth') or str(model_or_path).endswith('.pt'):
            try:
                # state_dict 형식 시도
                state_dict = torch.load(model_or_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                # 간단한 래퍼 모델 생성
                class StateDict(nn.Module):
                    def __init__(self, state_dict):
                        super().__init__()
                        self._state_dict = state_dict
                    
                    def state_dict(self):
                        return self._state_dict
                
                model = StateDict(state_dict)
            except:
                # 전체 모델 로드 시도
                model = torch.load(model_or_path, map_location='cpu')
                if hasattr(model, 'eval'):
                    model.eval()
        else:
            raise ValueError(f"Unsupported file format: {model_or_path}")
    
    elif isinstance(model_or_path, nn.Module):
        model = model_or_path
        model.eval()
    else:
        raise ValueError("Input must be a PyTorch model or path to model file")
    
    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.state_dict().values() 
                      if p.dtype in [torch.float32, torch.float16])
    total_size_mb = sum(p.numel() * 4 for p in model.state_dict().values() 
                       if p.dtype in [torch.float32, torch.float16]) / (1024 * 1024)
    
    print(f"📊 Model info:")
    print(f"   Parameters: {total_params:,}")
    print(f"   Original size: {total_size_mb:.1f} MB")
    print(f"   Layers: {len(model.state_dict())}")
    
    # Q4_K로 변환
    print(f"⚙️  Converting to Q4_K format...")
    save_q4k_weights(model, output_path)
    
    # 결과 확인
    if os.path.exists(output_path):
        compressed_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = total_size_mb / compressed_size_mb
        
        print(f"✅ Conversion completed!")
        print(f"   Output: {output_path}")
        print(f"   Compressed size: {compressed_size_mb:.1f} MB")
        print(f"   Compression ratio: {compression_ratio:.2f}x")
        print(f"   Space saved: {total_size_mb - compressed_size_mb:.1f} MB ({(1 - compressed_size_mb/total_size_mb)*100:.1f}%)")
    else:
        print("❌ Conversion failed!")

def convert_huggingface_model(model_name_or_path: str, output_path: str):
    """
    Hugging Face 모델을 Q4_K로 변환
    
    Args:
        model_name_or_path: HF 모델 이름 또는 로컬 경로
        output_path: 출력 파일 경로
    """
    try:
        from transformers import AutoModel, AutoConfig
    except ImportError:
        raise ImportError("transformers library is required for Hugging Face models")
    
    print(f"🤗 Loading Hugging Face model: {model_name_or_path}")
    
    # 설정 로드
    config = AutoConfig.from_pretrained(model_name_or_path)
    print(f"   Model type: {config.model_type}")
    print(f"   Architecture: {config.architectures}")
    
    # 모델 로드
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,  # Q4_K 변환을 위해 float32 사용
        device_map="cpu",
        trust_remote_code=True
    )
    
    model_name = model_name_or_path.split('/')[-1] if '/' in model_name_or_path else model_name_or_path
    convert_model_to_q4k(model, output_path, model_name)

def validate_q4k_file(file_path: str):
    """Q4_K 파일 유효성 검사"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        from correct_q4k_converter import load_q4k_weights
        weights = load_q4k_weights(file_path)
        
        print(f"✅ Q4_K file validation successful!")
        print(f"   File: {file_path}")
        print(f"   Size: {os.path.getsize(file_path) / (1024*1024):.1f} MB")
        print(f"   Tensors: {len(weights)}")
        
        for i, weight in enumerate(weights[:3]):  # 처음 3개만 표시
            print(f"   [{i+1}] {weight['name']}: {weight['original_shape']}")
        
        if len(weights) > 3:
            print(f"   ... and {len(weights) - 3} more tensors")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to Q4_K quantized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert PyTorch model file
  python simple_q4k_converter.py --input model.pth --output model_q4k.bin
  
  # Convert Hugging Face model
  python simple_q4k_converter.py --hf-model microsoft/DialoGPT-medium --output dialogpt_q4k.bin
  
  # Validate Q4_K file
  python simple_q4k_converter.py --validate model_q4k.bin
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=str, 
                      help='Input PyTorch model file (.pth, .pt)')
    group.add_argument('--hf-model', type=str,
                      help='Hugging Face model name or path')
    group.add_argument('--validate', type=str,
                      help='Validate existing Q4_K file')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Output Q4_K file path (.bin)')
    parser.add_argument('--name', type=str, default="converted_model",
                       help='Model name for metadata')
    
    args = parser.parse_args()
    
    # 유효성 검사 모드
    if args.validate:
        validate_q4k_file(args.validate)
        return
    
    # 출력 파일 경로 확인
    if not args.output:
        print("❌ Output path is required for conversion")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")
    
    try:
        # 변환 실행
        if args.input:
            convert_model_to_q4k(args.input, args.output, args.name)
        elif args.hf_model:
            convert_huggingface_model(args.hf_model, args.output)
        
        # 변환된 파일 검증
        print(f"\n🔍 Validating converted file...")
        validate_q4k_file(args.output)
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# 추가 유틸리티 함수들

def quick_convert(model, output_name: str = None):
    """
    Jupyter notebook 등에서 빠르게 변환하기 위한 함수
    
    Usage:
        model = torch.load('my_model.pth')
        quick_convert(model, 'my_model_q4k')
    """
    if output_name is None:
        output_name = "quick_converted_q4k.bin"
    elif not output_name.endswith('.bin'):
        output_name += '.bin'
    
    convert_model_to_q4k(model, output_name, "quick_convert")
    return output_name

def estimate_compression(model_or_path):
    """
    변환 전 압축률 추정
    
    Returns:
        dict: 예상 압축 정보
    """
    if isinstance(model_or_path, (str, Path)):
        model = torch.load(model_or_path, map_location='cpu')
        if isinstance(model, dict):
            params = model
        else:
            params = model.state_dict()
    else:
        params = model_or_path.state_dict()
    
    total_params = sum(p.numel() for p in params.values() 
                      if p.dtype in [torch.float32, torch.float16])
    original_size_mb = sum(p.numel() * 4 for p in params.values() 
                          if p.dtype in [torch.float32, torch.float16]) / (1024 * 1024)
    
    # Q4_K는 평균적으로 4.5 bits per weight
    estimated_size_mb = total_params * 4.5 / 8 / (1024 * 1024)
    estimated_ratio = original_size_mb / estimated_size_mb
    
    return {
        'total_parameters': total_params,
        'original_size_mb': original_size_mb,
        'estimated_q4k_size_mb': estimated_size_mb,
        'estimated_compression_ratio': estimated_ratio,
        'estimated_space_saved_mb': original_size_mb - estimated_size_mb,
        'estimated_space_saved_percent': (1 - estimated_size_mb/original_size_mb) * 100
    }

if __name__ == "__main__":
    main()