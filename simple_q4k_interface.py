#!/usr/bin/env python3
"""
Simple Q4_K Quantization Interface
=================================

Easy-to-use interface for converting PyTorch models to Q4_K format.
Based on llama.cpp's gguf-py implementation.

Usage:
    python simple_q4k_interface.py --model model.pth --output model_q4k.npz
    
Or in Python:
    from simple_q4k_interface import quantize_model
    quantize_model("model.pth", "model_q4k.npz")
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Union, Optional

# Import our accurate Q4_K converter
from accurate_q4k_converter import save_q4k_model, load_q4k_model, logger

def quantize_model(input_path: str, 
                  output_path: str,
                  model_name: Optional[str] = None) -> bool:
    """
    Quantize a PyTorch model to Q4_K format.
    
    Args:
        input_path: Path to input model (.pth, .pt, .safetensors)
        output_path: Path to output quantized model
        model_name: Optional model name for metadata
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine model name
        if model_name is None:
            model_name = Path(input_path).stem
        
        logger.info(f"Loading model from: {input_path}")
        
        # Load model
        if input_path.endswith('.safetensors'):
            # Load safetensors
            try:
                from safetensors.torch import load_file
                tensors = load_file(input_path)
                logger.info(f"Loaded {len(tensors)} tensors from safetensors file")
            except ImportError:
                logger.error("safetensors library not found. Install with: pip install safetensors")
                return False
        else:
            # Load PyTorch model
            checkpoint = torch.load(input_path, map_location='cpu')
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    tensors = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    tensors = checkpoint['state_dict']
                else:
                    tensors = checkpoint
            else:
                # Assume it's a model directly
                if hasattr(checkpoint, 'state_dict'):
                    tensors = checkpoint.state_dict()
                else:
                    logger.error("Unable to extract tensors from model file")
                    return False
        
        # Filter float tensors only
        float_tensors = {}
        for name, tensor in tensors.items():
            if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                float_tensors[name] = tensor
            else:
                logger.debug(f"Skipping non-float tensor: {name} ({tensor.dtype})")
        
        logger.info(f"Found {len(float_tensors)} float tensors to quantize")
        
        if not float_tensors:
            logger.error("No float tensors found to quantize")
            return False
        
        # Quantize and save
        save_q4k_model(float_tensors, output_path, model_name)
        
        logger.info("✅ Quantization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quantization failed: {str(e)}")
        return False

def verify_quantized_model(file_path: str) -> bool:
    """
    Verify a quantized model file.
    
    Args:
        file_path: Path to quantized model file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        logger.info(f"Verifying quantized model: {file_path}")
        
        # Load model
        data = load_q4k_model(file_path)
        
        metadata = data['metadata']
        tensors = data['quantized_tensors']
        
        logger.info(f"Model name: {metadata.get('model_name', 'Unknown')}")
        logger.info(f"Quantization type: {metadata.get('quantization_type', 'Unknown')}")
        logger.info(f"Number of quantized tensors: {len(tensors)}")
        
        # Show tensor info
        if 'tensors' in metadata:
            total_params = 0
            for name, info in metadata['tensors'].items():
                shape = info['original_shape']
                n_params = 1
                for dim in shape:
                    n_params *= dim
                total_params += n_params
                logger.info(f"  {name}: {shape} ({n_params:,} parameters)")
            
            logger.info(f"Total parameters: {total_params:,}")
        
        logger.info("✅ Model verification completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model verification failed: {str(e)}")
        return False

def compare_models(original_path: str, quantized_path: str) -> None:
    """
    Compare original and quantized model sizes.
    
    Args:
        original_path: Path to original model
        quantized_path: Path to quantized model
    """
    try:
        original_size = os.path.getsize(original_path)
        quantized_size = os.path.getsize(quantized_path)
        
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        
        logger.info("📊 Model Comparison:")
        logger.info(f"  Original size:  {original_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Quantized size: {quantized_size / 1024 / 1024:.1f} MB")
        logger.info(f"  Compression:    {compression_ratio:.2f}x")
        logger.info(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Comparison failed: {str(e)}")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to Q4_K quantized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize a PyTorch model
  python simple_q4k_interface.py --model model.pth --output model_q4k.npz
  
  # Quantize with custom name
  python simple_q4k_interface.py --model model.pth --output model_q4k.npz --name "My Model"
  
  # Verify a quantized model
  python simple_q4k_interface.py --verify model_q4k.npz
  
  # Compare original and quantized sizes
  python simple_q4k_interface.py --compare model.pth model_q4k.npz
        """
    )
    
    parser.add_argument('--model', '-m', type=str,
                       help='Input model file (.pth, .pt, .safetensors)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output quantized model file (.npz)')
    parser.add_argument('--name', '-n', type=str,
                       help='Model name for metadata')
    parser.add_argument('--verify', '-v', type=str,
                       help='Verify a quantized model file')
    parser.add_argument('--compare', '-c', nargs=2, metavar=('ORIGINAL', 'QUANTIZED'),
                       help='Compare original and quantized model sizes')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel('DEBUG')
    
    # Handle different operations
    if args.verify:
        success = verify_quantized_model(args.verify)
        sys.exit(0 if success else 1)
    
    elif args.compare:
        compare_models(args.compare[0], args.compare[1])
        sys.exit(0)
    
    elif args.model and args.output:
        # Check input file exists
        if not os.path.exists(args.model):
            logger.error(f"Input model file not found: {args.model}")
            sys.exit(1)
        
        # Quantize model
        success = quantize_model(args.model, args.output, args.name)
        
        if success and os.path.exists(args.model):
            # Show comparison
            compare_models(args.model, args.output)
        
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()