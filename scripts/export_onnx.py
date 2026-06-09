"""
export_onnx.py — Export PyTorch model sang ONNX.
"""
import argparse
import os
import sys
import yaml
import torch
from pathlib import Path

# Add project root to sys.path so we can import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recognizer.model import CRNN
from src.trainer.train_cer import CharsetCodec

def parse_args():
    parser = argparse.ArgumentParser(description="Export CRNN PyTorch model to ONNX")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument("--output", type=str, default="checkpoints/crnn.onnx", help="Path to save the .onnx model")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    return parser.parse_args()

def export_to_onnx(config_path, ckpt_path, output_path, opset_version):
    print(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    charset_path = config['paths']['charset']
    codec = CharsetCodec(charset_path)

    print(f"Loading checkpoint from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # Infer num_classes from checkpoint if possible
    if 'fc.bias' in state_dict:
        num_classes = state_dict['fc.bias'].shape[0] - 1
        print(f"Inferred num_classes={num_classes} from checkpoint.")
    else:
        num_classes = len(codec)
        print(f"Using num_classes={num_classes} from charset.")

    print("Initializing CRNN model...")
    model = CRNN(
        num_classes=num_classes,
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        lstm_dropout=0.0
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Create dummy input: (batch_size, channels, target_h, width)
    target_h = config['preprocess']['target_h']
    dummy_input = torch.randn(1, 1, target_h, 200)

    print(f"Exporting to ONNX: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 3: 'width'},
            'output': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Export completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(args.config, args.ckpt, args.output, args.opset)
