import os
import gc
import json
import argparse
from collections import defaultdict
from typing import Optional, Any

from tqdm import tqdm
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from compressed_tensors.compressors import pack_to_int32


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    ) 
    parser.add_argument(
        "--quantized_model_path", 
        type=str, 
        required=True,
        help="Path to quantized model."
    )
    parser.add_argument(
        "--packed_model_path", 
        type=str, 
        required=True,
        help="Whether to save packed model."
    )
    # Quantization params
    parser.add_argument(
        "--bits",
        type=int,
        required=True,
        help="Quantization bitwidth.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Whether to permute in activation order.",
    )
    parser.add_argument(
        "--sym", 
        action="store_true", 
        help="Whether to use symmetric quantization"
    )
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="Fit a unique quantizer to each output dim",
    )
    args = parser.parse_args()
    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def load_param_shard(weight_dir: str, weight_path: str) -> dict[str, torch.Tensor]:
    param_shard = {}
    with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
        param_shard_keys = f.keys()
        for k in param_shard_keys:
            param_shard[k] = f.get_tensor(k)
    return param_shard


def pack_weight(
    weight: dict[torch.Tensor],
    bits: int,
    sym: bool,
    group_size: Optional[int] = None,
) -> dict[torch.Tensor]:
    compressed_data = {}
    qweight, scale, zero = weight['qweight'], weight['scale'], weight['zero']
    group_size = group_size or qweight.shape[-1]
    qweight_shifted = qweight.to(torch.int8) - zero.repeat_interleave(group_size, dim=-1).to(torch.int8)
    qweight_packed = pack_to_int32(qweight_shifted, bits)
    compressed_data = {
        "weight_packed": qweight_packed,
        "weight_shape": torch.tensor(qweight.shape),
        "weight_scale": scale
    }
    if not sym:
        compressed_data["weight_zero_point"] = weight['zero']
    return compressed_data


def prepare_quantization_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config_groups": {
            "group_0": {
                "input_activations": None,
                "output_activations": None,
                "targets": [
                    "Linear"
                ],
                "weights": {
                    "actorder": None,
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": args.group_size,
                    "num_bits": args.bits,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int"
                }
            }
        },
        "format": "pack-quantized",
        "ignore": ["lm_head"],
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed"
    }


def main():
    args = parse_args()

    # Load DeepSeek model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    num_output_shards = len(model.model.layers) + 2
    current_output_shard_id = 1
    quantized_layer_names = defaultdict(list)
    for layer_name in sorted(os.listdir(args.quantized_model_path)):
        block_idx = int(layer_name.split(".")[2])
        quantized_layer_names[block_idx].append(layer_name)
    safetensors_index = {}
    # Prepare directory to save packed weights
    os.makedirs(args.packed_model_path, exist_ok=True)

    # Load initial weight shard
    weight_dir = args.model_name_or_path
    current_input_shard_id = 1
    weight_path = f"model-{current_input_shard_id:05}-of-000163.safetensors"

    param_buffer = load_param_shard(weight_dir, weight_path)

    # Save embeddings
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {"model.embed_tokens.weight": param_buffer["model.embed_tokens.weight"]}, 
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index["model.embed_tokens.weight"] = current_output_shard_path
    del param_buffer["model.embed_tokens.weight"]

    # Process blocks
    for block_idx, block in tqdm(
        enumerate(model.model.layers), 
        desc="Processing transformer blocks",
        total=len(model.model.layers)
    ):
        if block_idx == 4:
            assert False
        current_output_shard_id += 1
        prefix = f"model.layers.{block_idx}."
        block_keys_with_prefix = set(f"{prefix}{k}" for k in block.state_dict())

        while not is_subset(block_keys_with_prefix, set(param_buffer.keys())):
            current_input_shard_id += 1
            weight_path = f"model-{current_input_shard_id:05}-of-000163.safetensors"
            param_buffer.update(load_param_shard(weight_dir, weight_path))

        block_state_dict = {f"{prefix}{k}": param_buffer[f"{prefix}{k}"] for k in block.state_dict().keys()}

        for layer_name in quantized_layer_names[block_idx]:
            weight_state_dict = torch.load(
                os.path.join(args.quantized_model_path, layer_name, "quantized_weight.pt"),
                weights_only=True
            )
            packed_weight_state_dict = pack_weight(weight_state_dict, args.bits, args.sym, args.group_size)
            block_state_dict.pop(f"{layer_name}.weight")
            block_state_dict.pop(f"{layer_name}.weight_scale_inv", None)
            block_state_dict.update({f"{layer_name}.{k}": v for k, v in packed_weight_state_dict.items()})

        # Save block
        current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
        save_file(
            block_state_dict, 
            os.path.join(args.packed_model_path, current_output_shard_id)
        )
        for k in block_state_dict:
            safetensors_index[k] = current_output_shard_path

        for k in block_keys_with_prefix:
            param_buffer.pop(k, None) 

        del block_state_dict
        gc.collect()

    # Load final shard
    if current_input_shard_id < 163:
        current_input_shard_id = 163
        weight_path = f"model-{current_input_shard_id:05}-of-000163.safetensors"
        param_buffer.update(load_param_shard(weight_dir, weight_path))

    # Save lm head
    current_output_shard_id += 1
    current_output_shard_path = f"model-{current_output_shard_id:05}-of-{num_output_shards:05}.safetensors"
    save_file(
        {
            "lm_head.weight": param_buffer["lm_head.weight"],
            "model.norm.weight": param_buffer["model.norm.weight"]
        }, 
        os.path.join(args.packed_model_path, current_output_shard_path)
    )
    safetensors_index["lm_head.weight"] = current_output_shard_path
    # Save safetensors index
    with open(os.path.join(args.packed_model_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": safetensors_index}, f)
    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(args)
    # Save configs
    config.save_pretrained(args.packed_model_path)
    model.generation_config.save_pretrained(args.packed_model_path)
    # Save tokenizer
    tokenizer.save_pretrained(args.packed_model_path)


if __name__ == "__main__":
    main()