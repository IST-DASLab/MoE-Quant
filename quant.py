import os
import gc
import re
import math
import argparse

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


from src import dist_utils
from src import data_utils
from src import model_utils
from src import gptq


FP8_GROUP_SIZE = 128
ROUTED_EXPERTS_REGEX = ".*mlp.experts.\d+.(down|gate|up)_proj$"


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the DeepSeek model",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to calibration dataset",
    )
    parser.add_argument(
        "--num_calibration_samples", 
        default=128, 
        type=int, 
        help="Number of samples for calibration."
    )
    parser.add_argument(
        "--max_sequence_length", 
        default=8192, 
        type=int, 
        help="Calibration sequence length."
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
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    # Save params
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True, 
        help="where to save quantized model."
    )
    # Misc params
    parser.add_argument(
        "--offload_activations", 
        action="store_true", 
        help="whether to offload activations to CPU."
    )
    parser.add_argument(
        "--seed", 
        default=0, 
        type=int, 
        help="Random seed."
    )
    args = parser.parse_args()
    return args


def is_subset(set1: set, set2: set):
    return set1 <= set2


def dequantize_weight_from_fp8(W, s):
    g = FP8_GROUP_SIZE
    # Dequantize weight
    d_out, d_in = W.shape
    # Pad weight if needed
    pad_out = math.ceil(d_out / g) * g - d_out
    pad_in = math.ceil(d_in / g) * g - d_in
    W = F.pad(W, (0, pad_in, 0, pad_out))
    d_out_pad, d_in_pad = W.shape

    W = W.view(d_out_pad // g, g, d_in_pad // g, g) 
    s = s.view(d_out_pad // g, 1, d_in_pad // g, 1)
    W = (W * s).view(d_out_pad, d_in_pad)

    # Remove padding
    W = W[:d_out, :d_in]
    return W


def load_param_shard(weight_dir: str, weight_path: str) -> dict[str, torch.Tensor]:
    param_shard = {}
    with safe_open(os.path.join(weight_dir, weight_path), framework="pt", device="cpu") as f:
        param_shard_keys = f.keys()
        for k in param_shard_keys:
            param_shard[k] = f.get_tensor(k)
    return param_shard


def dequantize_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    state_dict_keys = list(state_dict.keys())
    # Dequantize
    for k in state_dict_keys:
        if k.endswith("scale_inv"):
            layer_name, _ = k.rsplit(".", 1)

            W = state_dict[f"{layer_name}.weight"].to(torch.bfloat16) 
            s = state_dict[f"{layer_name}.weight_scale_inv"].to(torch.bfloat16) 

            state_dict[f"{layer_name}.weight"] = dequantize_weight_from_fp8(W, s)
            del state_dict[f"{layer_name}.weight_scale_inv"]


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    torch.set_grad_enabled(False)
    torch.cuda.set_device(device)
    offload_device = "cpu" if args.offload_activations else None

    # Load DeepSeek model
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")
    config.ep_size = world_size

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config=config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        ).eval()
        model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # Prepare calibration dataset
    calibration_dataset = data_utils.prepare_calibration_dataset(
        args.dataset_name_or_path,
        tokenizer,
        args.max_sequence_length,
        args.num_calibration_samples,
        args.seed
    )

    # Take slices (if running on multiple workers)
    num_seq_per_rank = len(calibration_dataset) // world_size
    calibration_dataset = calibration_dataset[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    dist_utils.barrier()

    # Load initial weight shard
    weight_dir = args.model_name_or_path
    current_shard_id = 1
    weight_path = f"model-{current_shard_id:05}-of-000163.safetensors"

    param_buffer = {}
    if dist_utils.is_main():
        param_buffer = load_param_shard(weight_dir, weight_path)
    dist_utils.barrier()
        
    # Prepare input embedding
    inputs = []
    model.model.embed_tokens.to_empty(device=device)
    if dist_utils.is_main():
        model.model.embed_tokens.data = param_buffer["model.embed_tokens.weight"]
    if dist_utils.is_dist_available_and_initialized():
        dist_utils.broadcast_parameters(model.model.embed_tokens)
    for i in range(num_seq_per_rank):
        inputs.append(model.model.embed_tokens(calibration_dataset[i].to(device)).to(offload_device))
    # Offload embeddings back to meta
    model.model.embed_tokens.to(device="meta")
    param_buffer.pop("model.embed_tokens.weight", None)

    for block_idx, block in tqdm(
        enumerate(model.model.layers), 
        desc="Processing transformer blocks",
        total=len(model.model.layers)
    ):
        prefix = f"model.layers.{block_idx}."

        # Collect state dict keys from all processes
        rank_block_keys = [k for k in block.state_dict()]
        if dist_utils.is_main():
            block_keys_with_prefix = [f"{prefix}{k}" for k in rank_block_keys]
            other_ranks_keys = []
            for i in range(1, world_size):
                other_rank_keys = [None for _ in rank_block_keys]
                dist.recv_object_list(other_rank_keys, src=i)
                block_keys_with_prefix.extend([f"{prefix}{k}" for k in other_rank_keys])
                other_ranks_keys.append(other_rank_keys)
            # Make it a set
            block_keys_with_prefix = set(block_keys_with_prefix)
        else:
            block_keys_with_prefix  = []
            other_ranks_keys = []
            dist.send_object_list(rank_block_keys, dst=0)

        if dist_utils.is_main():
            while not is_subset(block_keys_with_prefix, set(param_buffer.keys())):
                current_shard_id += 1
                weight_path = f"model-{current_shard_id:05}-of-000163.safetensors"
                param_buffer.update(load_param_shard(weight_dir, weight_path))
            # Select weights corresponding to chosen block and dequantize them
            block_state_dict = {k[len(prefix):]: v for k, v in param_buffer.items() if k.startswith(prefix)}
            dequantize_state_dict(block_state_dict)

        # Put block onto GPU
        block.to_empty(device=device)

        # Simply load block state dict on master and broadcast
        if block_idx < model.config.first_k_dense_replace:        
            if dist_utils.is_main():
                block.load_state_dict(block_state_dict)
            if dist_utils.is_dist_available_and_initialized():
                dist_utils.broadcast_parameters(block)
        # Send dict with part of expets to target device
        else:
            if dist_utils.is_main():
                for i in range(1, world_size):
                    rank_state_dict = {k: block_state_dict[k] for k in other_ranks_keys[i - 1]}
                    for k in rank_state_dict:
                        dist.send(rank_state_dict[k].to(device), dst=i)
            else:
                rank_state_dict = block.state_dict()
                for k in block.state_dict():
                    dist.recv(rank_state_dict[k], src=0)
            del rank_state_dict
        # Clear memory before calibration
        torch.cuda.empty_cache()
        gc.collect()  

        # Hessian estimate
        layers = model_utils.select_layers(model, prefix, ".*", model_utils.LINEAR_LAYERS)
        handles = {}
        hooks = {}

        for layer_name, layer in layers.items():
            def update_handle_hook(name):
                def _hook(_, inp, out):
                    handles[name].update(inp[0])
                return _hook

            handles[layer_name] = gptq.GPTQ(
                layer,
                args.perchannel,
                args.group_size,
                args.sym,
                args.rel_damp,
                args.block_size,
                args.act_order,
                is_distributed=re.search(ROUTED_EXPERTS_REGEX, layer_name) is not None
            )
            hooks[layer_name] = layer.register_forward_hook(update_handle_hook(layer_name))

        # Collect Hessians
        for i in range(num_seq_per_rank):
            block(inputs[i].to(device))

        for _, h in hooks.items():
            h.remove()

        dist_utils.barrier()
 
        shared_handles = {k: v for k, v in handles.items() if re.search(ROUTED_EXPERTS_REGEX, k) is None}
        expert_handles = {k: v for k, v in handles.items() if k not in shared_handles}

        # Quantized shared handles first
        for handle_name, handle in shared_handles.items():
            dist_utils.print_on_main(f"Quantizing layer {handle_name}")
            qweight, scale, zero, perm = handle.quantize(args.bits)
            if args.save_dir and dist_utils.is_main():
                os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
                torch.save(
                    {"qweight": qweight, "scale": scale, "zero": zero, "perm": perm}, 
                    os.path.join(args.save_dir, handle_name, f"quantized_weight.pt")
                )
            # Destroy handle
            handle.reset()

        # Quantize experts
        if len(expert_handles) > 0:
            dist_utils.print_on_main(f"Processing experts")
        for handle_name, handle in expert_handles.items():
            qweight, scale, zero, perm = handle.quantize(args.bits)
            if args.save_dir:
                os.makedirs(os.path.join(args.save_dir, handle_name), exist_ok=True)
                torch.save(
                    {"qweight": qweight, "scale": scale, "zero": zero, "perm": perm}, 
                    os.path.join(args.save_dir, handle_name, f"quantized_weight.pt")
                )

        dist_utils.barrier()

        # Update activations
        for i in range(num_seq_per_rank):
            inputs[i] = block(inputs[i].to(device))[0].to(offload_device)

        # Offload block
        block.to(device="meta")
        for k in block_keys_with_prefix:
            param_buffer.pop(k, None)
            
        torch.cuda.empty_cache()
        gc.collect()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
