#!/usr/bin/env python3
"""Debug copy attention to understand why it's failing."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import mlx.core as mx

from imt.config import IMTConfig, TrainingConfig
from imt.model.imt import IndexedMemoryTransformer
from imt.data.tokenizer import ASCIITokenizer
from imt.data.chunked_dataset import ChunkedHashHopDataset


def load_config(config_path: str):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    model_config = IMTConfig(**config_dict.get("model", {}))
    train_config = TrainingConfig(**config_dict.get("training", {}))
    return model_config, train_config


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/imt_benchmark_10k_v2.yaml"
    checkpoint_name = config_path.split("/")[-1].replace(".yaml", "").replace("imt_", "")

    config, training_config = load_config(config_path)

    print(f"Loading config from {config_path}")
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  retrieval_top_k: {config.retrieval_top_k}")

    model = IndexedMemoryTransformer(config)
    checkpoint_path = f"checkpoints/{checkpoint_name}/final"
    model.load_weights(f"{checkpoint_path}/weights.safetensors")
    print(f"Loaded weights from {checkpoint_path}")

    tokenizer = ASCIITokenizer()
    dataset = ChunkedHashHopDataset(
        config=config,
        train_config=training_config,
        tokenizer=tokenizer,
        use_simple_format=True,
    )

    # Find a sample where retrieval succeeds but generation fails
    for i in range(20):
        sample = dataset.generate_sample()
        raw = sample.raw_sample
        query = list(raw.targets.keys())[0]
        expected_answer = raw.targets[query]

        chunk_tokens = sample.chunk_tokens
        query_tokens = sample.query_tokens[0:1, :]
        target_tokens = sample.target_tokens[0:1, :]

        # Get model predictions
        (logits, copy_attention, copy_gate, retrieval_scores,
         chunk_indices, index_keys, all_chunk_scores) = model(
            chunk_tokens, query_tokens, target_tokens
        )

        # Check retrieval
        target_idx = sample.target_chunk_indices[0].item()
        retrieved_indices = [chunk_indices[0, k].item() for k in range(config.retrieval_top_k)]

        if target_idx not in retrieved_indices:
            continue

        # Get prediction
        tf_predictions = mx.argmax(logits, axis=-1)[0]
        tf_output = tokenizer.decode(tf_predictions.tolist()[:4])

        if tf_output == expected_answer:
            continue

        # Found a case where retrieval succeeded but generation failed
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: Retrieval OK, Generation FAILED")
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Expected: {expected_answer}")
        print(f"Got: {tf_output}")
        print(f"Target chunk: {target_idx}")
        print(f"Retrieved chunks: {retrieved_indices[:8]}...")

        # Find target chunk position in retrieved context
        target_pos_in_retrieved = retrieved_indices.index(target_idx)
        context_start = target_pos_in_retrieved * config.chunk_size
        print(f"Target chunk is at position {target_pos_in_retrieved} in retrieved (context pos {context_start}-{context_start+config.chunk_size})")

        # Look at copy attention for first output position
        copy_attn_0 = copy_attention[0, 0, :].tolist()  # First output position

        # Find where query appears in target chunk
        target_chunk_tokens = chunk_tokens[target_idx].tolist()
        target_chunk_str = tokenizer.decode(target_chunk_tokens)
        query_pos_in_chunk = target_chunk_str.find(query)

        print(f"\nTarget chunk content: {target_chunk_str}")
        print(f"Query '{query}' found at position {query_pos_in_chunk} in chunk")

        if query_pos_in_chunk >= 0:
            value_start = query_pos_in_chunk + len(query) + 1  # +1 for '>'
            print(f"VALUE should start at position {value_start} in chunk")
            print(f"Context position for VALUE: {context_start + value_start}")

            # Check copy attention at that position
            expected_context_pos = context_start + value_start
            if expected_context_pos < len(copy_attn_0):
                print(f"\nCopy attention at expected position: {copy_attn_0[expected_context_pos]:.4f}")

                # Find max attention position
                max_attn_pos = copy_attn_0.index(max(copy_attn_0))
                max_attn_val = max(copy_attn_0)
                print(f"Max copy attention: {max_attn_val:.4f} at position {max_attn_pos}")

                # What token is at max attention?
                flat_chunk_tokens = []
                for idx in retrieved_indices:
                    flat_chunk_tokens.extend(chunk_tokens[idx].tolist())
                if max_attn_pos < len(flat_chunk_tokens):
                    max_token = tokenizer.decode([flat_chunk_tokens[max_attn_pos]])
                    print(f"Token at max attention: '{max_token}'")

        print(f"\nCopy gate mean: {mx.mean(copy_gate).item():.3f}")

        # Check all 4 output positions
        print(f"\n--- Copy attention per output position ---")
        for out_pos in range(4):
            copy_attn_pos = copy_attention[0, out_pos, :].tolist()
            max_pos = copy_attn_pos.index(max(copy_attn_pos))
            max_val = max(copy_attn_pos)

            expected_pos = context_start + value_start + out_pos
            expected_attn = copy_attn_pos[expected_pos] if expected_pos < len(copy_attn_pos) else 0

            token_at_max = tokenizer.decode([flat_chunk_tokens[max_pos]]) if max_pos < len(flat_chunk_tokens) else "?"
            expected_token = expected_answer[out_pos]

            print(f"  Position {out_pos}: max_attn={max_val:.4f} at pos {max_pos} ('{token_at_max}'), expected pos {expected_pos} has attn={expected_attn:.4f} ('{expected_token}')")

        # Check final logits
        print(f"\n--- Final logits analysis ---")
        for out_pos in range(4):
            pos_logits = logits[0, out_pos, :].tolist()
            max_logit_idx = pos_logits.index(max(pos_logits))
            max_logit_token = tokenizer.decode([max_logit_idx])
            expected_token = expected_answer[out_pos]
            expected_token_idx = ord(expected_token) - ord(' ') + 4  # ASCII tokenizer offset
            expected_logit = pos_logits[expected_token_idx]
            print(f"  Position {out_pos}: argmax={max_logit_idx} ('{max_logit_token}'), expected token '{expected_token}' (idx {expected_token_idx}) has logit={expected_logit:.4f}")

        # Only analyze first error
        break


if __name__ == "__main__":
    main()
