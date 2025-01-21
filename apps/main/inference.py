from dataclasses import dataclass, field

import sys
import gc
import time
import logging
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
from tabulate import tabulate
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from apps.main.generate import (
    PackedCausalTransformerGenerator,
    PackedCausalTransformerGeneratorArgs,
    load_consolidated_model_and_tokenizer,
)
from apps.main.transformer import LMTransformer, LMTransformerArgs

logger = logging.getLogger()


@dataclass
class LatencyArgs:
    ckpt_dir: list = field(default_factory=list)
    num_samples: int = 2
    input_length: int = 512
    output_length: int = 128
    batch_size: int = 1
    generator: PackedCausalTransformerGeneratorArgs = PackedCausalTransformerGeneratorArgs()


def main():
    cli_args = OmegaConf.from_cli()
    file_cfg = OmegaConf.load(cli_args.config)
    del cli_args.config
    default_cfg = OmegaConf.structured(LatencyArgs())
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)

    print(f"Configuration: {cfg}", flush=True)

    for ckpt in cfg.ckpt_dir:
        assert Path(ckpt).exists(), f"Checkpoint {ckpt} does not exist"
        consolidate_path = Path(ckpt) / "consolidated"
        model, tokenizer, _ = load_consolidated_model_and_tokenizer(
            str(consolidate_path),
            model_cls=LMTransformer,
            model_args_cls=LMTransformerArgs,
        )
        model = model.to(dtype=torch.bfloat16)

        model.eval()
        generator = PackedCausalTransformerGenerator(cfg.generator, model, tokenizer)
        generator.batch_size = cfg.batch_size
        generator.max_gen_len = cfg.output_length

        prompts = [[1 for _ in range(cfg.input_length)] for _ in range(cfg.num_samples)]

        results = []
        latencies = []
        for _ in range(4):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = model(
                    torch.tensor(prompts, dtype=torch.int64, device=next(model.parameters()).device)[: generator.batch_size]
                )
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start_time) * 1000)

        for i in range(0, len(prompts), generator.batch_size):
            batch = prompts[i : i + generator.batch_size]
            result = generator.generate_benchmark(batch)
            results.append(result)

        assert len(results) > 1, "The first result is the warmup result"

        df = pd.DataFrame(results[1:])
        averages = df.mean().round(3)
        df.loc["Average"] = averages
        # Create a DataFrame with the averages
        result_df = pd.DataFrame({"Metric": averages.index, "Average": averages.values})

        # Print the table using tabulate
        print("=" * 80)
        print(ckpt)
        print(latencies)
        print(f"Forward latency: {sum(latencies[1:]) / 3:.3f} ms")
        print(tabulate(result_df, headers="keys", tablefmt="pretty", floatfmt=".3f"))

        # Clean up
        del model
        del tokenizer
        del generator
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()


# srun -J train -N 1 --gpus-per-node 8 --exclusive python apps/main/inference.py config=apps/main/mem_cfg/infer.yaml
