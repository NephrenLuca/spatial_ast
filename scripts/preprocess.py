"""
DeepCAD dataset preprocessing pipeline.

Reads raw DeepCAD JSON files, converts them to AST trees, validates,
serialises to token sequences, annotates with metadata, and saves
the result as Arrow files split into train/val/test.

Usage::

    python -m scripts.preprocess \
        --json_dir data/raw/cad_json \
        --split_file data/raw/data/train_val_test_split.json \
        --out_dir data/processed \
        --workers 8 \
        --max_seq_len 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import NodeType, NodeRegistry
from core.ast_node import ASTNode, reset_id_counter
from core.grammar import validate_ast
from core.serializer import ASTSerializer
from core.tokenizer import TOKEN_PAD

from compiler.emitter import IREmitter
from compiler.backend import DeepCADBackend

from data.deepcad_parser import DeepCADParser
from data.decompiler import DeepCADDecompiler
from data.meta_annotator import MetaAnnotator
from data.statistics import DatasetStatistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Per-sample processing
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ProcessResult:
    file_id: str = ""
    success: bool = False
    tokens: Optional[List[int]] = None
    depths: Optional[List[int]] = None
    types: Optional[List[int]] = None
    roles: Optional[List[int]] = None
    parents: Optional[List[int]] = None
    siblings: Optional[List[int]] = None
    geom_desc: Optional[List[List[float]]] = None
    seq_len: int = 0
    node_count: int = 0
    reject_reason: str = ""


def process_one(
    json_path: str,
    file_id: str,
    max_seq_len: int,
    check_roundtrip: bool,
) -> ProcessResult:
    """Process a single DeepCAD JSON file through the full pipeline."""
    result = ProcessResult(file_id=file_id)

    try:
        reset_id_counter()
        NodeRegistry.reset()

        # 1. Parse JSON → command list
        parser = DeepCADParser()
        commands = parser.parse_file(json_path)
        if not commands:
            result.reject_reason = "empty_commands"
            return result

        # 2. Decompile → AST
        decompiler = DeepCADDecompiler()
        ast = decompiler.decompile(commands)

        # 3. Validate AST
        val_result = validate_ast(ast)
        if not val_result.is_valid:
            result.reject_reason = f"validation:{val_result.errors[0]}"
            return result

        # 4. Serialize → tokens
        serializer = ASTSerializer(max_seq_len=max_seq_len)
        tokens, metas = serializer.serialize(ast, pad=False)

        if len(tokens) > max_seq_len:
            result.reject_reason = f"seq_too_long:{len(tokens)}"
            return result

        # 5. Round-trip check: AST → IR → commands → decompile → AST
        if check_roundtrip:
            try:
                reset_id_counter()
                emitter = IREmitter()
                ir = emitter.emit(ast)
                backend = DeepCADBackend()
                cmds_back = backend.ir_to_commands(ir)
                ast_back = DeepCADDecompiler().decompile(cmds_back)
                if not ast.structurally_equal(ast_back):
                    result.reject_reason = "roundtrip_mismatch"
                    return result
            except Exception:
                result.reject_reason = "roundtrip_error"
                return result

        # 6. Annotate with metadata
        reset_id_counter()
        NodeRegistry.reset()
        ast2 = decompiler.decompile(commands)
        annotator = MetaAnnotator(max_seq_len=max_seq_len)
        sample = annotator.annotate(ast2)

        result.success = True
        result.tokens = sample.tokens
        result.depths = sample.depths
        result.types = sample.types
        result.roles = sample.roles
        result.parents = sample.parents
        result.siblings = sample.siblings
        result.geom_desc = sample.geom_desc
        result.seq_len = sample.seq_len
        result.node_count = ast2.subtree_size

    except Exception as e:
        result.reject_reason = f"exception:{type(e).__name__}:{str(e)[:100]}"

    return result


# ═══════════════════════════════════════════════════════════════════
# Arrow serialisation
# ═══════════════════════════════════════════════════════════════════

def results_to_arrow(results: List[ProcessResult], out_path: str) -> None:
    """Write a list of successful ProcessResults to a Parquet/Arrow file."""
    successful = [r for r in results if r.success]
    if not successful:
        log.warning("No successful results to write to %s", out_path)
        return

    table = pa.table({
        "file_id":   [r.file_id for r in successful],
        "tokens":    [r.tokens for r in successful],
        "depths":    [r.depths for r in successful],
        "types":     [r.types for r in successful],
        "roles":     [r.roles for r in successful],
        "parents":   [r.parents for r in successful],
        "siblings":  [r.siblings for r in successful],
        "geom_desc": [_flatten_geom(r.geom_desc) for r in successful],
        "seq_len":   [r.seq_len for r in successful],
        "node_count": [r.node_count for r in successful],
    })
    pq.write_table(table, out_path, compression="zstd")
    log.info("Wrote %d samples to %s (%.1f MB)",
             len(successful), out_path,
             os.path.getsize(out_path) / 1024 / 1024)


def _flatten_geom(geom_desc: List[List[float]]) -> List[float]:
    """Flatten [L, 4] geometry descriptors to [L*4] for Arrow storage."""
    flat = []
    for gd in geom_desc:
        flat.extend(gd)
    return flat


# ═══════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════

def find_json_files(json_dir: str) -> Dict[str, str]:
    """Return {file_id: full_path} for all JSON files under json_dir."""
    files = {}
    for root, _, filenames in os.walk(json_dir):
        for fn in filenames:
            if fn.endswith(".json"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, json_dir)
                file_id = rel.replace("\\", "/").replace(".json", "")
                files[file_id] = full
    return files


def load_splits(split_file: str) -> Dict[str, List[str]]:
    """Load train/val/test split file."""
    with open(split_file, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    json_dir: str,
    split_file: str,
    out_dir: str,
    workers: int,
    max_seq_len: int,
    check_roundtrip: bool,
    limit: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    log.info("Scanning JSON files in %s ...", json_dir)
    all_files = find_json_files(json_dir)
    log.info("Found %d JSON files", len(all_files))

    splits = load_splits(split_file)
    log.info("Split sizes: train=%d, val=%d, test=%d",
             len(splits.get("train", [])),
             len(splits.get("validation", [])),
             len(splits.get("test", [])))

    for split_name in ("train", "validation", "test"):
        split_ids = splits.get(split_name, [])
        if limit > 0:
            split_ids = split_ids[:limit]

        tasks = []
        for fid in split_ids:
            if fid in all_files:
                tasks.append((all_files[fid], fid))

        log.info("Processing %s split: %d files (workers=%d)",
                 split_name, len(tasks), workers)

        results: List[ProcessResult] = []
        reject_counts: Dict[str, int] = {}
        stats = DatasetStatistics()

        t0 = time.time()

        if workers <= 1:
            for json_path, fid in tasks:
                r = process_one(json_path, fid, max_seq_len, check_roundtrip)
                results.append(r)
                if r.success:
                    stats.seq_lengths.append(r.seq_len)
                    stats.num_samples += 1
                else:
                    cat = r.reject_reason.split(":")[0]
                    reject_counts[cat] = reject_counts.get(cat, 0) + 1
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for json_path, fid in tasks:
                    fut = pool.submit(
                        process_one, json_path, fid, max_seq_len, check_roundtrip,
                    )
                    futures[fut] = fid

                done = 0
                for fut in as_completed(futures):
                    done += 1
                    r = fut.result()
                    results.append(r)
                    if r.success:
                        stats.seq_lengths.append(r.seq_len)
                        stats.num_samples += 1
                    else:
                        cat = r.reject_reason.split(":")[0]
                        reject_counts[cat] = reject_counts.get(cat, 0) + 1

                    if done % 5000 == 0 or done == len(tasks):
                        elapsed = time.time() - t0
                        log.info("  %s: %d/%d done (%.1fs), %d accepted, rejects: %s",
                                 split_name, done, len(tasks), elapsed,
                                 stats.num_samples, dict(reject_counts))

        elapsed = time.time() - t0

        # Save
        out_name = {"validation": "val"}.get(split_name, split_name)
        out_path = os.path.join(out_dir, f"{out_name}.parquet")
        results_to_arrow(results, out_path)

        success_count = sum(1 for r in results if r.success)
        total = len(results)
        rate = success_count / total * 100 if total else 0

        log.info("=== %s complete ===", split_name)
        log.info("  Total: %d, Accepted: %d (%.1f%%), Rejected: %d",
                 total, success_count, rate, total - success_count)
        log.info("  Reject breakdown: %s", dict(reject_counts))
        log.info("  Time: %.1fs (%.1f samples/sec)", elapsed,
                 total / elapsed if elapsed > 0 else 0)

        if stats.seq_lengths:
            log.info("  Seq length: mean=%.0f, P50=%d, P95=%d, max=%d",
                     stats.mean_seq_length,
                     stats.percentile_seq_length(50),
                     stats.percentile_seq_length(95),
                     stats.max_seq_length)

    # Final summary
    for split in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{split}.parquet")
        if os.path.exists(path):
            t = pq.read_table(path)
            log.info("Output %s: %d samples, %.1f MB",
                     split, len(t), os.path.getsize(path) / 1024 / 1024)


def main():
    parser = argparse.ArgumentParser(description="Preprocess DeepCAD dataset")
    parser.add_argument("--json_dir", default="data/raw/cad_json",
                        help="Directory containing DeepCAD JSON files")
    parser.add_argument("--split_file", default="data/raw/data/train_val_test_split.json",
                        help="Path to train/val/test split JSON")
    parser.add_argument("--out_dir", default="data/processed",
                        help="Output directory for Arrow/Parquet files")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum token sequence length")
    parser.add_argument("--no_roundtrip", action="store_true",
                        help="Skip round-trip consistency check")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit samples per split (0 = no limit, for testing)")
    args = parser.parse_args()

    run_pipeline(
        json_dir=args.json_dir,
        split_file=args.split_file,
        out_dir=args.out_dir,
        workers=args.workers,
        max_seq_len=args.max_seq_len,
        check_roundtrip=not args.no_roundtrip,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
