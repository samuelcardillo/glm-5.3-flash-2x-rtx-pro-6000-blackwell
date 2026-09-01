#!/usr/bin/env python3
"""Apply the DCP-aware sparse-indexer workspace optimization exactly once."""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple


class Edit(NamedTuple):
    old: bytes
    new: bytes
    expected_count: int = 1


CAP_HELPER_OLD = b'''def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
'''
CAP_HELPER_NEW = CAP_HELPER_OLD + b'''\n\ndef get_sparse_indexer_prefill_caps(
    vllm_config: VllmConfig, index_kpool: int
) -> tuple[int, int]:
    """Return global chunk rows and the worst-case DCP rank-local rows."""
    max_model_len = vllm_config.model_config.max_model_len
    max_num_seqs = vllm_config.scheduler_config.max_num_seqs
    dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    inputs = (
        max_model_len,
        max_num_seqs,
        index_kpool,
        dcp_world_size,
        interleave_size,
    )
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in inputs
    ):
        raise ValueError("sparse indexer workspace cap inputs must be positive integers")
    if dcp_world_size > 1 and interleave_size != 1:
        raise ValueError(
            "DCP sparse-indexer workspace requires cp_kv_cache_interleave_size=1"
        )
    compressed_rows_per_seq = max_model_len // index_kpool
    if compressed_rows_per_seq <= 0:
        raise ValueError("sparse indexer compressed rows per sequence must be positive")
    global_cap = max_num_seqs * compressed_rows_per_seq
    rank_local_rows_per_seq = (\n        compressed_rows_per_seq + dcp_world_size - 1\n    ) // dcp_world_size\n    return global_cap, max_num_seqs * rank_local_rows_per_seq
'''

KPOOL_REMAP_ANCHOR = b'''    own_block = block_table[:num_reqs, 0].index_select(0, req).to(torch.int64)\n    pos = positions[:num_actual_tokens].to(torch.int64)\n    out[:num_actual_tokens] = own_block * kpool + torch.remainder(pos, kpool)\n'''

PATCHES: dict[str, tuple[Edit, ...]] = {
    "vllm/v1/attention/backends/mla/indexer.py": (
        Edit(CAP_HELPER_OLD, CAP_HELPER_NEW),
        Edit(
            b'''        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.\n        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)\n''',
            b'''        # Workspace caps are derived after reading the indexer's compression ratio.\n''',
        ),
        Edit(
            b'''        if isinstance(self.kv_cache_spec, MLAAttentionSpec):\n            self.compress_ratio = self.kv_cache_spec.compress_ratio\n        if self.dcp_world_size > 1 and self.compress_ratio > 1:\n''',
            b'''        if isinstance(self.kv_cache_spec, MLAAttentionSpec):\n            self.compress_ratio = self.kv_cache_spec.compress_ratio\n        (\n            self.max_prefill_buffer_size,\n            self.max_prefill_local_buffer_size,\n        ) = get_sparse_indexer_prefill_caps(\n            self.vllm_config, self.compress_ratio\n        )\n        if self.dcp_world_size > 1 and self.compress_ratio > 1:\n''',
        ),
        Edit(
            b'''                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,\n                )\n''',
            b'''                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,\n                    max_local_total_seq_lens=self.max_prefill_local_buffer_size,\n                )\n''',
        ),
        Edit(
            b'''    cp_kv_cache_interleave_size: int = 1,\n) -> DeepseekV32IndexerPrefillChunkMetadata | None:\n''',
            b'''    cp_kv_cache_interleave_size: int = 1,\n    max_local_total_seq_lens: int | None = None,\n) -> DeepseekV32IndexerPrefillChunkMetadata | None:\n''',
        ),
        Edit(
            b'''        local_total_seq_lens = int(local_cu_seq_lens[-1].item())\n        max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item())\n\n    query_start_loc = (\n''',
            b'''        local_total_seq_lens = int(local_cu_seq_lens[-1].item())\n        max_local_total_seq_lens_for_chunk = int(\n            local_seq_lens.sum(dim=0).max().item()\n        )\n    else:\n        max_local_total_seq_lens_for_chunk = total_seq_lens\n\n    if max_local_total_seq_lens is not None:\n        if local_total_seq_lens > max_local_total_seq_lens:\n            raise RuntimeError(\n                f"rank-local sparse-indexer rows {local_total_seq_lens} exceed "\n                f"workspace cap {max_local_total_seq_lens}"\n            )\n        if max_local_total_seq_lens_for_chunk > max_local_total_seq_lens:\n            raise RuntimeError(\n                f"worst rank-local sparse-indexer rows "\n                f"{max_local_total_seq_lens_for_chunk} exceed workspace cap "\n                f"{max_local_total_seq_lens}"\n            )\n\n    query_start_loc = (\n''',
        ),
        Edit(
            b'''        max_local_total_seq_lens=max_local_total_seq_lens,\n''',
            b'''        max_local_total_seq_lens=max_local_total_seq_lens_for_chunk,\n''',
        ),
    ),
    "vllm/models/glm5next/nvidia/attention.py": (
        Edit(
            b'''        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size\n\n        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)\n''',
            b'''        from vllm.v1.attention.backends.mla.indexer import (\n            get_sparse_indexer_prefill_caps,\n        )\n\n        (\n            self.max_total_seq_len,\n            self.max_local_total_seq_len,\n        ) = get_sparse_indexer_prefill_caps(vllm_config, self.index_kpool)\n''',
        ),
        Edit(
            b'''            self.max_model_len,\n            self.max_total_seq_len,\n            self.topk_indices_buffer,\n''',
            b'''            self.max_model_len,\n            self.max_local_total_seq_len,\n            self.topk_indices_buffer,\n''',
        ),
    ),
    "vllm/model_executor/layers/sparse_attn_indexer_kpool.py": (
        Edit(b"        max_total_seq_len: int,\n", b"        max_local_total_seq_len: int,\n"),
        Edit(
            b"        self.max_total_seq_len = max_total_seq_len\n",
            b"        self.max_local_total_seq_len = max_local_total_seq_len\n",
        ),
        Edit(
            b"            self.max_total_seq_len,\n",
            b"            self.max_local_total_seq_len,\n",
            2,
        ),
    ),
}

ORIGINAL_SHA256 = {
    "vllm/v1/attention/backends/mla/indexer.py":
        "ef32db87f3735cf72adfac7cd8906a8951c69178e7eb1c0903965f2a1fca796c",
    "vllm/models/glm5next/nvidia/attention.py":
        "7d42294c63ea52a7845a6ad0289ae3596098f5aee857718cfed34d6dcf0418f2",
    "vllm/model_executor/layers/sparse_attn_indexer_kpool.py":
        "e7935f2a9b8ecbc75410defff571d9940ad2abf511286bf0936d0c324936df39",
}
PATCHED_SHA256 = {
    "vllm/v1/attention/backends/mla/indexer.py":
        "8d310b4782c3c9fbcf2d58e6f14236cbf3176eb0759289602a863754c1e5006c",
    "vllm/models/glm5next/nvidia/attention.py":
        "085d856ee518cc5ad0aebb3e3f69803d250b19e6fff3741b2390f178b08d0e92",
    "vllm/model_executor/layers/sparse_attn_indexer_kpool.py":
        "0324447527369e9958f30136826c7704d4dbfe62aa82957ae7fb1f29f5bec68d",
}


def transform(relative: str, data: bytes) -> bytes:
    if relative == "vllm/v1/attention/backends/mla/indexer.py":
        if data.count(KPOOL_REMAP_ANCHOR) != 1:
            raise SystemExit("pinned Kpool remap anchor drifted; refusing workspace patch")
    result = data
    for edit in PATCHES[relative]:
        old_count = result.count(edit.old)
        new_count = result.count(edit.new)
        if old_count != edit.expected_count or new_count != 0:
            raise SystemExit(
                f"source anchor mismatch: {relative} "
                f"old_count={old_count} expected={edit.expected_count} "
                f"new_count={new_count}"
            )
        result = result.replace(edit.old, edit.new)
    if relative == "vllm/v1/attention/backends/mla/indexer.py":
        if result.count(KPOOL_REMAP_ANCHOR) != 1:
            raise SystemExit("Kpool remap changed during workspace patch")
    return result


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_python(data: bytes, relative: str) -> None:
    try:
        compile(data, relative, "exec")
    except (SyntaxError, ValueError, TypeError) as error:
        raise SystemExit(f"transformed source does not compile: {relative}: {error}") from error


def _atomic_write(path: Path, data: bytes) -> None:
    source_stat = path.stat(follow_symlinks=False)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), source_stat.st_mode)
            try:
                os.fchown(handle.fileno(), source_stat.st_uid, source_stat.st_gid)
            except PermissionError as error:
                raise SystemExit(
                    f"cannot preserve ownership for {path}: {error}"
                ) from error
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def apply_overlay(
    root: Path,
    *,
    original_hashes: dict[str, str] = ORIGINAL_SHA256,
    patched_hashes: dict[str, str] = PATCHED_SHA256,
    compiler: Callable[[bytes, str], None] = validate_python,
    writer: Callable[[Path, bytes], None] = _atomic_write,
    verify_only: bool = False,
) -> str:
    """Preflight and compile all sources, then replace all three transactionally."""
    states: dict[str, str] = {}
    originals: dict[str, bytes] = {}
    transformed: dict[str, bytes] = {}

    for relative in PATCHES:
        path = root / relative
        if path.is_symlink():
            raise SystemExit(f"refusing symlink source: {path}")
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")
        data = path.read_bytes()
        current = digest(data)
        if current == original_hashes[relative]:
            states[relative] = "original"
            originals[relative] = data
            candidate = transform(relative, data)
            compiler(candidate, relative)
            if digest(candidate) != patched_hashes[relative]:
                raise SystemExit(f"transformed hash mismatch: {relative}")
            transformed[relative] = candidate
        elif current == patched_hashes[relative]:
            states[relative] = "patched"
            compiler(data, relative)
        else:
            raise SystemExit(
                f"unknown source revision: {relative} sha256={current}; refusing patch"
            )

    state_set = set(states.values())
    if len(state_set) != 1:
        raise SystemExit(f"partial sparse-indexer A2 state: {states}; refusing patch")
    state = next(iter(state_set))
    if verify_only:
        if state != "patched":
            raise SystemExit(f"installed source is not patched: {states}")
        return "verified"
    if state == "patched":
        return "already_patched"

    attempted: list[str] = []
    try:
        for relative in PATCHES:
            attempted.append(relative)
            target = root / relative
            writer(target, transformed[relative])
            committed = target.read_bytes()
            if digest(committed) != patched_hashes[relative]:
                raise RuntimeError(f"post-commit hash mismatch: {relative}")
    except BaseException as error:
        rollback_errors = []
        for relative in reversed(attempted):
            try:
                _atomic_write(root / relative, originals[relative])
            except BaseException as rollback_error:
                rollback_errors.append(f"{relative}: {rollback_error}")
        if rollback_errors:
            raise SystemExit(
                f"A2 commit failed ({error}); rollback also failed: "
                + "; ".join(rollback_errors)
            ) from error
        raise SystemExit(f"A2 commit failed and was rolled back: {error}") from error
    return "patched"


def derive_workspace_caps(
    *,
    max_model_len: int,
    max_num_seqs: int,
    index_kpool: int,
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int = 1,
) -> tuple[int, int]:
    """Return (global compressed-row chunk cap, per-rank allocation cap)."""
    inputs = (
        max_model_len,
        max_num_seqs,
        index_kpool,
        dcp_world_size,
        cp_kv_cache_interleave_size,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in inputs):
        raise ValueError("workspace cap inputs must be positive integers")
    if dcp_world_size > 1 and cp_kv_cache_interleave_size != 1:
        raise ValueError("DCP workspace cap derivation requires interleave size 1")
    compressed_rows_per_seq = max_model_len // index_kpool
    if compressed_rows_per_seq <= 0:
        raise ValueError("compressed rows per sequence must be positive")
    global_cap = max_num_seqs * compressed_rows_per_seq
    local_rows_per_seq = (
        compressed_rows_per_seq + dcp_world_size - 1
    ) // dcp_world_size
    return global_cap, max_num_seqs * local_rows_per_seq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_packages", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = apply_overlay(args.site_packages, verify_only=args.verify)
    hashes = ",".join(
        f"{relative}={digest((args.site_packages / relative).read_bytes())}"
        for relative in PATCHES
    )
    print(f"sparse_indexer_workspace={result} {hashes}")


if __name__ == "__main__":
    main()
