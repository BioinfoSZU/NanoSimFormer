import os
import random
import numpy as np
import re
import torch
import torch.nn.functional as F
import contextlib
import numpy.core.multiarray
from packaging import version
from pathlib import Path


MODEL_DIR = Path(__file__).parent / "models"
default_config_path = Path(__file__).parent / "config.toml"


def safe_globals():
    # Starting from version 2.4 PyTorch introduces a check for the objects loaded
    # with torch.load(weights_only=True). Starting from 2.6 weights_only=True becomes
    # a default and requires allowlisting of objects being loaded.
    # See: https://github.com/pytorch/pytorch/pull/137602
    # See: https://pytorch.org/docs/stable/notes/serialization.html#torch.serialization.add_safe_globals
    # See: https://github.com/huggingface/accelerate/pull/3036
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    allowlist = [np.core.multiarray._reconstruct, np.core.multiarray.scalar, np.ndarray, np.dtype]  # np.core is allowed in numpy.__version__ < 2.0
    # numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
    # all versions of numpy
    allowlist += [bytes]
    allowlist += [type(np.dtype(np.int32))]
    allowlist += [type(np.dtype(np.uint32))]
    allowlist += [type(np.dtype(np.float32))]
    allowlist += [type(np.dtype(np.float64))]

    return torch.serialization.safe_globals(allowlist)


def get_canonical_seq(seq):
    seq = seq.upper()
    seq = re.sub(r"[^ATCG]", "N", seq)
    return seq


def N_to_ACTG(read):
    return "".join(random.choice("ACGT") if base == "N" else base for base in read)


def get_strand():
    return random.choice("+-")


def split_into_chunks(seq: torch.Tensor, chunk_length: int):
    N = seq.size(0)
    B = (N + chunk_length - 1) // chunk_length  # number of chunks

    pad_len = B * chunk_length - N
    padded = F.pad(seq, (0, pad_len), value=0)

    chunks = padded.view(B, chunk_length)

    seq_lengths = torch.full((B,), chunk_length, dtype=torch.long)
    seq_lengths[-1] = chunk_length - pad_len if pad_len > 0 else chunk_length

    return chunks, seq_lengths


def split_into_chunks_random(seq: torch.Tensor, min_len: int, max_len: int, len_threshold=80):
    assert seq.dim() == 1, "Only 1D sequence supported"
    assert 0 < min_len <= max_len

    N = seq.size(0)
    chunks = []
    lengths = []

    i = 0
    while i < N:
        chunk_len = torch.randint(min_len, max_len + 1, (1,)).item()

        chunk_len = min(chunk_len, N - i)

        new_i = i + chunk_len
        if (N - new_i) < len_threshold:
            chunk_len += (N - new_i)

        chunk = seq[i:i + chunk_len]
        chunks.append(chunk)
        lengths.append(chunk_len)
        i += chunk_len

    B = len(chunks)
    max_chunk_len = max_len + len_threshold

    padded_chunks = []
    for c in chunks:
        pad_len = max_chunk_len - c.size(0)
        padded_chunks.append(F.pad(c, (0, pad_len), value=0))

    chunks_tensor = torch.stack(padded_chunks)  # [B, max_chunk_len]
    seq_lengths = torch.tensor(lengths, dtype=torch.long)

    return chunks_tensor, seq_lengths


def find_homopolymers_torch(x: torch.Tensor, x_lens: torch.Tensor, threshold: int = 5):
    B, L = x.shape
    homo_all = []

    for b in range(B):
        seq = x[b, :x_lens[b]]  # trim padding
        if seq.numel() == 0:
            homo_all.append([])
            continue

        eq = seq[1:] == seq[:-1]  # shape [L-1]

        run_starts = torch.cat([
            torch.tensor([0], device=x.device),
            (~eq).nonzero(as_tuple=True)[0] + 1
        ])

        run_ends = torch.cat([
            (~eq).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([seq.size(0)], device=x.device)
        ])

        homos = []
        for s, e in zip(run_starts.tolist(), run_ends.tolist()):
            length = e - s
            if length >= threshold:
                base = seq[s].item()
                homos.append((s, e, length, base))

        homo_all.append(homos)

    return homo_all


def exec_basecaller(output_directory, prefix, basecall_model, gpu):
    basecall_fq_path = os.path.join(output_directory, f"{prefix}.fastq")
    basecall_lib_path = "/usr/local/lib/dorado"
    lib_env = ""
    if os.path.exists(basecall_lib_path):
        lib_env = f"LD_LIBRARY_PATH={basecall_lib_path}:$LD_LIBRARY_PATH"
    basecall_command = f"{lib_env} dorado basecaller --recursive --emit-fastq --device cuda:{gpu} {basecall_model} {output_directory} > {basecall_fq_path}"
    _ = os.system(basecall_command)
