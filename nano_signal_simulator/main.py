import os
import sys
import struct
import pod5
import pickle
import mappy
import pysam
import toml
import random
import numpy as np
import torch
import torch.multiprocessing as mp
import uuid
import pandas as pd
from scipy.stats import gamma, beta, expon
from enum import Enum
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from .utils import *
from .version import __version__
from .model import Model
from argparse import ArgumentParser


class FileType(Enum):
    FASTA = 1
    FASTQ = 2
    BAM = 3


class Mode(Enum):
    REFERENCE = 1
    READ = 2


FileTypeDict = {
    '.fna': FileType.FASTA,
    '.fa': FileType.FASTA,
    '.fasta': FileType.FASTA,
    '.fq': FileType.FASTQ,
    '.fastq': FileType.FASTQ,
    '.bam': FileType.BAM,
    '.sam': FileType.BAM,
}


ModeDict = {
    'Reference': Mode.REFERENCE,
    'Read': Mode.READ,
}


def sample_read_len(read_len_dist_df, basic_mean_len, mean, num=10000, seed=42, min_read_len=40, max_read_len=None):
    read_samples = {}
    total_samples = 0
    max_k = None
    max_v = 0
    for k, v in read_len_dist_df.items():
        read_samples[k] = round(num * v)
        total_samples += read_samples[k]
        if read_samples[k] > max_v:
            max_v = read_samples[k]
            max_k = k
    rest = num - total_samples
    read_samples[max_k] += rest

    ret = []
    for k, v in read_samples.items():
        if v <= 0:
            continue
        a, b = k
        ret.extend([random.randrange(a, b) for _ in range(v)])

    ret = np.array(ret).astype(float)
    ret = (ret * mean / basic_mean_len).astype(np.int64)
    ret = np.clip(ret, min_read_len, max_read_len)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(ret)
    return ret


def sample_read_len_expon_dist(loc, scale, basic_mean_len, mean, num, seed=42, min_read_len=40, max_read_len=None):
    sample = expon.rvs(
        loc=loc, scale=scale, size=num, random_state=seed
    )
    sample = (sample * mean / basic_mean_len).astype(np.int64)
    sample = np.clip(sample, min_read_len, max_read_len)
    return sample


def reference_sampling(
        genome_seqs, genome_seq_names, genome_seq_lens, genome_total_len, is_circular_dict,
        sample_reads, coverage, sample_fasta_path,
        length_dist_mode, length_dist_model, expon_len_dist_loc, expon_len_dist_scale,
        min_read_len, max_read_len, mean_read_len, basic_mean_read_len,
        seed, preset, max_retries=30, maximum_N_ratio=0.01,
):
    sample_fasta = open(sample_fasta_path, 'w') if sample_fasta_path is not None else None

    avg_genome_len = genome_total_len / len(genome_seqs)
    if sample_reads == -1:
        seq_num = round(coverage * genome_total_len / mean_read_len)
    else:
        seq_num = sample_reads
    if mean_read_len > avg_genome_len:
        print(f'WARNING: Mean read length {mean_read_len} is larger than average genome length {avg_genome_len}.')

    try_cnt = int(float(seq_num) / 0.98)
    if length_dist_mode == 'stat':
        with open(length_dist_model, "rb") as f:
            read_len_dist_df = pickle.load(f)
        preload_sampled_lens = sample_read_len(
            read_len_dist_df,
            basic_mean_len=basic_mean_read_len,
            mean=mean_read_len,
            num=try_cnt,  # seq_num,
            seed=seed,
            min_read_len=min_read_len,
            max_read_len=max_read_len,
        )
    elif length_dist_mode == 'expon':
        preload_sampled_lens = sample_read_len_expon_dist(
            loc=expon_len_dist_loc,
            scale=expon_len_dist_scale,
            basic_mean_len=basic_mean_read_len,
            mean=mean_read_len,
            num=try_cnt,  # seq_num,
            seed=seed,
            min_read_len=min_read_len,
            max_read_len=max_read_len,
        )
    else:
        raise NotImplementedError(f'length_dist_mode {length_dist_mode} is not implemented.')

    sampled_read_seqs = []
    sampled_read_names = []
    sampled_read_lens = []
    sampled_total_len = 0

    genome_seq_lens = np.asarray(genome_seq_lens)
    genome_seq_lens_cumsum = np.cumsum(np.r_[0, genome_seq_lens])

    for sampled_read_idx in tqdm(range(try_cnt), desc="Sample Reads ..."):
        retries = 0
        while retries < max_retries:
            start_pos = random.randint(0, genome_total_len - 1)
            genome_index = np.searchsorted(genome_seq_lens_cumsum[1:], start_pos, side="right")
            start_index = start_pos - genome_seq_lens_cumsum[genome_index]
            read_length = preload_sampled_lens[sampled_read_idx]

            ref_name = genome_seq_names[genome_index]
            ref_seq = genome_seqs[genome_index]
            ref_len = genome_seq_lens[genome_index]
            is_circular = is_circular_dict.get(ref_name, False)

            if read_length > ref_len:
                retries += 1
                continue

            end_index = start_index + read_length
            if end_index <= ref_len:
                read = ref_seq[start_index:end_index]
            else:
                if is_circular:
                    wrap_len = end_index - ref_len
                    read = ref_seq[start_index:] + ref_seq[:wrap_len]
                else:
                    retries += 1
                    continue

            if "dna" in preset:
                read_strand = get_strand()
            else:
                read_strand = "+"

            if read.count('N') < int(maximum_N_ratio * read_length):
                if "N" in read:
                    read = N_to_ACTG(read)
                if read_strand == "-":
                    read = mappy.revcomp(read)

                read_name = f"{sampled_read_idx}:{ref_name}:{start_index}:{read_length}:{read_strand}"
                sampled_read_seqs.append(read)
                sampled_read_names.append(read_name)
                sampled_read_lens.append(read_length)
                if len(sampled_read_seqs) <= seq_num:
                    sampled_total_len += read_length
                if sample_fasta is not None:
                    sample_fasta.write(f">{read_name}\n{read}\n")
                    sample_fasta.flush()
                break
            else:
                retries += 1

    return sampled_read_seqs[:seq_num], sampled_read_names[:seq_num], sampled_read_lens[:seq_num], sampled_total_len


def transcript_sampling(tr_seqs, tr_seq_names, tr_seq_lens, tr_total_len, tr_profile):
    profile_df = pd.read_csv(tr_profile, sep="\t", header=0)
    tr_dict = {}
    for tid, ts in zip(tr_seq_names, tr_seqs):
        if "N" in ts:
            raise RuntimeError("Don't allow N base in transcript references")
        tr_dict[tid] = ts

    sampled_read_seqs = []
    sampled_read_names = []
    sampled_read_lens = []
    sampled_total_len = 0

    for sim_read_id in range(len(profile_df)):
        ref_seq_id = profile_df.iloc[sim_read_id]['transcript_name']
        ref_st = profile_df.iloc[sim_read_id]['trunc_start']
        ref_en = profile_df.iloc[sim_read_id]['trunc_end']
        sampled_read = tr_dict[ref_seq_id][ref_st: ref_en] + 'A' * 100

        sampled_read_seqs.append(sampled_read)
        sampled_read_names.append(f"transcript_{sim_read_id}")
        sampled_read_lens.append(len(sampled_read))
        sampled_total_len += len(sampled_read)

    return sampled_read_seqs, sampled_read_names, sampled_read_lens, sampled_total_len


def get_circular_info(path):
    is_circular_dict = {}
    for s in SeqIO.parse(path, "fasta"):
        seq_name = str(s.id)
        is_circular_dict[seq_name] = False
        desc_list = str(s.description).split(" ")
        for desc in desc_list:
            if desc.split("=")[0] == "circular" and desc.split("=")[1] == "true":
                is_circular_dict[seq_name] = True
    return is_circular_dict


def preprocess(path, _type, _mode):
    seqs = []
    seq_names = []
    seq_lens = []
    total_len = 0
    if _type == FileType.FASTA:
        for s in SeqIO.parse(path, "fasta"):
            seq = get_canonical_seq(str(s.seq))
            seqs.append(seq)
            seq_names.append(str(s.id))
            seq_lens.append(len(seq))
            total_len += len(seq)

    elif _type == FileType.FASTQ:
        for s in SeqIO.parse(path, "fastq"):
            seq = get_canonical_seq(str(s.seq))
            seqs.append(seq)
            seq_names.append(str(s.id))
            seq_lens.append(len(seq))
            total_len += len(seq)

    elif _type == FileType.BAM:
        with pysam.AlignmentFile(path) as bam:
            for read in bam:
                rid = read.query_name
                if _mode == Mode.REFERENCE:
                    # require MD tag for reference-mode and bam input
                    if read.is_reverse:
                        seq = mappy.revcomp(get_canonical_seq(read.get_reference_sequence()))
                    else:
                        seq = get_canonical_seq(read.get_reference_sequence())
                else:
                    seq = get_canonical_seq(read.query_sequence)
                seqs.append(seq)
                seq_names.append(rid)
                seq_lens.append(len(seq))
                total_len += len(seq)

    return seqs, seq_names, seq_lens, total_len


def get_trans_conv_upsample_length(length, model_type):
    if model_type == "DNA":
        length = (length - 1) * 2 + 5 - 4
        length = (length - 1) * 3 + 6 - 4
    else:
        length = (length - 1) * 2 + 4 - 2
        length = (length - 1) * 3 + 7 - 4
    return length


def sampling(method, args, shape, dtype, device, min_val=None):
    if method == 'fix':
        ret = torch.full(shape, fill_value=args['value'], dtype=dtype, device=device)
    elif method == 'gaussian':
        ret = (args['mean'] + args['stdv'] * torch.randn(shape)).to(dtype).to(device)
    elif method == 'gamma':
        dist = torch.distributions.Gamma(concentration=args['shape'], rate=1.0 / args['scale'])
        ret = dist.sample(shape).to(dtype).to(device)
    else:
        raise NotImplementedError
    if min_val is not None:
        ret = ret.clamp(min=min_val)
    return ret


def construct_path_args(dist_name, mean, stdv):
    if dist_name == "gaussian":
        return {
            'mean': mean,
            'stdv': stdv,
        }
    elif dist_name == "gamma":
        return {
            'shape': (mean / stdv) ** 2,
            'scale': stdv ** 2 / mean,
        }
    else:
        raise NotImplementedError


def syn_seq_batch_version(
        model,
        x,
        x_lens,
        x_noises,
        config,
        model_type,
        duration_stdv=None,
):
    device = x.device

    path = sampling(
        method=config['duration_params']['dist'],
        args=construct_path_args(
            dist_name=config['duration_params']['dist'],
            mean=config['duration_params']['mean'],
            stdv=config['duration_params']['stdv'] if duration_stdv is None else duration_stdv,
        ),
        shape=x.shape,
        dtype=torch.long,
        device=device,
        min_val=int(config['duration_params']['min']),
    )

    if model_type == "DNA":
        homo_all = find_homopolymers_torch(x, x_lens, threshold=7)
        for b in range(x.size(0)):
            update_vals = []
            for (homo_st, homo_en, homo_l, homo_base) in homo_all[b]:
                if homo_en < x_lens[b]:
                    update_vals.append((
                        homo_en, homo_l * 2 - path[b, homo_st:homo_en].sum()
                    ))
            for up, uv in update_vals:
                if uv > 0:
                    path[b, up] = uv

    mask = torch.zeros(
        [x.size(0), x.size(1), x.size(1)],
        dtype=torch.bool,
        device=device,
    )
    for b in range(x.size(0)):
        mask[b, :x_lens[b], :x_lens[b]] = True

    with torch.no_grad(), torch.autocast('cuda', enabled=True):
        x = model.seq_encoder(x, mask)

    max_duration_ranges = config['max_duration_ranges']
    padding_x_ext = torch.zeros((x.size(0), max_duration_ranges, 512), dtype=torch.float32, device=device)
    x_ext_lens = torch.zeros((x.size(0),), dtype=torch.long, device=device)
    for b in range(x.size(0)):
        curr_ext_len = path[b, :x_lens[b]].sum().item()
        padding_x_ext[b, None, :curr_ext_len, :] = torch.repeat_interleave(x[b, None, :x_lens[b], :], path[b, :x_lens[b]], dim=1)
        x_ext_lens[b] = curr_ext_len

    with torch.no_grad(), torch.autocast('cuda', enabled=True):
        syn_sigs_gpu = model.sig_decoder(padding_x_ext).float()
    x_ext_upsample_len = get_trans_conv_upsample_length(x_ext_lens, model_type)

    pod5_params = config['pod5_params']
    zc_pa_shift = pod5_params['zc_pa_shift']
    zc_pa_scale = pod5_params['zc_pa_scale']
    syn_sigs_gpu = syn_sigs_gpu * zc_pa_scale + zc_pa_shift
    syn_sigs_gpu = syn_sigs_gpu + torch.randn_like(syn_sigs_gpu) * x_noises.unsqueeze(1)

    return syn_sigs_gpu.cpu(), x_ext_upsample_len.cpu()


def write_pod5(signal, seq_id, seq_number, config, pod5_writer: pod5.Writer, run_info, mode):
    pod5_params = config['pod5_params']
    default_pa_scale = pod5_params['default_pa_scale']
    pa_scale = struct.unpack('>d', struct.pack('>Q', default_pa_scale))[0]
    offset_mean = pod5_params['offset_mean']
    offset_stdv = pod5_params['offset_stdv']
    offset_args = {'mean': offset_mean, 'stdv': offset_stdv}
    median_before_mean = pod5_params['median_before_mean']
    median_before_stdv = pod5_params['median_before_stdv']
    median_before_args = {'mean': median_before_mean, 'stdv': median_before_stdv}

    offset = float(sampling('gaussian', args=offset_args, shape=(1,), dtype=torch.long, device='cpu', min_val=None)[0].item())
    median_before = float(sampling('gaussian', args=median_before_args, shape=(1,), dtype=torch.float32, device='cpu', min_val=None)[0].item())
    signal_dac = np.round((signal.astype(np.float64) / pa_scale) - offset).astype(np.int16)

    pod5_writer.add_read(pod5.Read(
        read_id=uuid.uuid4() if mode == Mode.REFERENCE else uuid.UUID(seq_id),  # use seq_id when using READ mode (make sure the original read_id is UUID)
        pore=pod5.Pore(
            channel=123,
            well=1,
            pore_type='not_set',
        ),
        calibration=pod5.Calibration(
            offset=offset,
            scale=pa_scale,
        ),
        read_number=seq_number,
        start_sample=0,
        median_before=median_before,
        end_reason=pod5.EndReason(
            reason=pod5.EndReasonEnum.SIGNAL_POSITIVE,
            forced=False,
        ),
        run_info=run_info,
        signal=signal_dac,
    ))


def merge_pod5_files(tmp_paths, output_path):
    """Merge multiple temporary POD5 files into one final output."""
    with pod5.Writer(output_path, software_name="NanoSimFormer") as writer:
        for tmp in tmp_paths:
            with pod5.Reader(tmp) as reader:
                for read in reader.reads():
                    writer.add_read(read.to_read())
            os.remove(tmp)


def run_inference_single_gpu(
        gpu, model, seqs, seq_names, seq_lens, noise_stds,
        config, model_type, mode, batch_size, duration_stdv,
        output_path, seq_number_offset=0, desc_prefix="",
):
    """Run inference loop on a single GPU and write results to a POD5 file.

    This is the core inference loop extracted from main(), used both for
    single-GPU mode and as the per-worker body in multi-GPU mode.
    """
    encode_dict = {'A': 1, 'C': 2, 'G': 3, 'T': 4}

    pod5_params = config['pod5_params']
    global_run_info = pod5.RunInfo(
        acquisition_id=str(uuid.uuid4()),
        acquisition_start_time=datetime.now(),
        adc_max=pod5_params['adc_max'],
        adc_min=pod5_params['adc_min'],
        context_tags={},
        experiment_name="PGXXSX240041",
        flow_cell_id="PAS76629",
        flow_cell_product_code=pod5_params['flow_cell'],
        protocol_name=pod5_params['protocol_name'],
        protocol_run_id=str(uuid.uuid4()),
        protocol_start_time=datetime.now(),
        sample_id="simulation",
        sample_rate=pod5_params['sample_rate'],
        sequencing_kit=pod5_params['sequencing_kit'],
        sequencer_position="5B",
        sequencer_position_type="PromethION",
        software="",
        system_name="PC48B226",
        system_type="PromethION 48",
        tracking_id={},
    )

    cache = {}
    global_chunks = None
    global_chunk_lens = []
    global_chunk_noises = []
    global_ranges = []
    global_seq_number = seq_number_offset

    pod5_writer = pod5.Writer(output_path, software_name="NanoSimFormer")
    tqdm_desc = f"{desc_prefix}GPU:{gpu}" if desc_prefix else f"GPU:{gpu}"
    for idx, (__syn_seq, syn_seq_name, syn_seq_len, noise_std) in tqdm(
            enumerate(zip(seqs, seq_names, seq_lens, noise_stds)),
            total=len(seqs), desc=tqdm_desc):
        if model_type == "DNA":
            syn_seq = __syn_seq
        else:
            syn_seq = __syn_seq[::-1]

        x = torch.from_numpy(np.array([encode_dict[ch] for ch in syn_seq])).to(torch.long)
        if model_type == "DNA":
            x, x_lens = split_into_chunks(x, chunk_length=config['chunksize'])
        else:
            x, x_lens = split_into_chunks_random(x, min_len=500, max_len=config['chunksize'], len_threshold=80)

        num_chunks = x.shape[0]
        for cid in range(num_chunks):
            global_ranges.append(syn_seq_name)
        cache[syn_seq_name] = ([], num_chunks)

        if global_chunks is None:
            global_chunks = x
        else:
            global_chunks = torch.cat((global_chunks, x), dim=0)  # type: ignore
        global_chunk_lens.extend(x_lens.tolist())
        global_chunk_noises.extend([noise_std for _ in range(num_chunks)])

        if len(global_chunks) >= batch_size:
            syn_sigs, syn_sig_lens = syn_seq_batch_version(
                model=model,
                x=global_chunks[:batch_size].cuda(gpu, non_blocking=True),
                x_lens=torch.from_numpy(np.array(global_chunk_lens[:batch_size])).to(torch.long).cuda(gpu, non_blocking=True),
                x_noises=torch.from_numpy(np.array(global_chunk_noises[:batch_size])).to(torch.float32).cuda(gpu, non_blocking=True),
                config=config,
                model_type=model_type,
                duration_stdv=duration_stdv,
            )
            for b in range(len(syn_sigs)):
                curr_syn_seq_id = global_ranges[b]
                cache[curr_syn_seq_id][0].append(syn_sigs[b, :syn_sig_lens[b]])
                if len(cache[curr_syn_seq_id][0]) == cache[curr_syn_seq_id][1]:
                    final_sig = torch.cat(cache[curr_syn_seq_id][0]).to(torch.float32).numpy()
                    write_pod5(final_sig, seq_id=curr_syn_seq_id, seq_number=global_seq_number, config=config,
                               pod5_writer=pod5_writer, run_info=global_run_info, mode=mode)
                    global_seq_number += 1
                    _ = cache.pop(curr_syn_seq_id, None)
            global_chunks = global_chunks[batch_size:]
            global_chunk_lens = global_chunk_lens[batch_size:]
            global_chunk_noises = global_chunk_noises[batch_size:]
            global_ranges = global_ranges[batch_size:]

    if global_chunks is not None and len(global_chunks) > 0:
        syn_sigs, syn_sig_lens = syn_seq_batch_version(
            model=model,
            x=global_chunks.cuda(gpu, non_blocking=True),
            x_lens=torch.from_numpy(np.array(global_chunk_lens)).to(torch.long).cuda(gpu, non_blocking=True),
            x_noises=torch.from_numpy(np.array(global_chunk_noises)).to(torch.float32).cuda(gpu, non_blocking=True),
            config=config,
            model_type=model_type,
            duration_stdv=duration_stdv,
        )
        for b in range(len(syn_sigs)):
            curr_syn_seq_id = global_ranges[b]
            cache[curr_syn_seq_id][0].append(syn_sigs[b, :syn_sig_lens[b]])
            if len(cache[curr_syn_seq_id][0]) == cache[curr_syn_seq_id][1]:
                final_sig = torch.cat(cache[curr_syn_seq_id][0]).to(torch.float32).numpy()
                write_pod5(final_sig, seq_id=curr_syn_seq_id, seq_number=global_seq_number, config=config,
                           pod5_writer=pod5_writer, run_info=global_run_info, mode=mode)
                global_seq_number += 1
                _ = cache.pop(curr_syn_seq_id, None)
        if len(cache) > 0:
            print(f'WARNING - The cache on GPU:{gpu} is not empty.')

    num_written = global_seq_number - seq_number_offset
    pod5_writer.close()
    return num_written


def _gpu_worker(gpu_id, assigned_seqs, assigned_names, assigned_lens, assigned_noises,
                config, model_type, mode, batch_size, duration_stdv,
                tmp_output_path, seq_number_offset, result_queue):
    """Worker function spawned per GPU for multi-GPU inference."""
    try:
        torch.cuda.set_device(gpu_id)

        model = Model(dim=512, m_type=model_type)
        model = model.cuda(gpu_id)
        model_checkpoint_path = str(os.path.join(MODEL_DIR, config['checkpoint'], 'model.pth'))
        with safe_globals():
            checkpoint = torch.load(model_checkpoint_path, map_location=f'cuda:{gpu_id}')
        model.load_state_dict(checkpoint)
        model.eval()

        num_written = run_inference_single_gpu(
            gpu=gpu_id,
            model=model,
            seqs=assigned_seqs,
            seq_names=assigned_names,
            seq_lens=assigned_lens,
            noise_stds=assigned_noises,
            config=config,
            model_type=model_type,
            mode=mode,
            batch_size=batch_size,
            duration_stdv=duration_stdv,
            output_path=tmp_output_path,
            seq_number_offset=seq_number_offset,
        )
        result_queue.put((gpu_id, num_written, tmp_output_path, None))
    except Exception as e:
        result_queue.put((gpu_id, 0, tmp_output_path, str(e)))


def run_multigpu_inference(
        gpu_ids, seqs, seq_names, seq_lens, noise_stds,
        config, model_type, mode, batch_size, duration_stdv,
        output_directory, output_path,
):
    """Orchestrate multi-GPU inference by partitioning reads across GPUs."""
    num_gpus = len(gpu_ids)
    num_reads = len(seqs)
    print(f"Multi-GPU inference: distributing {num_reads} reads across {num_gpus} GPUs {gpu_ids}")

    # Round-robin partition reads across GPUs for load balance
    partitions = {gid: ([], [], [], []) for gid in gpu_ids}
    for i in range(num_reads):
        gid = gpu_ids[i % num_gpus]
        partitions[gid][0].append(seqs[i])
        partitions[gid][1].append(seq_names[i])
        partitions[gid][2].append(seq_lens[i])
        partitions[gid][3].append(noise_stds[i])

    # Compute seq_number offsets so global numbering is consistent
    seq_number_offsets = {}
    offset = 0
    for gid in gpu_ids:
        seq_number_offsets[gid] = offset
        offset += len(partitions[gid][0])

    # Spawn worker processes
    mp.set_start_method('spawn', force=True)
    result_queue = mp.Queue()
    processes = []
    tmp_paths = []

    for gid in gpu_ids:
        tmp_path = os.path.join(output_directory, f"tmp_gpu{gid}.pod5")
        tmp_paths.append(tmp_path)
        p_seqs, p_names, p_lens, p_noises = partitions[gid]
        p = mp.Process(
            target=_gpu_worker,
            args=(
                gid, p_seqs, p_names, p_lens, p_noises,
                config, model_type, mode, batch_size, duration_stdv,
                tmp_path, seq_number_offsets[gid], result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to finish
    results = []
    for _ in processes:
        results.append(result_queue.get())
    for p in processes:
        p.join()

    # Check for errors
    total_written = 0
    finished_tmp_paths = []
    for gpu_id, num_written, tmp_path, error in sorted(results, key=lambda r: r[0]):
        if error is not None:
            print(f"ERROR on GPU:{gpu_id}: {error}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue
        print(f"GPU:{gpu_id} simulated {num_written} reads.")
        total_written += num_written
        finished_tmp_paths.append(tmp_path)

    # Merge temporary POD5 files into final output
    print(f"Merging {len(finished_tmp_paths)} temporary POD5 files...")
    merge_pod5_files(finished_tmp_paths, output_path)
    print(f"Total simulated: {total_written} reads.")
    return total_written


def main():
    parser = ArgumentParser(prog="python -m nano_signal_simulator", description="Nanopore sequencing signal simulator")
    parser.add_argument('--input', type=str, required=True,
                        help='FASTA file path for reference (genome or transcriptome) simulation; '
                             'FASTQ file path for basecalled read simulation')
    parser.add_argument('--output', type=str, required=True, help='output directory')
    parser.add_argument('--prefix', type=str, default='simulate', help='output prefix (default: simulate)')
    parser.add_argument('--basecall', action="store_true", default=False, help='enable basecalling simulated reads (default: False)')
    parser.add_argument('--emit-bam', action='store_true', default=False, help='basecalling simulated reads are stored in BAM files (default: False)')
    parser.add_argument('--mode', type=str, choices=['Reference', 'Read'], required=True,
                        help='(Reference or Read) simulation mode')
    parser.add_argument('--coverage', type=float, default=1, help='sequencing coverage (default: 1)')
    parser.add_argument('--sample-reads', type=int, default=-1,
                        help='number of reads to simulate (default: -1)')
    parser.add_argument('--sample-output', type=str, default=None,
                        help='output sampled reads (FASTA format) from reference (default: None)')
    parser.add_argument('--trans-profile', type=str, default=None,
                        help="3-column TSV file for simulating transcripts with specific abundance and truncation (default: None)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--config', type=str, default=None, help='model configuration file (default: None)')
    parser.add_argument('--preset', type=str, required=True,
                        choices=['ont_r1041_dna_5khz', 'ont_rna004_4khz',],
                        default='ont_r1041_dna_5khz', help='ont platform preset')
    parser.add_argument('--noise-stdv', type=float, default=None, help='noise sampler standard deviation (default: None)')
    parser.add_argument('--duration-stdv', type=float, default=None, help='duration sampler standard deviation (default: None)')
    parser.add_argument('--mean-read-length', type=float, default=None, help='mean read length (default: None)')
    parser.add_argument('--min-read-length', type=int, default=40, help='min read length (default: 40)')
    parser.add_argument('--max-read-length', type=int, default=None, help='max read length (default: None)')
    parser.add_argument('--length-dist-mode', type=str, choices=['stat', 'expon'], default='stat',
                        help='simulated read length using exponential distribution or statistical model derived from the HG002 R10.4.1 sample (default: stat)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--multi-gpu', action='store_true', default=False,
                        help='enable multi-GPU inference using all available GPUs (default: False)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='comma-separated GPU ids for multi-GPU inference, e.g. "0,1,2" ')
    parser.add_argument('--version', action="version", version=f"NanoSimFormer {__version__}")
    args = parser.parse_args()

    # Resolve GPU configuration
    use_multi_gpu = args.multi_gpu or args.gpus is not None
    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
        if len(gpu_ids) == 1:
            use_multi_gpu = False
            args.gpu = gpu_ids[0]
    elif use_multi_gpu:
        gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) <= 1:
            print('WARNING: --multi-gpu specified but only 1 GPU available, falling back to single-GPU mode.')
            use_multi_gpu = False
            args.gpu = gpu_ids[0]

    gpu = args.gpu
    batch_size = args.batch_size
    # torch.set_default_device(f'cuda:{gpu}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = True

    input_path = Path(args.input)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    output_directory = Path(args.output)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, f"{args.prefix}.pod5")

    sample_fasta_path = Path(args.sample_output) if args.sample_output is not None else None

    file_type = FileTypeDict[input_path.suffix]
    mode = ModeDict[args.mode]
    if args.config is not None:
        config = toml.load(args.config)[args.preset]
    else:
        config = toml.load(default_config_path)[args.preset]
    model_type = "DNA" if args.preset == "ont_r1041_dna_5khz" else "RNA"

    seqs, seq_names, seq_lens, total_len = preprocess(input_path, _type=file_type, _mode=mode)
    if mode == Mode.REFERENCE and file_type == FileType.FASTA:
        if model_type == "DNA":
            is_circular_dict = get_circular_info(input_path)
            seqs, seq_names, seq_lens, total_len = reference_sampling(
                genome_seqs=seqs, genome_seq_names=seq_names, genome_seq_lens=seq_lens, genome_total_len=total_len,
                is_circular_dict=is_circular_dict,
                sample_reads=args.sample_reads, coverage=args.coverage, sample_fasta_path=sample_fasta_path,
                length_dist_mode=args.length_dist_mode,
                length_dist_model=os.path.join(MODEL_DIR, config['length_dist']['model']),
                expon_len_dist_loc=config['length_dist']['loc'],
                expon_len_dist_scale=config['length_dist']['scale'],
                min_read_len=args.min_read_length,
                max_read_len=args.max_read_length,
                mean_read_len=args.mean_read_length if args.mean_read_length is not None else config['length_dist']['mean_len'],
                basic_mean_read_len=config['length_dist']['basic_mean_len'],
                seed=args.seed, preset=args.preset,
            )
        else:
            if args.trans_profile is None:
                raise RuntimeError('trans_profile must be provided for transcriptome reference-based simulation')
            seqs, seq_names, seq_lens, total_len = transcript_sampling(
                tr_seqs=seqs,
                tr_seq_names=seq_names,
                tr_seq_lens=seq_lens,
                tr_total_len=total_len,
                tr_profile=args.trans_profile,
            )

    # Validate model checkpoint exists before spawning workers
    model_checkpoint_path = str(os.path.join(MODEL_DIR, config['checkpoint'], 'model.pth'))
    if not os.path.isfile(model_checkpoint_path):
        raise RuntimeError('no checkpoint found')

    num_syn_seqs = len(seqs)
    if args.noise_stdv is None:
        preload_noise_stds = sampling(
            method=config['noise_intensity_params']['dist'],
            args={
                'shape': config['noise_intensity_params']['shape'],
                'scale': config['noise_intensity_params']['scale'],
            },
            shape=(num_syn_seqs,),
            dtype=torch.float32,
            device='cpu',
            min_val=None,
        ).numpy()
    else:
        preload_noise_stds = [args.noise_stdv] * num_syn_seqs

    if use_multi_gpu:
        # -- Multi-GPU path --
        run_multigpu_inference(
            gpu_ids=gpu_ids,
            seqs=seqs,
            seq_names=seq_names,
            seq_lens=seq_lens,
            noise_stds=preload_noise_stds,
            config=config,
            model_type=model_type,
            mode=mode,
            batch_size=batch_size,
            duration_stdv=args.duration_stdv,
            output_directory=str(output_directory),
            output_path=output_path,
        )
    else:
        # -- Single-GPU path --
        model = Model(dim=512, m_type=model_type)
        model = model.cuda(gpu)
        with safe_globals():
            checkpoint = torch.load(model_checkpoint_path, map_location=f'cuda:{gpu}')
        model.load_state_dict(checkpoint)
        model.eval()

        num_written = run_inference_single_gpu(
            gpu=gpu,
            model=model,
            seqs=seqs,
            seq_names=seq_names,
            seq_lens=seq_lens,
            noise_stds=preload_noise_stds,
            config=config,
            model_type=model_type,
            mode=mode,
            batch_size=batch_size,
            duration_stdv=args.duration_stdv,
            output_path=output_path,
        )
        print(f'Simulated {num_written} reads.')

    if args.basecall:
        basecall_model = os.path.join(MODEL_DIR, config['basecall']['model'])
        exec_basecaller(output_directory, args.prefix, basecall_model, gpu_ids if gpu_ids is not None else gpu, args.emit_bam)
