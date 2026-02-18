NanoSimFormer An end-to-end Transformer-based simulator for nanopore sequencing signal data
-----

NanoSimFormer is a high-fidelity nanopore sequencing signal simulator based on Transformer architectures. It supports both DNA (R10.4.1) and Direct RNA (RNA004) signal simulation, allowing users to generate synthetic POD5 files from references or existing basecalled reads.


## 🚀 Download and Install

### System Dependencies

  * NVIDIA GPU with CUDA compute capability \>= 8.x (e.g., Ampere, Ada, or Hopper GPUs like A100, RTX 3090, RTX 4090, H100)
  * NVIDIA driver version \>= 450.80.02
  * CUDA Toolkit \>= 11.8

NanoSimFormer can be installed on Linux and has been tested on Ubuntu 22.04.

### Install from Docker 

We recommend installing NanoSimFormer using the pre-built Docker image. Ensure you have Docker and the NVIDIA Container Toolkit installed by 
following this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```shell 
docker pull chobits323/NanoSimFormer:latest
```

-----

## ⚙️ Usage

### Basic

#### Using Docker
```bash
docker run --rm -it --gpus=all --ipc=host chobits323/NanaSimFormer:latest python -m nano_signal_simulator [-h]  ...
```

#### Subcommands and Options

```text
usage: python -m nano_signal_simulator [-h] --input INPUT --output OUTPUT [--prefix PREFIX] [--basecall] --mode {Reference,Read} [--coverage COVERAGE] [--sample-reads SAMPLE_READS] [--sample-output SAMPLE_OUTPUT] [--trans-profile TRANS_PROFILE]
                                       [--gpu GPU] [--batch-size BATCH_SIZE] [--config CONFIG] --preset {ont_r1041_dna_5khz,ont_rna004_4khz} [--noise-stdv NOISE_STDV] [--duration-stdv DURATION_STDV] [--mean-read-length MEAN_READ_LENGTH]
                                       [--min-read-length MIN_READ_LENGTH] [--max-read-length MAX_READ_LENGTH] [--length-dist-mode {stat,expon}] [--seed SEED] [--version]

Nanopore sequencing signal simulator

options:
  -h, --help            show this help message and exit
  --input INPUT         FASTA file path for reference (genome or transcriptome) simulation; FASTQ file path for basecalled read simulation
  --output OUTPUT       output directory
  --prefix PREFIX       output prefix (default: simulate)
  --basecall            enable basecalling simulated reads (default: False)
  --mode {Reference,Read}
                        (Reference or Read) simulation mode
  --coverage COVERAGE   sequencing coverage (default: 1)
  --sample-reads SAMPLE_READS
                        number of reads to simulate (default: -1)
  --sample-output SAMPLE_OUTPUT
                        output sampled reads (FASTA format) from reference (default: None)
  --trans-profile TRANS_PROFILE
                        3-column TSV file for simulating transcripts with specific abundance and truncation (default: None)
  --gpu GPU             GPU device id (default: 0)
  --batch-size BATCH_SIZE
                        batch size (default: 64)
  --config CONFIG       model configuration file (default: None)
  --preset {ont_r1041_dna_5khz,ont_rna004_4khz}
                        ont platform preset
  --noise-stdv NOISE_STDV
                        noise sampler standard deviation (default: None)
  --duration-stdv DURATION_STDV
                        duration sampler standard deviation (default: None)
  --mean-read-length MEAN_READ_LENGTH
                        mean read length (default: None)
  --min-read-length MIN_READ_LENGTH
                        min read length (default: 40)
  --max-read-length MAX_READ_LENGTH
                        max read length (default: None)
  --length-dist-mode {stat,expon}
                        simulated read length using exponential distribution or statistical model derived from the HG002 R10.4.1 sample (default: stat)
  --seed SEED           random seed (default: 42)
  --version             show program's version number and exit

```

### DNA Simulation Examples (R10.4.1)

#### Reference-based Simulation
Simulate reads from a reference FASTA under a specific read number a specific sequencing coverage.

```text
python -m nano_signal_simulator --input ref.fa --output ./out --mode Reference --preset ont_r1041_dna_5khz --sample-reads 1000
```

#### Adjusting Noise and Duration parameters
Tune the stochastic properties of the synthetic signal.

```text
python -m nano_signal_simulator --input ref.fa --output ./out --mode Reference --preset ont_r1041_dna_5khz --noise-stdv 1.5 --duration-stdv 0.8
```

#### Modifying Sampled read length 
Change the mean read length or switch between statistical and exponential length distribution.

```text
python -m nano_signal_simulator --input ref.fa --output ./out --mode Reference --preset ont_r1041_dna_5khz --mean-read-length 5000 --length-dist-mode expon
```

#### Circular Reference Simulation
The simulator detects circularity via the FASTA header (circular=true) to allow reads to wrap around the end of the sequence.

```text 
>chr_example circular=true
ATCG...
```

```text 
python -m nano_signal_simulator --input circular_ref.fa --output ./out --mode Reference --preset ont_r1041_dna_5khz
```

#### Read-based simulation
Generate signals based on specific sequences in a FASTQ file. Note: read IDs in FASTQ file must be in UUID format for POD5 compatibility.

```text 
python -m nano_signal_simulator --input reads.fastq --output ./out --mode Read --preset ont_r1041_dna_5khz
```

#### Full Pipeline (Signal + Basecalling) simulation

Generate POD5 signals and automatically run Dorado to produce a FASTQ.

```text 
python -m nano_signal_simulator --input ref.fa --output ./out --mode Reference --preset ont_r1041_dna_5khz --basecall
```


### DRS Simulation Examples (RNA004)
#### Transcriptome simulation
Requires a 3-column TSV profile defining transcript_name, trunc_start, and trunc_end, like below:

| transcript_name | trunc_start | trunc_end |
| --- | --- | --- |
| ENST000001 | 0 | 1200 |

```text
python -m nano_signal_simulator --input transcriptome.fa --output ./out --mode Reference --preset ont_rna004_4khz --trans-profile profile.tsv
```

-----

## ©️ Copyright

Copyright 2026 Zexuan Zhu <zhuzx@szu.edu.cn>.<br>
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

