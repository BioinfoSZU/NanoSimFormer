NanoSimFormer An end-to-end Transformer-based simulator for nanopore sequencing signal data
-----

NanoSimFormer is a high-fidelity nanopore sequencing signal simulator built on Transformer architectures. It supports both DNA (R10.4.1) and Direct RNA (RNA004) signal simulation, enabling users to generate synthetic POD5 files from references or existing basecalled reads.

## 🚀 Download and Installation

### System Requirements

  * **GPU:** NVIDIA GPU with CUDA compute capability \>= 8.x (e.g., Ampere, Ada, or Hopper GPUs like A100, RTX 3090, RTX 4090, H100)
  * **Driver:** NVIDIA driver version \>= 450.80.02
  * **CUDA:** CUDA Toolkit \>= 11.8

NanoSimFormer is compatible with Linux and has been fully tested on Ubuntu 22.04.

### Installation via Docker 

We recommend installing NanoSimFormer using the pre-built Docker image. Ensure you have Docker and the NVIDIA Container Toolkit installed by 
following this [tutorial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```shell 
docker pull chobits323/nano-sim:latest 
```

-----

## ⚙️ Usage

### Basic

#### Quick Start with Docker
```shell
# Define your working directory
EXAMPLE_DIR="[WORKING_DIRECTORY_OF_EXAMPLE_DATA]"  # /home/user_name/NanoSimFormer/example (absolute path)

# Print the help messages
docker run --rm -it --gpus=all -v ${EXAMPLE_DIR}:${EXAMPLE_DIR} --ipc=host chobits323/nano-sim:latest python -m nano_signal_simulator [-h]  ...
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

### DNA Sequencing Simulation Examples (R10.4.1)

#### Reference-Based Simulation
Simulate reads from a reference genome (FASTA) given a specific read number or sequencing coverage.

```shell
# Simulate 1000 reads from the chromosome 22 reference.
# The output POD5 will be located at ${EXAMPLE_DIR}/DNA_R10.4.1/output/simulate.pod5
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/chr22.fasta --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Reference --sample-reads 1000 --gpu 0 --preset ont_r1041_dna_5khz 

# Simulate reads with 0.1x sequencing coverage from the chromosome 22 reference.
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/chr22.fasta --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Reference --coverage 0.1 --gpu 0 --preset ont_r1041_dna_5khz
```

#### Adjusting Noise and Duration parameters
Adjust the standard deviation of the amplitude noise or duration samplers to generate simulated signals with varying qualities.

```shell 
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/chr22.fasta --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Reference --sample-reads 1000 --gpu 0 --preset ont_r1041_dna_5khz --noise-stdv 1.0 --duration-stdv 0.8
```

#### Circular Reference-Based Simulation
NanoSimFormer detects genome circularity via the FASTA header (circular=true/false, see example FASTA below) to allow simulated reads to seamlessly wrap around the end of the sequences.

*Example FASTA format*: 
```text 
>contig_1 circular=true
ATCG...
>contig_2 circular=false
CGAA...
```

```shell  
# Simulate 1000 reads from a circular E.coli reference genome (including plasmids) 
# utilizing a custom mean read length and an exponential read length distribution.
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/ecoli.fasta --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Reference --sample-reads 1000 --gpu 0 --preset ont_r1041_dna_5khz --mean-read-length 4394 --length-dist-mode expon 
```

#### Read-based simulation
Generate signals from basecalled reads provided in a FASTQ file (one-by-one). 

**Note**: Read IDs in FASTQ file must be in UUID format for POD5 compatibility.

```shell 
# Simulate reads from the FASTQ file one by one. 
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/example.fastq --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Read --gpu 0 --preset ont_r1041_dna_5khz
```

#### Full Pipeline (Signal simulation + Basecalling) 
Use `--basecall` option to automatically run [Dorado](https://github.com/nanoporetech/dorado) that basecalling the reads into a FASTQ file after signal simulation.

```shell 
# Simulate 1000 reads from the chromosome 22 reference. 
# The basecalled FASTQ output will be saved to ${EXAMPLE_DIR}/DNA_R10.4.1/output/simulate.fastq 
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/DNA_R10.4.1/chr22.fasta --output ${EXAMPLE_DIR}/DNA_R10.4.1/output --mode Reference --sample-reads 1000 --gpu 0 --preset ont_r1041_dna_5khz --basecall 
```

### Direct-RNA Sequencing Simulation Examples (RNA004)
#### Transcriptome Reference-Based simulation
For Direct RNA sequencing (DRS), NanoSimFormer requires a 3-column TSV profile defining `transcript_name`, `trunc_start`, and `trunc_end` to simulate realistic transcript abundances and 5'/3' truncations. 
Each row in the profile TSV represents the metadata for a single simulated read.

*Example TSV Profile*: 

| transcript_name | trunc_start | trunc_end |
|-----------------|-------------|-----------|
| ENST000001      | 10          | 1200      |

```shell 
# Simulate DRS reads given a transcriptome reference and a custom transcript profile. 
python -m nano_signal_simulator --input ${EXAMPLE_DIR}/RNA004/trans_ref.fasta --trans-profile ${EXAMPLE_DIR}/RNA004/trans_profile.tsv --output ${EXAMPLE_DIR}/RNA004/output --mode Reference --gpu 0 --preset ont_rna004_4khz --basecall
```

-----

## 🙏 Acknowledgement 

Some code snippets used to build the model were adapted from [torchtune](https://github.com/meta-pytorch/torchtune) library. 
We also integrated some preprocessing code snippets from [seq2squiggle](https://github.com/ZKI-PH-ImageAnalysis/seq2squiggle) to handle read sampling at given sequencing coverage.

## 📖 Citation 

Please cite our publication if you use `NanoSimFormer` in your work:

```bibtex
@article{nanosimformer,
  title={NanoSimFormer: An end-to-end Transformer-based simulator for nanopore sequencing signal data},
  author={Xie, Shaohui and Ding, Lulu and Liu, Ling and Zhu, Zexuan},
  journal={bioRxiv},
  pages={2026--01},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## ©️ Copyright

Copyright 2026 Zexuan Zhu <zhuzx@szu.edu.cn>.<br>
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
