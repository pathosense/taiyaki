# Fork details

I (Nick) made this fork of Taiyaki to have a repo for file preparation of ONT fast5 reads using Taiyaki for subsequent Bonito (ONT) basecalling. 
First download and install both Taiyaki and Bonito from their original (and updated) repos:

## Taiyaki
* [Taiyaki repository](https://github.com/nanoporetech/taiyaki)

Taiyaki is research software for training models for basecalling Oxford Nanopore reads. 

Oxford Nanopore's devices measure the flow of ions through a nanopore, and detect changes
in that flow as molecules pass through the pore.
These signals can be highly complex and exhibit long-range dependencies, much like spoken 
or written language. Taiyaki can be used to train neural networks to understand the 
complex signal from a nanopore device, using techniques inspired by state-of-the-art
language processing.

Taiyaki is used to train the models used to basecall DNA and RNA found in Oxford Nanopore's 
Guppy basecaller (version 2.2 at time of writing). This includes the flip-flop models,
which are trained using a technique inspired by Connectionist Temporal Classification
(Graves et al 2006).

Main features:
*  Prepare data for training basecallers by remapping signal to reference sequence
*  Train neural networks for flip-flop basecalling and squiggle prediction
*  Export basecaller models for use in Guppy

Taiyaki is built on top of pytorch and is compatible with Python 3.5 or later.
It is aimed at advanced users, and it is an actively evolving research project, so
expect to get your hands dirty.

* [Bonito repository](https://github.com/nanoporetech/bonito)

A PyTorch Basecaller for Oxford Nanopore Reads.


# Contents

1. [Workflows](#workflows)<br>
        * [Steps from fast5 files to basecalling](#steps-from-fast5-files-to-basecalling)<br>
        * [Preparing a training set (Taiyaki)](#preparing-a-training-set)<br>
        * [Basecalling (Bonito)](#basecalling)<br>
2. [Environment variables](#environment-variables)
3. [CUDA](#cuda)<br>
        * [Troubleshooting](#troubleshooting)<br>
4. [Using multiple GPUs](#using-multiple-gpus)<br>
        * [How to launch training with multiple GPUs](#how-to-launch-training-with-multiple-gpus)<br>
        * [Choice of learning rates for multi-GPU training](#choice-of-learning-rates-for-multi-gpu-training)<br>
        * [Selection of GPUs](#selection-of-gpus-for-multi-gpu-training)<br>
        * [More than one multi-GPU training group on a single machine](#more-than-one-multi-gpu-training-group-on-a-single-machine)<br>


# Workflows

The paragraph below describes the steps in the workflow in more detail.

## Steps from fast5 files to basecalling

The script **bin/prepare_mapped_reads.py** prepares a file containing mapped signals. This file is the main ingredient used to train a basecalling model.

The simplest workflow looks like this. The flow runs from top to bottom and lines show the inputs required for each stage.
The scripts in the Taiyaki package are shown, as are the files they work with. All steps executed by Bonito are indicated with an *.

                       fast5 files
                      /          \
                     /            \
                    /              \
                   /   generate_per_read_params.py
                   |                |
                   |                |               fasta with reference
                   |   per-read-params file         sequence for each read
                   |   (tsv, contains shift,        (produced with get_refs_from_sam.py
                   |   scale, trim for each read)   or some other method)
                    \               |               /
                     \              |              /
                      \             |             /
                       \            |            /
                        \           |           /
                         \          |          /
                         prepare_mapped_reads.py
                         (also uses remapping flip-flop
                         model from models/)
                                    |
                                    |
                         mapped-signal-file (hdf5)
                                    |
                                    |
                             Bonito Convert *
                       (Executed using Bonito tool)
                                    |
                                    |
                               Bonito Train *
                       (Executed using Bonito tool)
                                    |
                                    |
                     trained Bonito model in directory *
                       (suitable for use by Bonito)
                                   

Each script in bin/ has lots of options, which you can find out about by reading the scripts.
Basic usage is as follows:

    bin/generate_per_read_params.py <directory containing fast5 files> --output <name of output per_read_tsv file>

    bin/get_refs_from_sam.py <genomic references fasta> <one or more SAM/BAM files> --output <name of output reference_fasta>

    bin/prepare_mapped_reads.py <directory containing fast5 files> <per_read_tsv> <output mapped_signal_file>  <file containing model for remapping>  <reference_fasta>

Some scripts mentioned also have a useful option **--limit** which limits the number of reads to be used. This allows a quick test of a workflow.


## Preparing a training set (Taiyaki)

The `prepare_mapped_reads.py` script prepares a data set to use to train a new basecaller. Each member of this data set contains:

  * The raw signal for a complete nanopore read (lifted from a fast5 file)
  * A reference sequence that is the "ground truth" for the that read
  * An alignment between the signal and the reference

As input to this script, we need a directory containing fast5 files (either single-read or multi-read) and a fasta file that contains the ground-truth reference for each read. In order to match the raw signal to the correct ground-truth sequence, the IDs in the fasta file should be the unique read ID assigned by MinKnow (these are the same IDs that Guppy uses in its fastq output). For example, a record in the fasta file might look like:

    >17296436-f2f1-4713-adaf-169ed9cf6aa6
    TATGATGTGAGCTTATATTATTAATTTTGTATCAATCTTATTTTCTAATGTATGCATTTTAATGCTATAAATTTCCTTCTAAGCACTAC...

The recommended way to produce this fasta file is as follows:

  1. Extract read IDs from a given fastq read file.

    awk '{if(NR%4==1) print $1}' ./READS.fastq | sed -e "s/^@//" > READS.txt 

  2. Extract reads from multi_read_fast5_file(s) based on a list of read_ids using the [ont_fast5_api](https://github.com/nanoporetech/ont_fast5_api#fast5_subset)  package
 
    fast5_subset -i ./fast5 -s output_directory -l READS.txt 
    
  3. Align Guppy fastq basecalls to a reference genome using Guppy Aligner (or [Minimap2](https://github.com/lh3/minimap2). This will produce one or more .sam files.
    
    ~/ont-guppy/bin/guppy_aligner -i input_directory -s alignment --align_ref REFERENCE.fasta -t 8

  4. Use the `get_refs_from_sam.py` script (Taiyaki) to extract a snippet of the reference for each mapped read. You can also filter reads by coverage.


## Convert to Bonito chunkify files
Convert taiyaki chunkify file output using Bonito's convert-data and will creat the 4 .npy files required as input for bonito train 
* chunks.npy with shape (665899, 4800)
* chunk_lengths.npy with shape (665899,) 
* references.npy with shape (665899, 400)
* reference_lengths.npy shape (665899,)

      bonito convert --chunks INT chunkify_file.hdf5 output_directory


## Training your own model (Bonito)

       bonito train [-h] [--directory DIRECTORY] [--device DEVICE] [--lr LR] [--seed SEED] [--epochs EPOCHS] [--batch BATCH] [--chunks CHUNKS] [--validation_split VALIDATION_SPLIT][--amp] [-f] training_directory config

Automatic mixed precision can be used to speed up training with the --amp flag (however apex needs to be installed manually).

For multi-gpu training use the $CUDA_VISIBLE_DEVICES environment variable to select which GPUs and add the --multi-gpu flag.

       export CUDA_VISIBLE_DEVICES=0,1,2,3
       bonito train --amp --multi-gpu --batch 256 /data/model-dir

For a model you have trainined yourself, replace dna_r9.4.1 with the model directory.

## Basecalling (Bonito)

Installation of Bonito can be found on the [Bonito](https://github.com/nanoporetech/bonito) repo. 

A PyTorch Basecaller for Oxford Nanopore Reads.

       bonito basecaller [-h] [--device DEVICE] [--weights WEIGHTS] [--beamsize BEAMSIZE] [--half] model_directory reads_directory > basecalls.fasta


# Environment variables

The environment variables `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` can have an impact on performance.
The optimal value will depend on your system and on the jobs you are running, so experiment.
As a starting point, we recommend:

    OPENBLAS_NUM_THREADS=1
    OMP_NUM_THREADS=8


# CUDA

In order to use a GPU to accelerate model training, you will need to ensure that CUDA is installed (specifically nvcc) and that CUDA-related environment variables are set.
This should be done before running `make install` described above. If you forgot to do this, just run `make install` again once everything is set up.
The Makefile will try to detect which version of CUDA is present on your system, and install matching versions of pytorch and cupy.
Taiyaki depends on pytorch version 1.2, which supports CUDA versions 9.2 and 10.0.

To see what version of CUDA will be detected and which torch and cupy packages will be installed you can run:

    make show_cuda_version

Expert users can override the detected versions on the command line. For example, you might want to do this if you are building Taiyaki on one machine to run on another.

    # Force CUDA version 9.2
    CUDA=9.2 make install

    # Override torch package, and don't install cupy at all
    TORCH=my-special-torch-package CUPY= make install

Users who install Taiyaki system-wide or into an existing activated Python environment will need to make sure CUDA and a corresponding version of PyTorch have been installed.

## Troubleshooting

During training, if this error occurs:

    AttributeError: module 'torch._C' has no attribute '_cuda_setDevice'

or any other error related to the device, it suggests that you are trying to use pytorch's CUDA functionality but that CUDA (specifically nvcc) is either not installed or not correctly set up. 

If:

    nvcc --version

returns

    -bash: nvcc: command not found

nvcc is not installed or it is not on your path.

Ensure that you have installed CUDA (check NVIDIA's intructions) and that the CUDA compiler `nvcc` is on your path.

To place cuda on your path enter the following:

    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

Once CUDA is correctly configured and you are installing Taiyaki in a new virtual environment (as recommended), you may need to run `make install` again to ensure that you have the correct pytorch package to match your CUDA version.


# Using multiple GPUs

The script **bin/train_flipflop.py** can be used in multi-GPU mode with Pytorch's DistributedDataParallel class. With N GPUs available on a single machine, we can run N processes, each
using one of the GPUs and processing different random selections from the same training data. The gradients are synchronised by averaging across the processes. The outcome is that the batch
size is larger by a factor N than the batch size in single-GPU mode.

## How to launch training with multiple GPUs

Multi-GPU training runs can be launched using the Pytorch **distributed.launch** module. For example, in a Taiyaki environment:

    python -m torch.distributed.launch --nproc_per_node=4 train_flipflop.py --lr_max 0.004 --lr_min 0.0002 taiyaki/models/mLstm_flipflop.py mapped_reads.hdf5
    
This command line launches four processes, each using a GPU. Four GPUs numbered 0,1,2,3 must be available.

Note that all command-line options for **train_flipflop.py** are used in the same way as normal, apart from **device**.

The script **workflow/test_multiGPU.sh** provides an example. Note that the line choosing GPUs (**export CUDA_VISIBLE_DEVICES...**) may need to be edited to specify the GPUs to be used on your system.

## Choice of learning rates for multi-GPU training

A higher learning rate can be used for large-batch or multi-GPU training. As a starting point, with N GPUs we recommend using a learning rate sqrt(N) times higher than used for a single GPU. With these
settings we expect to make roughly the same training progress as a single-GPU training run but in N times fewer batches. This will not always be true: as always, experiments are necessary to find the
best choice of hyperparameters. In particular, a lower learning rate than suggested by the square-root rule may be necessary in the early stages of training. One way to achieve this is by using the
command-line arguments **lr_warmup** and **warmup_batches**. Also bear in mind that the timescale for the learning rate schedule, **lr_cosine_iters** should be changed to take into account the faster progress of training.

## Selection of GPUs for multi-GPU training

The settings above use the first **nproc_per_node** GPUs available on the machine. For example, with 8 GPUs and **nproc_per_node** = 4, we will use the GPUs numbered 0,1,2,3. This selection can be altered
using the environment variable **CUDA_VISIBLE_DEVICES**. For example,

    export CUDA_VISIBLE_DEVICES="2,4,6,7"
    
will make the GPUs numbered 2,4,6,7 available to CUDA as if they were numbers 0,1,2,3.
If we then launch using the command line above (**python -m torch.distributed.launch...**), GPUs 2,4,6,7 will be used.

See below for how this applies in a SGE system.

## More than one multi-GPU training group on a single machine

Suppose that there are 8 GPUs on your machine and you want to train two models, each using 4 GPUs. Setting **CUDA_VISIBLE_DEVICES** to **"4,5,6,7"** for the second training job, you set things off,
but find that the second job fails with an error message like this

    File "./bin/train_flipflop.py", line 178, in main
        torch.distributed.init_process_group(backend='nccl')
    File "XXXXXX/taiyaki/venv/lib/python3.5/site-packages/torch/distributed/distributed_c10d.py", line 354, in init_process_group
        store, rank, world_size = next(rendezvous(url))
    File "XXXXXX/taiyaki/venv/lib/python3.5/site-packages/torch/distributed/rendezvous.py", line 143, in _env_rendezvous_handler
        store = TCPStore(master_addr, master_port, start_daemon)
    RuntimeError: Address already in use

The reason is that **torch.distributed.launch** sets up the process group with a fixed default IP address and port for communication between
processes (**master_addr** 127.0.0.1, **master_port** 29500). The two process groups are trying to use the same port.
To fix this, set off your second process group with a different address and port:

    python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.2 --master_port 29501 train_flipflop.py <command-line-options>



<p align="center">
  <img src="ONT_logo.png">
</p>

---

This is a research release provided under the terms of the Oxford Nanopore Technologies' Public Licence. 
Research releases are provided as technology demonstrators to provide early access to features or stimulate Community development of tools.
Support for this software will be minimal and is only provided directly by the developers. Feature requests, improvements, and discussions are welcome and can be implemented by forking and pull requests.
However much as we would like to rectify every issue and piece of feedback users may have, the developers may have limited resource for support of this software.
Research releases may be unstable and subject to rapid iteration by Oxford Nanopore Technologies.

© 2019 Oxford Nanopore Technologies Ltd.
Taiyaki is distributed under the terms of the Oxford Nanopore Technologies' Public Licence.
