---
layout: post
title: Deep Dive into Large Scale Distributed Training (Part 1)
tag: "distributed training, Pytorch, Tensorflow, Flax"
date: 2023-11-27 18:10:00
desc: Deep dive into methods and strategies for distributed training with Pytorch code snippets
author: Bearnardd
---

## Introduction

As deep learning models get larger, traditional single-GPU systems struggle to train them within 
a reasonable timeframe. In more extreme cases, fitting the entire model into their memory becomes
a challenge. __Distributed training__ is a paradigm in which the workload is distributed among 
several nodes (e.g. GPUs, TPUs). Distributed training methods can help in both faster training as well as fitting
bigger models. In this blog post, we will take a deep dive into several such methods and strategies:
* Data Parallelism (Part 1)
* Model Parallelism (Part 1)
    * Tensor Parallelism
    * Pipeline Parallelism
* ZeRO (Part 2)
* Fully Sharded Data Parallel (Part 2)


## Data Parallelism

Data Parallelism is one of the most commonly used parallelisation techniques that involves dividing the input data batch into smaller batches and distributing each of these smaller batches to different computational units. The key takeaway is that every node shares the same model weights, and optimizer states, and performs identical computations, but on different inputs. One question that may arise is how we update model weights and keep them identical during training for each node. We can achieve that using the "AllReduce" operation. There are two main implementations of this operation. The first one is straightforward: we select a single node to be the master node. Other nodes perform computations on their part of the input and then send gradients to the master node, which performs reduction operations (such as sum, min, max) and updates the weights accordingly. Then, other nodes can fetch the updated weights from the master node, and the loop continues.

![]({{"/assets/images/data_parallel.png" | relative_url }} )

The main drawback of this implementation is the communication bottleneck that grows linearly with the number of nodes: $ (\texttt{num\_nodes} - 1) \times \texttt{input\_size}$ for both gather and fetch operations. The second implementation, called __Ring All-Reduce__, mitigates this problem because its communication cost is independent of the number of nodes in a system. As the name suggests, we arrange nodes into a ring-like structure. Each node sends a portion of its state to the right node and receives a portion of the state from the left node. Each node adds its value to the value received from the previous node, and the result is passed along the ring until it returns to the starting node. It can be decomposed into two stages, scatter-reduce and allgather operations. It is best explained by vizualization.


![]({{"/assets/images/all_reduce_setup.png" | relative_url }} )
![]({{"/assets/images/all_reduce_iter1.png" | relative_url }} )
![]({{"/assets/images/all_reduce_iter2.png" | relative_url }} )
![]({{"/assets/images/all_reduce_iter3.png" | relative_url }} )

At the end of the scatter-reduce stage, each node contains a portion of the final state. To ensure that every node possesses the complete final state, we must distribute the final values across all nodes. This can be accomplished through an allgather operation. It is nearly identical to the scatter-reduce operation, with the difference being that it overwrites the current chunk with the received one instead of reducing it. It is worth mentioning that the ring reduce can also be implemented as a hierarchical 2D ring algorithm, which reduces latency. Additionally, there is a 'Double Binary Trees' algorithm that offers logarithmic latency while maintaining full bandwidth [[1](#references)]. Let's take a look at the main loop of the original implementation of __allreduce__ operation by Baidu [[2](#references)]

```cpp
// rank is an unique id assigned to each node (0 to N - 1)
// size is the total number of nodes in MPI communicator

for (int i = 0; i < size - 1; i++) {
    // Calculate ids for left and right nodes 
    int recv_chunk = (rank - i - 1 + size) % size;
    int send_chunk = (rank - i + size) % size;
    float* segment_send = &(output[segment_ends[send_chunk] -
                                segment_sizes[send_chunk]]);

    MPI_Irecv(buffer, segment_sizes[recv_chunk],
            datatype, recv_from, 0, MPI_COMM_WORLD, &recv_req);

    MPI_Send(segment_send, segment_sizes[send_chunk],
            MPI_FLOAT, send_to, 0, MPI_COMM_WORLD);

    float *segment_update = &(output[segment_ends[recv_chunk] -
                                        segment_sizes[recv_chunk]]);

    // Wait for recv to complete before reduction
    MPI_Wait(&recv_req, &recv_status);

    reduce(segment_update, buffer, segment_sizes[recv_chunk]);
}
```

The loop involves calculating the IDs of both the node from which we receive a chunk and the node to which we will send a chunk. To receive and send data __MPI__ collectives are begin used. In the end, we perform a
reduction operation. After getting some insights about the inner workings of data parallelism let's check how we can utilize it in Pytorch with the help of __DistributedDataParallel__.



### Pytorch Implmentation

First, we need to define the function that will set up DPP. As advised by the PyTorch team, we should use the __nccl__ backend (NVIDIA Collective Communications Library) for GPU acceleration [[3](#references)].

```python
import torch.multiprocessing as mp
from torch.distributed import init_process_group


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
```

As a next step, we create the __main__ function that will be run by every GPU.

```python
def main(rank: int, world_size: num_epochs: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()
```

We have to destroy each group after the training terminates. Lastly, we kick off the training by spawning `world_size` number of processes.

```python
world_size = torch.cuda.device_count()
mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
```


As seen above, PyTorch makes it really easy to utilize data parallelism. If you want to study an end-to-end example of DPP usage, you can find it [here](https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py). 


### Jax Implementation

JAX provides utilities for performing data parallelisation on both a single device and multiple devices. Let's begin by creating a simple function that we will parallelise, along with the input data.

```python
import numpy
import jax
import jax.numpy as jnp

def dot_product(x, y):
    return jnp.dot(x, y)

batch_size = 5
dim = 3

inputs = jnp.array(np.random(batch_size, dim))
weights = jnp.array(np.random(dim))
```

In a single-device scenario, we can utilize the __jax.vmap__ (vectorized map) function. To prevent the creation of redundant copies of weights, we can specify the axes of an input array over which mapping should occur.

```python
output = jax.vmap(dot_product, in_axes=(0, None))(inputs, weights)
```

For a multi-device scenario, we can utilize __jax.pmap__ (parallelisation map) function. It is used to express single-program multiple-data programs. It will compile a function and then run it in parallel on multiple devices.

```python
num_devices = 4

# we have to split input data into batches
split_inputs = jnp.stack(jnp.split(inputs, num_devices))
output = jax.pmap(dot_product, in_axes=(0, None))(split_inputs, weights)
```


Data parallelism is suitable for many use cases, but it is not effective when dealing with models too large to fit into a single GPU/TPU instance. In such situations, we can employ a technique known as __Model Parallelism__, which will be the primary focus of the next section!

## Model Parallelism

Model parallelism is a parallelization technique used to train models that are too large to fit into the memory of a single device. It achieves this by distributing different parts of the model across multiple devices. Two best known techniques are __tensor parallelism__ and 
__pipeline parallelism__.





### Tensor Parallelism

 Tensor Parallelisation splits individual layers of the model over multiple devices and performs given computation on a chunks of an input tensor. In a nutshell, it is a parallelization technique that works at the level of individual tensor operations like GEMMs. Let's take a look at a simple example:

![]({{"/assets/images/a_b_mult.png" | relative_url }} )
![]({{"/assets/images/tensor_par_mult.png" | relative_url }} )

### Pytorch Implementation

Using tensor parallelism in Pytorch is trivial.

```python
import torch
import torch.distributed.tensor.parallel as tp
from torch.distributed._tensor import DeviceMesh
from torch.distributed import init_process_group

import os

world_size = torch.cuda.device_count()
rank = 0
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12355"
init_process_group(backend="nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(rank)

# initialize a new device mesh for TP for the given tp world size
device_mesh = DeviceMesh("cuda", torch.arange(world_size))
# colwise parallel of a Linear module
layer_one = torch.nn.Linear(8,16)
tp.parallelize_module(layer_one, device_mesh, tp.ColwiseParallel())
```


### Pipeline Parallelism

Pipeline Parallelisation splits the model vertically (different layers are assigned to different nodes) into separate stages that can be executed concurrently. Given minibatch
input we split it further into micro batches that are then pipelined for execution across nodes. Because of that we minimise the idle time of nodes. It is important to pick
good hyper-parameter `chunks` which defines how many microbatches each stage will process for a single mini-batch of data. We want to pick such `chunks` values that will minimize the size of a `bubble` which is a part of a pipeline that cannot be parallelise caused by a stage not having any work to perform since it has to wait for a data from
the previous stage.

![]({{"/assets/images/pipeline_parallel.png" | relative_url }} )

```python
import torch
import torch.nn as nn
from torch.distributed.pipeline.sync import Pipe

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
torch.distributed.rpc.init_rpc('worker', rank=0, world_size=1)

# Build pipe.
fc1 = nn.Linear(16, 8).cuda(0)
fc2 = nn.Linear(8, 4).cuda(1)
model = nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)
input_tensor = torch.rand(16, 16).cuda(0)
output_rref = model(input_tensor)
```


## References

[1] [https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4)  
[2] [https://github.com/baidu-research/baidu-allreduce](https://github.com/baidu-research/baidu-allreduce)  
[3] [https://developer.nvidia.com/nccl](https://developer.nvidia.com/nccl)  
[4] [https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/)  
[5] [https://www.mishalaskin.com/posts/data_parallel#multiple-data-points-on-one-device](https://www.mishalaskin.com/posts/data_parallel#multiple-data-points-on-one-device)  
[6] [https://pytorch.org/docs/stable/rpc.html](https://pytorch.org/docs/stable/rpc.html)
[7] [https://pytorch.org/docs/stable/pipeline.html](https://pytorch.org/docs/stable/pipeline.html)
[8] [https://huggingface.co/docs/transformers/v4.15.0/parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism)