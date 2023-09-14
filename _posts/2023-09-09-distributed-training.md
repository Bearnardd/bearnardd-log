---
layout: post
title: Deep Dive into Distributed Training 
tag: "distributed training, Pytorch, Tensorflow, Flax"
date: 2023-09-09 09:18:00
desc: Deep dive into methods and strategies for distributed training with code snippets for Pytorch Tensorflow as Flax
author: Bearnardd
---

## Introduction

As deep learning models get larger, traditional single-GPU systems struggle to train them within 
a reasonable timeframe. In more extreme cases, fitting the entire model into their memory becomes
a challenge. __Distributed training__ is a paradigm in which the workload is distributed among 
several nodes (e.g. GPUs, TPUs). Distributed training methods can help in both faster training as well as fitting
bigger models. In this blog post, we will take a deep dive into several such methods and strategies:
* Data Parallelism
* Model Parallelism
    * Tensor Parallelism
    * Pipeline Parallelism
* ZeRO
* Fully Sharded Data Parallel


### Data Parallelism

Data Parallelism is one of the most commonly used parallelization techniques that involves dividing the input data batch into smaller batches and distributing each of these smaller batches to different computational units. The key takeaway is that every node shares the same model weights, and optimizer states, and performs identical computations, but on different inputs. One question that may arise is how we update model weights and keep them identical during training for each node. We can achieve that using the "AllReduce" operation. There are two main implementations of this operation. The first one is straightforward: we select a single node to be the master node. Other nodes perform computations on their part of the input and then send gradients to the master node, which performs reduction operations (such as sum, min, max) and updates the weights accordingly. Then, other nodes can fetch the updated weights from the master node, and the loop continues.

![](../../../../assets/images/data_parallel.png)

The main drawback of this implementation is the communication bottleneck that grows linearly with the number of nodes: $ (\texttt{num\_nodes} - 1) \times \texttt{input\_size}$ for both gather and fetch operations. The second implementation,
__Ring All-Reduce__, mitigates that problem as its communication cost is independent of the number of nodes in a system. As the name might suggest
we arrange nodes into a ring like structure. Every node sends part of its state to the right node and receives part of the input from the left node.

```python
import torch.nn as nn
import torch

class MyClass(nn.Module):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    m = MyClass("Bartek")
```


### Tensor Parallelism

```python
import torch.nn as nn
import torch

class MyClass(nn.Module):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    m = MyClass("Bartek")
```


### Pipeline Parallelism

```python
import torch.nn as nn
import torch

class MyClass(nn.Module):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    m = MyClass("Bartek")
```


### ZeRO Parallelism

```python
import torch.nn as nn
import torch

class MyClass(nn.Module):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    m = MyClass("Bartek")
```


### Fully Sharded Data Parallel 

```python
import torch.nn as nn
import torch

class MyClass(nn.Module):
    def __init__(self, name):
        self.name = name


if __name__ == "__main__":
    m = MyClass("Bartek")
```