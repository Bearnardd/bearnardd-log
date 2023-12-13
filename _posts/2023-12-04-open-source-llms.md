---
layout: post
title: Open Source LLMs 
tag: llms, open-source, nlp
date: 2023-12-04 11:10:05
desc: Deep dive into several open source llms with implementation details 
author: Bearnardd
---

## Introduction

Everybody either uses or at least heard about ChatGPT, GPT-4, Bard of Claude. One of the connections between their traits is all of them are accessible only via closed API (blackbox) or GUI. It is important to know that there is also 
a lot of work done with LLMs in the open source landscape. In this blog post, we will take a closer look at several such models and describe the key characteristics of each of them. Here is the complete list of the LLMs that
will be included in this blog post:

* LLama2

* Mistral 7B

* Mistral Mixtral 8x7b

* Orca 2

* Tulu

Each of these models contains some interesting implementation details that make them both relatively small and efficient while
at the same time showcasing great performance both on the benchmarks and empirically.


### Mistral 7b

**Mistral 7b** is a 7b parameter model released by **Mistral.ai** team. It has a transformer based architecture with the following implementation choices:
 - Grouped-Query Attention
 - Sliding Window Attention
 - Rolling Buffer Cache
 - Byte-fallback BPE tokenizer
 - Pre-fill and chunking

 **#### Grouped-Query Attention**

 * significantly accelerates the inference speed
 * reduces the memory requirement during decoding -> allows for higher batch sizes -> higher throughput

 **Grouped-query attention** is a generalization of **Multi-query attention**, which uses a single key-value
head for all query heads. It aims to reduce computational costs and memory usage while preserving the
overall performance of **Multi-head attention**. GQA tries to find the golden mean between using distinct kv
heads for each single head, like in **MHA**, and **MQA** approach where a single kv head is being used. It
interpolates those approaches by using a single kv head per subgroup of query heads.

![]({{"/assets/images/MHA.png" | relative_url }} )
![]({{"/assets/images/MQA.png" | relative_url }} )
![]({{"/assets/images/GQA.png" | relative_url }} )

GQA-G refers to grouped-query attention with **G** groups. We can construct both MQA (by using G=1) and MHA
(by using G=H where H indicates number of attention heads) using GQA. Moreover, we can convert MH checkpoint
to GQA checkpoint by constructing each group by mean pooling all the original heads within that group


**#### Sliding Window Attention**

* allows handling longer sequences more effectively at a reduced computational cost

The main drawback of **self attention** operation is its computational complexity, which is quadratic
w.r.t to the input sequence length and its memory complexity which is linear w.r.t to input sequence length:

(a) since each token in the sequence attends to every other token in the sequence leading to $n \times n$ attention matrix that involves matrix multiplication resulting in $O({n^2 \times d})$ time complexity, where $n$ denotes the length of the input sequence and $d$ denotes the dimensionality of the embeddings.

(b) 