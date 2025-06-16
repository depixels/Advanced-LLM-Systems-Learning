# 01: LLM Architectures

This section explores the architectural foundations of Large Language Models (LLMs). We begin with the seminal Transformer architecture, dissecting its core components, and then move towards understanding its variants, scaling properties, analysis methods, and key optimizations.

## Table of Contents

### 1. Transformer Fundamentals

The foundational concepts of the Transformer model, which revolutionized sequence modeling.

*   [**Attention Mechanism**](./1_transformer_fundamentals/attention_mechanism.md): Understanding the core Self-Attention, Multi-Head Attention, and Cross-Attention mechanisms.
*   [**Feed-Forward Networks**](./1_transformer_fundamentals/feed_forward_networks.md): Exploring the position-wise feed-forward layers within the Transformer block.
*   [**Layer Normalization**](./1_transformer_fundamentals/layer_normalization.md): How normalization is applied to stabilize training and improve performance.
*   [**Positional Encoding**](./1_transformer_fundamentals/positional_encoding.md): Techniques used to inject sequence order information into the model.

### 2. Transformer Variants

Exploring architectures derived from or inspired by the original Transformer, focusing on how they adapt the core design for different tasks, scales, or efficiencies.

*   [**BERT (Bidirectional Encoder Representations from Transformers)**](./2_transformer_variants/bert_bidirectional.md): Learn how BERT utilizes a bidirectional encoder to understand context.
*   [**GPT (Generative Pre-trained Transformer) Architecture**](./2_transformer_variants/gpt_architecture.md): Explore the auto-regressive, decoder-only architecture of the GPT series models.
*   [**LLaMA Innovations**](./2_transformer_variants/llama_innovations.md): Review the architectural adjustments and training improvements introduced in LLaMA models.
*   [**MoE (Mixture of Experts) Architectures**](./2_transformer_variants/moe_architectures.md): Investigate how MoE enables model scaling through sparsely activated expert networks.
*   [**PaLM (Pathways Language Model)**](./2_transformer_variants/palm_pathways.md): Understand how PaLM leverages the Pathways system for large-scale efficient training.
*   [**T5 (Text-to-Text Transfer Transformer)**](./2_transformer_variants/t5_encoder_decoder.md): Learn how T5 unifies all NLP tasks into a text-to-text format.

### 3. Scaling Laws

*(Content to be added)* - Investigating the relationship between model size, dataset size, compute, and performance.

### 4. Model Analysis

*(Content to be added)* - Techniques for understanding and interpreting the behavior and capabilities of LLMs.

### 5. Attention Optimizations

*(Content to be added)* - Methods to make the attention mechanism more computationally efficient (e.g., Sparse Attention, Linear Attention).

---

Navigate through these topics to understand the building blocks and evolution of modern LLM architectures.
