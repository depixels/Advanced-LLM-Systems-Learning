# GPT: Generative Pre-trained Transformer Architecture

GPT, standing for Generative Pre-trained Transformer, is a family of influential language models developed by OpenAI. Unlike BERT, which uses the Transformer's Encoder stack for language understanding tasks, GPT models primarily leverage the **Transformer Decoder** stack and are renowned for their strong **text generation** capabilities. The key characteristic of GPT is its **unidirectional** or **autoregressive** nature.

## 1. Core Architecture: Transformer Decoder

GPT models are based on stacks of **Transformer Decoder** blocks. A standard Transformer Decoder block, adapted for GPT's purpose, typically includes:

1.  **Masked Multi-Head Self-Attention:** This is the core component enabling GPT to process context, but only from left-to-right.
2.  **(Cross-Attention - Often Omitted/Modified in Pre-training):** The original Transformer Decoder has a second attention mechanism to attend to the Encoder's output. In GPT's pre-training phase (where it only learns from the input text itself), there's no separate Encoder output to attend to. This layer might be omitted or adapted depending on the specific GPT variant and task (it becomes relevant in conditional generation if fine-tuned on seq2seq tasks).
3.  **Position-wise Feed-Forward Network (FFN):** Identical in structure to the one used in the Encoder (two linear transformations with a ReLU or similar activation).

Each sub-layer (Masked Self-Attention, FFN) is followed by a Residual Connection and Layer Normalization.

```
Input -> Masked MHSA -> Add & Norm -> FFN -> Add & Norm -> Output
```
*(Note: Simplified view, omits the cross-attention layer typical in seq2seq Transformers)*

## 2. Key Mechanism: Masked Self-Attention

The defining feature of GPT's attention mechanism is the **masking**. In the Scaled Dot-Product Attention calculation:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M \right)V$

*   The mask $M$ is applied before the softmax operation.
*   $M$ is typically a matrix where entries corresponding to **future positions** (positions `j > i` when calculating the attention for position `i`) are set to negative infinity (-\(\infty\)), and entries for current and past positions (`j <= i`) are set to 0.
*   Setting values to -\(\infty\) ensures that after the softmax operation, the attention weights for future tokens become zero.

**Why Masking?** This masking ensures that the prediction for a token at position `t` can only depend on the known outputs at positions less than `t` (i.e., $w_1, ..., w_{t-1}$). This maintains the **autoregressive** property, which is essential for generation: the model generates text one token at a time, based only on the tokens it has already generated (or the prompt provided). This contrasts sharply with BERT's bidirectional self-attention, which looks at the entire sequence at once.

## 3. Input Representation

GPT's input representation typically consists of the sum of:

*   **Token Embeddings:** Embeddings for the input tokens (usually subwords).
*   **Position Embeddings:** Embeddings indicating the position of each token. Like BERT, GPT models often use **learned** positional embeddings.

Special tokens like `[BOS]` (Beginning of Sequence) or `[EOS]` (End of Sequence) might be used depending on the specific implementation and task. Segment embeddings are generally not used in the standard GPT pre-training setup.

## 4. Pre-training Objective: Language Modeling

GPT models are pre-trained on vast amounts of text data using a standard **Language Modeling (LM)** objective. The goal is to predict the next token in a sequence given all the preceding tokens.

Mathematically, given a sequence of tokens $W = (w_1, w_2, ..., w_n)$, the model learns to maximize the likelihood of this sequence, which is factorized autoregressively:

$P(W) = \prod_{t=1}^{n} P(w_t | w_1, w_2, ..., w_{t-1}; \theta)$

The model's parameters $\theta$ are optimized to maximize this probability over the entire training corpus. This is achieved by minimizing the negative log-likelihood (cross-entropy loss) between the model's predicted probability distribution for the next token and the actual next token in the training data.

## 5. Fine-Tuning and Usage

*   **Fine-tuning:** Similar to BERT, pre-trained GPT models can be fine-tuned for various downstream tasks (e.g., text classification, question answering). This usually involves adding a small task-specific linear layer on top of the Transformer stack and training on labeled data. For classification, the final hidden state corresponding to the last token of the sequence is often used as input to the classification layer.
*   **Zero-shot / Few-shot Learning:** With the advent of very large GPT models (like GPT-3 and subsequent versions), a paradigm of **in-context learning** emerged. These models can often perform tasks reasonably well without any fine-tuning, simply by being prompted with a natural language description of the task and, optionally, a few examples (few-shot) or no examples (zero-shot).

## 6. Significance

GPT models demonstrated the effectiveness of large-scale, generative pre-training using the Transformer Decoder architecture. They have driven significant progress in open-ended text generation, conversational AI, and prompted the development of increasingly large and capable language models (LLMs). Their autoregressive nature makes them inherently suited for tasks where text needs to be generated sequentially.
