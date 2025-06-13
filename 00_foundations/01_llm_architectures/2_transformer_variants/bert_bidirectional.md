# BERT: Bidirectional Encoder Representations from Transformers (Principles)

The core idea behind BERT (Bidirectional Encoder Representations from Transformers) is to leverage the Transformer's Encoder architecture to learn **deeply bidirectional** language representations through specific pre-training tasks. Unlike unidirectional models like GPT, BERT considers both the left and right context for every token in the input sequence simultaneously.

## 1. Core Architecture: Transformer Encoder

BERT's foundational architecture is a stack of **Transformer Encoders**. A standard Transformer Encoder block consists of two main sub-layers:

*   **Multi-Head Self-Attention (MHSA):** This is key to achieving bidirectional context understanding.
*   **Position-wise Feed-Forward Network (FFN):** Applies a non-linear transformation to each position's representation.

Each sub-layer is followed by a Residual Connection and Layer Normalization.

```
Input -> MHSA -> Add & Norm -> FFN -> Add & Norm -> Output
```

### 1.1 Multi-Head Self-Attention (MHSA)

The self-attention mechanism allows the model to weigh the importance of all other words in the sequence when processing a specific word. Its core is the **Scaled Dot-Product Attention**:

$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

Where:
*   $Q$ (Query), $K$ (Key), $V$ (Value) are matrices obtained by linear transformations of the input representations.
*   $d_k$ is the dimension of the Key vectors, used for scaling the dot product to prevent vanishing gradients.
*   The `softmax` function ensures the weights sum to 1.

Since $Q, K, V$ are computed from the **entire input sequence**, the output representation for each token incorporates information from all positions in the sequence (both left and right), thus achieving **bidirectionality**.

MHSA enhances this by splitting $Q, K, V$ into multiple "heads," computing attention in parallel multiple times, and then concatenating the results followed by a final linear transformation. This allows the model to jointly attend to information from different representation subspaces at different positions.

$ text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$
$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### 1.2 Position-wise Feed-Forward Network (FFN)

The FFN further processes the output from MHSA:

$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$

It typically consists of two linear transformations with a ReLU activation function in between and is applied independently to each position in the sequence.

## 2. Input Representation

BERT's input representation is constructed by summing three types of embedding vectors:

$\text{Input Embedding} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}$

*   **Token Embeddings:** Embeddings for the tokens (words or subwords) obtained using algorithms like WordPiece.
*   **Segment Embeddings:** Used to distinguish between different sentences in the input (e.g., Sentence A and Sentence B in the NSP task). Typically, there are only two embedding vectors, $E_A$ and $E_B$.
*   **Position Embeddings:** Provide information about the token's position in the sequence. BERT uses **learned** positional embeddings, unlike the sinusoidal embeddings in the original Transformer.

**Special Tokens:**
*   `[CLS]`: Prepended to the start of the input sequence. Its final hidden state (output of the last Encoder layer) is typically used for sentence-level classification tasks (like NSP or text classification).
*   `[SEP]`: Used to separate different sentences.
*   `[MASK]`: Used to replace masked tokens in the MLM task.

## 3. Pre-training Tasks

BERT learns language representations through two unsupervised pre-training tasks:

### 3.1 Masked Language Model (MLM)

The goal of MLM is to predict the original tokens that were randomly masked in the input sequence.

*   **Masking Strategy:** Randomly select 15% of the input tokens for masking:
    *   80% of the time, replace the token with the `[MASK]` token.
    *   10% of the time, replace the token with a random other token.
    *   10% of the time, keep the original token unchanged.
*   **Objective:** The model needs to predict the **original** token at the masked positions based on the surrounding unmasked tokens (bidirectional context).
*   **Loss Function:** Typically Cross-Entropy Loss between the predicted tokens and the original tokens, calculated only for the masked positions.

**Pseudo-code Example (Masking Process):**
```python
# conceptual pseudo-code
tokens = [...] # input token sequence
output_tokens = tokens[:]
masked_indices = []

for i in range(len(tokens)):
  if random.random() < 0.15:
    masked_indices.append(i)
    prob = random.random()
    if prob < 0.8: # 80% -> [MASK]
      output_tokens[i] = "[MASK]"
    elif prob < 0.9: # 10% -> random token
      output_tokens[i] = random_token()
    # else: 10% -> keep original (no change needed)

# model input: output_tokens
# labels for MLM loss: tokens[masked_indices]
```

### 3.2 Next Sentence Prediction (NSP)

NSP is a binary classification task to determine if two input sentences (A and B) are consecutive in the original corpus.

*   **Input Format:** `[CLS] Sentence A [SEP] Sentence B [SEP]`
*   **Training Data:** In 50% of the samples, B is the actual next sentence following A (Label: `IsNext`); in the other 50%, B is a random sentence from the corpus (Label: `NotNext`).
*   **Objective:** The model uses the final hidden state $C \in \mathbb{R}^H$ corresponding to the `[CLS]` token for binary classification.
    $P(\text{IsNext} | A, B) = \text{softmax}(CW_{NSP})$
*   **Loss Function:** Binary Cross-Entropy Loss is calculated for the `IsNext`/`NotNext` prediction.

## 4. Fine-Tuning

After pre-training, the BERT model can be adapted for downstream tasks by fine-tuning it on task-specific labeled data. This typically involves adding one or more small, task-specific output layers on top of the pre-trained BERT model (e.g., a linear layer for classification, a CRF layer for sequence labeling) and then training the entire model (including BERT's parameters and the new layers) end-to-end on the task data, usually with a lower learning rate.

For example, for a text classification task, the final hidden state $C$ of the `[CLS]` token can be fed into a simple Softmax classifier:

$P(\text{class} | \text{sequence}) = \text{softmax}(CW_{\text{classify}})$
where $W_{\text{classify}}$ is the classifier's weights.

## 5. Conclusion

BERT is a powerful language model that leverages bidirectional context by using masked language modeling and next sentence prediction tasks during pre-training. Its architecture and pre-training objectives have significantly advanced the state of the art in natural language understanding tasks.
