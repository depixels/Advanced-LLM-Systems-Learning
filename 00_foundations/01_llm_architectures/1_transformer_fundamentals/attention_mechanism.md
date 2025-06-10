# Attention Mechanism (Scaled Dot-Product Attention)

## 1. Introduction

The attention mechanism allows a model to focus on relevant parts of the input sequence when predicting or generating an output sequence. Instead of relying solely on the last hidden state of an encoder (like in traditional sequence-to-sequence models), attention computes a context vector based on a weighted sum of all encoder hidden states, where the weights signify the importance of each input part.

The most common form used in Transformers is the Scaled Dot-Product Attention.

## 2. Concept

The core idea is to compute attention scores based on the similarity between a "Query" vector and several "Key" vectors. These scores are then used to weight corresponding "Value" vectors.

-   **Query (Q):** Represents the current element (e.g., a word in the target sequence) seeking information.
-   **Key (K):** Represents the elements in the source sequence providing information. The similarity between a Query and a Key determines the attention weight.
-   **Value (V):** Represents the actual content or information associated with each Key.

## 3. Formula

The Scaled Dot-Product Attention is calculated as follows:


```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```


Where:
-   $Q$is the matrix of queries (shape: `[sequence_length_q, d_k]`)
-   $K$is the matrix of keys (shape: `[sequence_length_k, d_k]`)
-   $V$is the matrix of values (shape: `[sequence_length_k, d_v]`)
-   $d_k$is the dimension of the keys (and queries). Scaling by $\sqrt{d_k}$prevents the dot products from becoming too large, which could push the softmax function into regions with very small gradients.
-   $QK^T$computes the dot product similarity between each query and all keys (shape: `[sequence_length_q, sequence_length_k]`).
-   $\text{softmax}(\cdot)$is applied row-wise to the similarity scores to obtain attention weights that sum to 1 (shape: `[sequence_length_q, sequence_length_k]`).
-   The final matrix multiplication with $V$computes the weighted sum of values based on the attention weights (shape: `[sequence_length_q, d_v]`).

## 4. Code Example (using NumPy)

Here's a simplified implementation using NumPy:

```python
import numpy as np

def softmax(x, axis=-1):
  """Compute softmax values for each sets of scores in x."""
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
  """
  Calculate the attention weights and the context vector.

  Args:
    Q: Query matrix (batch_size, seq_len_q, d_k)
    K: Key matrix (batch_size, seq_len_k, d_k)
    V: Value matrix (batch_size, seq_len_k, d_v)
    mask: Optional mask (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, 1, 1, seq_len_k)

  Returns:
    output: Context vector (batch_size, seq_len_q, d_v)
    attn_weights: Attention weights (batch_size, seq_len_q, seq_len_k)
  """
  d_k = Q.shape[-1]

  # 1. Calculate Similarity Scores: MatMul(Q, K.T)
  # (batch_size, seq_len_q, d_k) @ (batch_size, d_k, seq_len_k) -> (batch_size, seq_len_q, seq_len_k)
  matmul_qk = np.matmul(Q, K.swapaxes(-2, -1))

  # 2. Scale the scores
  scaled_attention_logits = matmul_qk / np.sqrt(d_k)

  # 3. Apply mask (if provided)
  # Masking is typically used to prevent attention to padding tokens or future tokens.
  if mask is not None:
    # Add a very large negative number to masked positions so they become zero after softmax
    scaled_attention_logits += (mask * -1e9)

  # 4. Apply Softmax to get attention weights
  # (batch_size, seq_len_q, seq_len_k)
  attn_weights = softmax(scaled_attention_logits, axis=-1)

  # 5. Compute the weighted sum of Values
  # (batch_size, seq_len_q, seq_len_k) @ (batch_size, seq_len_k, d_v) -> (batch_size, seq_len_q, d_v)
  output = np.matmul(attn_weights, V)

  return output, attn_weights

# Example Usage (Simplified without batch dimension for clarity)
seq_len_q = 3 # Example query sequence length
seq_len_k = 4 # Example key/value sequence length
d_k = 8       # Dimension of keys/queries
d_v = 16      # Dimension of values

# Random Q, K, V matrices (remove batch dimension for simplicity)
Q = np.random.rand(seq_len_q, d_k)
K = np.random.rand(seq_len_k, d_k)
V = np.random.rand(seq_len_k, d_v)

# Add batch dimension for the function
Q_batch = np.expand_dims(Q, axis=0)
K_batch = np.expand_dims(K, axis=0)
V_batch = np.expand_dims(V, axis=0)

# Calculate attention
output, attn_weights = scaled_dot_product_attention(Q_batch, K_batch, V_batch)

print("Query (Q):\n", Q)
print("\nKeys (K):\n", K)
print("\nValues (V):\n", V)
print("\nAttention Weights (softmax output):\n", attn_weights[0]) # Show weights for the first batch item
print("\nOutput Context Vector:\n", output[0]) # Show output for the first batch item
print("\nOutput shape:", output.shape)
print("Attention weights shape:", attn_weights.shape)

```

## Further Considerations

### 1. Multi-Head Attention

Instead of performing a single attention function, Multi-Head Attention projects the queries, keys, and values `h` times with different, learned linear projections. Attention is then performed in parallel for each of these projected versions. The outputs are concatenated and once again projected, resulting in the final values.

**Concept:** This allows the model to jointly attend to information from different representation subspaces at different positions. A single attention head might focus on one aspect of similarity, while multiple heads can capture diverse relationships.

**Formula:**

Given Queries $Q$, Keys $K$, and Values $V$:

1.  **Linear Projections:** Project Q, K, V into `h` different subspaces using learned weight matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. \
for each head $i=1, ..., h$. Typically, $d_k = d_v = d_{\text{model}} / h$. \
    $Q_i = Q W_i^Q \\
    K_i = K W_i^K \\
    V_i = V W_i^V$

2.  **Apply Scaled Dot-Product Attention:** Apply the attention function to each projected set in parallel:
    $\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i$

3.  **Concatenate:** Concatenate the outputs from all heads:
    $\text{Concat}(\text{head}_1, ..., \text{head}_h) = \text{Concat}( \text{Attention}(Q_1, K_1, V_1), ..., \text{Attention}(Q_h, K_h, V_h) )$
    The resulting matrix has dimensions `[sequence_length_q, h * d_v]`.

4.  **Final Linear Projection:** Project the concatenated output using another learned weight matrix $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$:
    $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O $
    The final output has dimensions `[sequence_length_q, d_model]`.

**Code Example (Conceptual, often implemented within frameworks like PyTorch/TensorFlow):**

```python
import numpy as np
# Assuming scaled_dot_product_attention and softmax are defined as before

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # Dimension per head

        # Simple simulation of linear layers (weight matrices)
        # In a real scenario, these would be learned parameters
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, d_k)."""
        # x shape: (batch_size, seq_len, d_model)
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        # transpose to shape: (batch_size, num_heads, seq_len, d_k)
        return x.transpose(0, 2, 1, 3)

    def call(self, q, k, v, mask):
        batch_size = q.shape[0]

        # 1. Linear projections
        # (batch_size, seq_len_q, d_model)
        q_proj = np.matmul(q, self.W_q)
        # (batch_size, seq_len_k, d_model)
        k_proj = np.matmul(k, self.W_k)
        # (batch_size, seq_len_v, d_model) # seq_len_k == seq_len_v
        v_proj = np.matmul(v, self.W_v)

        # 2. Split heads
        # (batch_size, num_heads, seq_len_q, d_k)
        q_split = self.split_heads(q_proj, batch_size)
        # (batch_size, num_heads, seq_len_k, d_k)
        k_split = self.split_heads(k_proj, batch_size)
        # (batch_size, num_heads, seq_len_v, d_v) # d_k == d_v here
        v_split = self.split_heads(v_proj, batch_size)

        # 3. Scaled dot-product attention for each head
        # scaled_attention shape: (batch_size, num_heads, seq_len_q, d_k)
        # attention_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        # Note: Mask needs appropriate broadcasting for multi-head
        # mask shape might be (batch_size, 1, 1, seq_len_k) or (batch_size, 1, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q_split, k_split, v_split, mask)

        # 4. Concatenate heads
        # Transpose back: (batch_size, seq_len_q, num_heads, d_k)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        # Concatenate: (batch_size, seq_len_q, d_model)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        # 5. Final linear projection
        # (batch_size, seq_len_q, d_model)
        output = np.matmul(concat_attention, self.W_o)

        return output, attention_weights # Return weights for analysis if needed

# --- Example Usage ---
batch_size = 2
seq_len_q = 5
seq_len_k = 4
d_model = 64
num_heads = 8

# Random input data
q_input = np.random.rand(batch_size, seq_len_q, d_model)
k_input = np.random.rand(batch_size, seq_len_k, d_model)
v_input = np.random.rand(batch_size, seq_len_k, d_model) # Often k and v come from same sequence

# Create MultiHeadAttention instance
mha = MultiHeadAttention(d_model, num_heads)

# No mask for simplicity in this example
output, attn_weights = mha.call(q_input, k_input, v_input, mask=None)

print("Multi-Head Attention Output Shape:", output.shape)
print("Multi-Head Attention Weights Shape:", attn_weights.shape) # (batch_size, num_heads, seq_len_q, seq_len_k)

```

### 2. Masking

Masking is essential for handling sequences of varying lengths (padding) and for preventing information leakage in autoregressive models (like decoders). Masks are applied *before* the softmax step in the scaled dot-product attention calculation.

**How it works:** Positions in the attention score matrix $\frac{QK^T}{\sqrt{d_k}}$that correspond to masked elements are set to a very large negative number (e.g., -1e9 or negative infinity). When the softmax function is applied, these positions will have probabilities extremely close to zero, effectively removing their influence on the weighted sum of values.

**Types of Masks:**

1.  **Padding Mask:**
    *   **Purpose:** To ensure the model does not attend to `<pad>` tokens added to make sequences in a batch have the same length.
    *   **How:** Create a boolean matrix where `True` (or 1) indicates a padding token. This mask is added (after being multiplied by a large negative number) to the attention scores.
    *   **Shape:** Usually broadcastable to `[batch_size, 1, 1, sequence_length_k]` for self-attention or cross-attention where keys/values come from the padded sequence.

2.  **Look-Ahead Mask (or Sequence Mask):**
    *   **Purpose:** Used in decoder self-attention layers to prevent positions from attending to subsequent positions. This maintains the autoregressive property, where predicting the current token can only depend on previous tokens and the encoder output.
    *   **How:** Create an upper triangular matrix where elements above the main diagonal are `True` (or 1). This mask is combined (usually via maximum or addition) with the padding mask, if present, and then added to the attention scores.
    *   **Shape:** Usually broadcastable to `[batch_size, 1, sequence_length_q, sequence_length_k]`, where `sequence_length_q == sequence_length_k` for self-attention.

**Code Example (Creating Masks):**

```python
import numpy as np

def create_padding_mask(seq, pad_token_id=0):
  """Creates a mask for padding tokens.
  Args:
    seq: Input sequence (batch_size, seq_len) with token IDs.
    pad_token_id: The ID used for padding.
  Returns:
    mask: Boolean mask (batch_size, 1, 1, seq_len) where True indicates padding.
  """
  # seq == pad_token_id creates a boolean matrix (batch_size, seq_len)
  # Reshape for broadcasting: (batch_size, 1, 1, seq_len)
  mask = (seq == pad_token_id)[:, np.newaxis, np.newaxis, :]
  return mask # True where padding exists

def create_look_ahead_mask(size):
  """Creates a look-ahead mask for decoder self-attention.
  Args:
    size: The length of the sequence (seq_len_q).
  Returns:
    mask: Upper triangular matrix (1, 1, size, size) where True indicates positions to be masked.
  """
  # Create an upper triangle matrix of 1s (including diagonal)
  # np.triu returns the upper triangle of an array. k=1 means exclude the diagonal.
  mask = 1 - np.triu(np.ones((size, size)), k=1) # Lower triangle and diagonal are 0
  mask = (mask == 0) # Make upper triangle True (to be masked)
  # Reshape for broadcasting: (1, 1, size, size)
  return mask[np.newaxis, np.newaxis, :, :] # True for upper triangle (excluding diagonal)


# --- Example Usage ---
# Example sequence batch with padding
example_seq = np.array([
    [10, 25, 5, 0, 0],  # seq_len = 5, 2 padding tokens
    [7, 32, 18, 21, 9]  # seq_len = 5, 0 padding tokens
])
batch_size, seq_len = example_seq.shape

padding_mask = create_padding_mask(example_seq, pad_token_id=0)
look_ahead_mask = create_look_ahead_mask(seq_len)

print("Example Sequence:\n", example_seq)
print("\nPadding Mask (Batch 1, True=Pad):\n", padding_mask[0, 0, 0, :]) # Shape (1, 1, 1, 5)
print("Padding Mask (Batch 2, True=Pad):\n", padding_mask[1, 0, 0, :])

print("\nLook-Ahead Mask (Size 5x5, True=Masked):\n", look_ahead_mask[0, 0]) # Shape (1, 1, 5, 5)

# In a decoder, you often combine masks:
# combined_mask = np.maximum(padding_mask, look_ahead_mask)
# This combined mask would then be used in the scaled_dot_product_attention function.
# The mask argument in scaled_dot_product_attention(Q, K, V, mask)
# would receive this combined_mask (or just padding_mask for encoder/cross-attention).
# Inside the function: scaled_attention_logits += (mask * -1e9)

```

**Integration into `scaled_dot_product_attention`:**

The `mask` argument in the `scaled_dot_product_attention` function (shown in the previous response) handles the application. The key line is:

```python
  if mask is not None:
    # Add a very large negative number to masked positions
    # Mask is typically boolean (True where masked), multiplying by -1e9 achieves the effect.
    scaled_attention_logits += (mask * -1e9)
```

This ensures that padded positions or future positions (depending on the mask type) have near-zero attention weights after the softmax operation.

### 3. Multi-Query Attention (MQA)

**Concept:**
Multi-Query Attention is an optimization of Multi-Head Attention (MHA) primarily designed to reduce the memory bandwidth requirements and computational cost during the *decoding* (autoregressive generation) phase of Transformers. In standard MHA, each head has its own Query (Q), Key (K), and Value (V) projection matrices. In MQA, while each head still has its own Q projection, all heads *share* a single Key (K) and Value (V) projection.

**Motivation:**
During autoregressive decoding, the Keys and Values corresponding to the previously generated tokens (the "KV cache") need to be stored and accessed at each step. In MHA, the size of this cache scales with the number of heads. By sharing the K and V projections, MQA significantly reduces the size of the KV cache, leading to:
*   Faster inference due to less data being read from memory (memory bandwidth is often a bottleneck).
*   Reduced memory footprint, allowing larger models or longer contexts to fit within memory constraints.

**Structure:**

1.  **Query Projections:** Project the input Q using `h` different learned weight matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$for each head $i=1, ..., h$.
    $Q_i = Q W_i^Q$
2.  **Shared Key/Value Projections:** Project the input K and V using *single* learned weight matrices $W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.
    $K_{\text{shared}} = K W^K \\
    V_{\text{shared}} = V W^V$
    *(Note: $d_k$and $d_v$here usually refer to the dimension per head, matching the $Q_i$dimensions)*
3.  **Apply Scaled Dot-Product Attention:** Apply the attention function for each query head $Q_i$using the *shared* $K_{\text{shared}}$and $V_{\text{shared}}$:
    $\text{head}_i = \text{Attention}(Q_i, K_{\text{shared}}, V_{\text{shared}}) = \text{softmax}\left(\frac{Q_i K_{\text{shared}}^T}{\sqrt{d_k}}\right)V_{\text{shared}}$
4.  **Concatenate & Final Projection:** Concatenate the outputs from all heads and apply a final linear projection $W^O$, similar to MHA:
    $\text{MultiQuery}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$

**Comparison to MHA:**
*   **Pros:** Significantly faster inference, lower memory usage for KV cache.
*   **Cons:** Potential for slight quality degradation compared to MHA, as the representational capacity of Keys and Values is reduced (shared across heads). However, studies often show minimal impact on quality for many tasks.

**Code Conceptual Changes:**
Compared to the MHA code example, the main changes would be:
*   Only define `W_k` and `W_v` once (not per head or split dimensionally in the same way).
*   When splitting heads, only `q_proj` needs splitting into `num_heads`. `k_proj` and `v_proj` are used directly (or perhaps broadcasted) in the `scaled_dot_product_attention` call for each head. The `scaled_dot_product_attention` function itself doesn't change, but the inputs `K` and `V` passed to it would be the same for all heads.

```python
# Conceptual modification within MultiHeadAttention class for MQA
# ... (init remains similar, but W_k, W_v represent the single shared projection)

# --- Inside call method ---
# 1. Linear projections
q_proj = np.matmul(q, self.W_q) # Shape: (batch_size, seq_len_q, d_model)
# Shared K and V projections (assuming W_k, W_v now map d_model -> d_k directly)
# For simplicity, let's assume d_k = d_model / num_heads is the dimension for the shared K/V
# In practice, the exact dimensions might vary. Let's assume W_k, W_v map to d_k
# shared_W_k = np.random.randn(d_model, self.d_k) # Example shared weight
# shared_W_v = np.random.randn(d_model, self.d_k) # Example shared weight (d_v = d_k often)
# k_shared = np.matmul(k, shared_W_k) # Shape: (batch_size, seq_len_k, d_k)
# v_shared = np.matmul(v, shared_W_v) # Shape: (batch_size, seq_len_k, d_k)

# --- Simplified view ---
# Assume W_k, W_v project to d_model first, then we extract the relevant part?
# Or, more likely, W_q projects to d_model, W_k/W_v project to d_k directly.
# Let's stick to the high-level concept:
k_shared = np.matmul(k, self.W_k) # Projects k to the shared key dimension/representation
v_shared = np.matmul(v, self.W_v) # Projects v to the shared value dimension/representation

# 2. Split only Query heads
q_split = self.split_heads(q_proj, batch_size) # (batch_size, num_heads, seq_len_q, d_k)

# 3. Scaled dot-product attention (using shared K/V for all heads)
# Need to adapt scaled_dot_product_attention or loop/broadcast carefully
# Conceptually: for each head 'i' in q_split: attention(q_split[:, i, :, :], k_shared, v_shared)
# A broadcasted implementation is more efficient:
# k_shared_b = np.expand_dims(k_shared, axis=1) # (batch_size, 1, seq_len_k, d_k)
# v_shared_b = np.expand_dims(v_shared, axis=1) # (batch_size, 1, seq_len_k, d_k)
# scaled_attention, attention_weights = scaled_dot_product_attention(
#     q_split, k_shared_b, v_shared_b, mask) # K/V are broadcast across num_heads dim

# ... (rest remains similar: transpose, concatenate, final projection) ...
# Note: This code snippet is highly conceptual and needs careful implementation
#       regarding dimensions and broadcasting in a real framework.
```

### 4. Grouped-Query Attention (GQA)

**Concept:**
Grouped-Query Attention is proposed as a middle ground between Multi-Head Attention (MHA) and Multi-Query Attention (MQA). Instead of having one Key/Value pair per Query head (MHA) or one Key/Value pair shared by all Query heads (MQA), GQA groups the Query heads and assigns a single Key/Value pair to each group.

**Structure:**
Let `h` be the total number of Query heads, and `g` be the number of Key/Value head groups (where `g` is a divisor of `h`, and `1 <= g <= h`).
*   If `g == h`, GQA is equivalent to MHA.
*   If `g == 1`, GQA is equivalent to MQA.

1.  **Query Projections:** Project the input Q using `h` different learned weight matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$for each query head $i=1, ..., h$.
    $Q_i = Q W_i^Q$
2.  **Grouped Key/Value Projections:** Project the input K and V using `g` different learned weight matrices $W_j^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$and $W_j^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$for each group $j=1, ..., g$.
    $K_j = K W_j^K \\
    V_j = V W_j^V$
3.  **Assign Q heads to K/V groups:** Each Query head $Q_i$is assigned to one of the K/V groups. Typically, heads $i = (j-1) \times (h/g) + 1$to $j \times (h/g)$are assigned to group $j$.
4.  **Apply Scaled Dot-Product Attention:** Apply the attention function for each query head $Q_i$using the Key $K_j$and Value $V_j$corresponding to its assigned group $j$.
    $\text{head}_i = \text{Attention}(Q_i, K_j, V_j) \quad \text{where } Q_i \text{ belongs to group } j$
5.  **Concatenate & Final Projection:** Concatenate the outputs from all `h` heads and apply a final linear projection $W^O$, identical to MHA/MQA:
    $\text{GroupedQuery}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$

**Comparison to MHA/MQA:**
*   GQA interpolates between MHA and MQA in terms of performance and quality.
*   It reduces the KV cache size compared to MHA (though not as much as MQA).
*   It often retains quality closer to MHA compared to MQA, while still offering significant inference speedups. It provides a tunable knob (`g`, the number of K/V groups) to balance this trade-off.

**Code Conceptual Changes:**
*   Requires defining `g` sets of K/V projection weights.
*   The `split_heads` logic might need adjustment, or the attention calculation needs to route the correct K/V group to the corresponding Q heads.
*   One common implementation strategy involves replicating the K/V heads for each group to match the number of Q heads within that group before the main `scaled_dot_product_attention` call, effectively making it look like MHA temporarily but with shared weights across the replicated heads within a group.

```python
# Conceptual modification within MultiHeadAttention class for GQA
# ... (init needs num_kv_heads or num_groups 'g')

# self.num_kv_heads = g
# self.num_q_per_kv = self.num_heads // self.num_kv_heads

# W_k, W_v might now represent projections for the 'g' groups,
# e.g., shape (d_model, g * d_k) or similar, needing splitting later.
# Or define g separate weight matrices.

# --- Inside call method ---
# 1. Linear projections
q_proj = np.matmul(q, self.W_q) # (batch_size, seq_len_q, d_model)
k_proj = np.matmul(k, self.W_k) # (batch_size, seq_len_k, g * d_k) - Example shape
v_proj = np.matmul(v, self.W_v) # (batch_size, seq_len_k, g * d_k) - Example shape

# 2. Split Q heads normally, split K/V heads into groups
q_split = self.split_heads(q_proj, batch_size) # (batch_size, h, seq_len_q, d_k)

# Split K/V into 'g' groups
# k_proj_g = k_proj.reshape(batch_size, -1, self.num_kv_heads, self.d_k)
# k_split_g = k_proj_g.transpose(0, 2, 1, 3) # (batch_size, g, seq_len_k, d_k)
# v_split_g = ... (similarly)

# 3. Replicate K/V heads to match Q heads for efficient computation
# k_replicated = np.repeat(k_split_g, self.num_q_per_kv, axis=1) # (batch_size, h, seq_len_k, d_k)
# v_replicated = np.repeat(v_split_g, self.num_q_per_kv, axis=1) # (batch_size, h, seq_len_k, d_k)

# 4. Scaled dot-product attention (now looks like MHA, but K/V were shared per group)
# scaled_attention, attention_weights = scaled_dot_product_attention(
#     q_split, k_replicated, v_replicated, mask)

# ... (rest remains similar: transpose, concatenate, final projection) ...
# Note: Again, highly conceptual. Frameworks provide optimized implementations.
```

These variants, MQA and GQA, are crucial for deploying large Transformer models efficiently, especially for tasks involving long sequence generation.
