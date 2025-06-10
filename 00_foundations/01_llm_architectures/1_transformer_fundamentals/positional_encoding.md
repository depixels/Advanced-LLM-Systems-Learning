### Positional Encoding

**Problem:**
The core components of the Transformer, namely the Self-Attention and Feed-Forward layers, are permutation-invariant. This means if you shuffle the input sequence, the output (before adding positional information) would essentially be a shuffled version of the original output. These layers process each position independently of its absolute or relative location within the sequence. However, the order of elements (e.g., words in a sentence) is crucial for understanding meaning.

**Solution:**
To address this, Transformers inject information about the position of each token into the input embeddings. This is done *before* the embeddings are fed into the first encoder or decoder layer. The "Positional Encoding" vector has the same dimension as the input embedding ($d_{\text{model}}$) and is added element-wise to the corresponding token embedding.

`final_embedding = token_embedding + positional_encoding`

**Standard Method (Sinusoidal Functions):**
The original Transformer paper proposed using sine and cosine functions of different frequencies to create the positional encoding vectors. This method has the advantage of being fixed (not learned) and potentially allowing the model to generalize to sequence lengths longer than those seen during training. It also allows the model to easily learn to attend by relative positions, since the encoding for position `pos + k` can be represented as a linear function of the encoding for position `pos`.

**Formula:**
For a token at position `pos` in the sequence (where `pos` is 0-indexed) and dimension `i` within the embedding vector (where `i` ranges from 0 to $d_{\text{model}}-1$), the positional encoding `PE` is calculated as follows:

$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$
$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$

Where:
*   `pos` is the position of the token in the sequence (0, 1, 2, ...).
*   `i` is the index of the dimension within the embedding vector (0, 1, 2, ..., $d_{\text{model}}/2 - 1$). Each pair of dimensions ($2i, 2i+1$) uses the same frequency but different functions (sine and cosine).
*   $d_{\text{model}}$ is the dimension of the embeddings.

**Explanation:**
*   Each dimension of the positional encoding corresponds to a sinusoid.
*   The wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. This range of frequencies allows the model to capture positional information at different granularities.
*   The choice of sine and cosine allows the positional encoding for any position `pos + k` to be represented as a linear transformation (specifically, a rotation) of the encoding at position `pos`. This property is thought to help the model learn relative positioning.

**Code Example (using NumPy):**

```python
import numpy as np

def get_positional_encoding(max_seq_len, d_model):
    """
    Generates sinusoidal positional encodings.

    Args:
      max_seq_len: Maximum sequence length.
      d_model: Dimension of the model embedding.

    Returns:
      pos_encoding: Positional encoding matrix (max_seq_len, d_model)
    """
    # Create an array of position indices [0, 1, ..., max_seq_len-1]
    # Shape: (max_seq_len, 1)
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Create an array of dimension indices [0, 1, ..., d_model-1]
    # Shape: (d_model,)
    dimensions = np.arange(d_model)

    # Calculate the denominator term (10000^(2i / d_model))
    # We calculate only for even dimensions (i ranges from 0 to d_model/2 - 1)
    # Shape: (d_model/2,)
    div_term = np.power(10000, (2 * (dimensions // 2)) / np.float64(d_model))

    # Calculate the arguments for sin and cos
    # positions / div_term broadcasts correctly due to shapes
    # (max_seq_len, 1) / (d_model/2,) -> (max_seq_len, d_model/2)
    # Need to reshape div_term to (1, d_model/2) for broadcasting if needed,
    # but numpy handles it here. Let's calculate for all dimensions first.
    angle_rates = 1 / div_term # Shape (d_model,) - note: repeats for odd/even pairs
    angle_rads = positions * angle_rates[np.newaxis, :] # Shape (max_seq_len, d_model)

    # Initialize the positional encoding matrix
    pos_encoding = np.zeros((max_seq_len, d_model))

    # Apply sin to even indices (0, 2, 4, ...)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices (1, 3, 5, ...)
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Often, the positional encoding matrix is returned with an extra dimension
    # for broadcasting across batches, e.g., (1, max_seq_len, d_model)
    # return pos_encoding[np.newaxis, :, :]
    return pos_encoding # Return as (max_seq_len, d_model) for clarity

# --- Example Usage ---
max_len = 50 # Maximum sequence length example
d_model = 128 # Embedding dimension example

positional_encodings = get_positional_encoding(max_len, d_model)

print("Positional Encoding Matrix Shape:", positional_encodings.shape)

# Optional: Visualize the encodings
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(positional_encodings, cmap='viridis')
    plt.xlabel('Embedding Dimension (i)')
    plt.xlim((0, d_model))
    plt.ylabel('Position (pos)')
    plt.ylim((max_len, 0)) # Invert y-axis to show position 0 at top
    plt.colorbar()
    plt.title('Sinusoidal Positional Encoding')
    # plt.show() # Display the plot if running locally
    print("\nVisualization created (requires matplotlib).")
except ImportError:
    print("\nMatplotlib not found. Skipping visualization.")

# Example: Adding to dummy embeddings
num_sequences = 1 # Example batch size (usually larger)
seq_length = 30 # Actual length of sequences in this batch (<= max_len)
dummy_embeddings = np.random.rand(num_sequences, seq_length, d_model)

# Add positional encodings (select up to seq_length)
final_embeddings = dummy_embeddings + positional_encodings[np.newaxis, :seq_length, :]
print("\nShape after adding positional encoding:", final_embeddings.shape)

```

**Alternative Methods:**
While sinusoidal encodings are common, other methods exist:
1.  **Learned Positional Embeddings:** Similar to word embeddings, a separate embedding matrix is created where each row corresponds to a position index. These embeddings are learned during training along with other model parameters. This is simpler but may not generalize as well to sequences longer than seen during training. Used by models like BERT.
2.  **Relative Positional Encodings:** Instead of encoding absolute positions, these methods encode the relative distance between tokens directly within the attention mechanism (e.g., Transformer-XL, T5, RoPE - Rotary Positional Embedding). RoPE, in particular, has become very popular in recent large language models.
3.  **No Positional Encoding:** Some architectures might try to implicitly learn position through other means, though explicit encoding is standard for Transformers.

Positional encoding is a fundamental technique enabling Transformers to process sequential data effectively.
