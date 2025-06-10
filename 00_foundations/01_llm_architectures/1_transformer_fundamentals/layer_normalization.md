### Layer Normalization (LayerNorm)

Layer Normalization is another crucial component used within the Transformer architecture to stabilize the hidden state dynamics, speed up training, and improve generalization. It is applied within the residual connections, typically *after* each sub-layer (Multi-Head Attention or Feed-Forward Network) *before* the output is added back to the sub-layer's input (residual connection).

**Concept:** Unlike Batch Normalization which normalizes across the batch dimension, Layer Normalization normalizes across the features (embedding dimension) for *each* data sample (sequence element) independently. This makes its computation independent of other samples in the batch and the batch size, which is particularly beneficial for sequence models like Transformers where sequence lengths can vary.

**Placement in Transformer:** In the standard Transformer architecture ("Attention is All You Need"), Layer Normalization is applied within the residual connection wrapper around each sub-layer (Self-Attention and FFN):

`output = LayerNorm(x + Sublayer(x))`

Where `x` is the input to the sub-layer (e.g., the output of the previous layer or the initial embedding), and `Sublayer(x)` is the output of the Multi-Head Attention or FFN.

*Note: Some variations, like Pre-LN Transformers, apply LayerNorm *before* the sub-layer: `output = x + Sublayer(LayerNorm(x))`. This variant is often found to be more stable during training.*

**Formula:**

For a given input vector $x$ (representing the features for a single position in the sequence, i.e., a vector of size $d_{\text{model}}$), Layer Normalization first calculates the mean ($\mu$) and variance ($\sigma^2$) across the feature dimension:

1.  **Calculate Mean:**
    ```math
    \mu = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} x_i
    ```

2.  **Calculate Variance:**
    ```math
    \sigma^2 = \frac{1}{d_{\text{model}}} \sum_{i=1}^{d_{\text{model}}} (x_i - \mu)^2
    ```

3.  **Normalize:** Normalize the input vector $x$ using the calculated mean and variance. A small epsilon ($\epsilon$) is added to the variance for numerical stability (to avoid division by zero).
    ```math
    \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
    ```

4.  **Scale and Shift:** Apply learnable scale ($\gamma$) and shift ($\beta$) parameters. These parameters allow the network to learn the optimal scale and mean for the normalized outputs, potentially recovering the original representation if needed. Both $\gamma$ and $\beta$ are vectors of size $d_{\text{model}}$ and are learned during training.
    ```math
    \text{LayerNorm}(x)_i = \gamma_i \hat{x}_i + \beta_i
    ```

The final output $\text{LayerNorm}(x)$ has the same dimension as the input $x$.

**Key Difference from Batch Normalization:**
*   **BatchNorm:** Normalizes across the *batch* dimension for each feature. Statistics (mean, variance) depend on the batch.
*   **LayerNorm:** Normalizes across the *feature* dimension for each sample. Statistics are independent of the batch.

**Code Example (using NumPy):**

```python
import numpy as np

class LayerNormalization:
    def __init__(self, d_model, epsilon=1e-6):
        self.d_model = d_model
        self.epsilon = epsilon
        # Learnable parameters: scale (gamma) and shift (beta)
        # Initialized to 1 and 0 respectively, as is common practice
        self.gamma = np.ones(d_model) # scale
        self.beta = np.zeros(d_model)  # shift
        # In a real framework, gamma and beta would be trainable parameters

    def call(self, x):
        """
        Apply Layer Normalization.
        Args:
          x: Input tensor (batch_size, seq_len, d_model)
        Returns:
          Normalized tensor (batch_size, seq_len, d_model)
        """
        # Calculate mean and variance across the last dimension (features/d_model)
        # keepdims=True maintains the dimension for broadcasting
        mean = np.mean(x, axis=-1, keepdims=True) # (batch_size, seq_len, 1)
        variance = np.var(x, axis=-1, keepdims=True) # (batch_size, seq_len, 1)

        # Normalize
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon) # (batch_size, seq_len, d_model)

        # Scale and shift
        # gamma and beta are (d_model,), they broadcast correctly
        output = self.gamma * x_normalized + self.beta # (batch_size, seq_len, d_model)

        return output

# --- Example Usage ---
batch_size = 2
seq_len = 10
d_model = 64

# Random input data (e.g., x + Sublayer(x))
input_tensor = np.random.rand(batch_size, seq_len, d_model) * 10 + 5 # Example data with some mean/variance

# Create LayerNorm instance
layer_norm = LayerNormalization(d_model)

# Apply LayerNorm
normalized_output = layer_norm.call(input_tensor)

# Verify the output statistics for one sample/position (should be close to mean=0, std=1 before scale/shift)
# We'll check the normalized part before scale/shift for clarity
mean_check = np.mean(input_tensor, axis=-1, keepdims=True)
variance_check = np.var(input_tensor, axis=-1, keepdims=True)
normalized_check = (input_tensor - mean_check) / np.sqrt(variance_check + layer_norm.epsilon)

print("Input Shape:", input_tensor.shape)
print("LayerNorm Output Shape:", normalized_output.shape)
print("\nMean of normalized output (before gamma/beta) for first sample, first position (should be near 0):",
      np.mean(normalized_check[0, 0, :]))
print("Std Dev of normalized output (before gamma/beta) for first sample, first position (should be near 1):",
      np.std(normalized_check[0, 0, :]))
print("\nMean of final LayerNorm output for first sample, first position (influenced by beta):",
      np.mean(normalized_output[0, 0, :])) # Close to beta's initial value (0)
print("Std Dev of final LayerNorm output for first sample, first position (influenced by gamma):",
      np.std(normalized_output[0, 0, :])) # Close to gamma's initial value (1)

```

Layer Normalization plays a vital role in making deep Transformers trainable by ensuring that the activations within the network remain well-behaved throughout training.
