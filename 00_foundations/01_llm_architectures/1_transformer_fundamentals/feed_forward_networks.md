###  Position-wise Feed-Forward Networks (FFN)

After the multi-head attention sub-layer in both the encoder and decoder, each position's output vector is passed through a fully connected feed-forward network (FFN). This network is applied independently and identically to each position.

**Concept:** While the attention mechanism handles capturing relationships *between* different positions in the sequence, the FFN processes the information *at each position* independently. It introduces non-linearity and allows the model to learn more complex representations of the data at that specific position, transforming the attention output into a more suitable format for the next layer or final output.

**Structure:** The FFN typically consists of two linear transformations with a non-linear activation function in between. The most common activation function used is ReLU (Rectified Linear Unit), although others like GELU (Gaussian Error Linear Unit) are also used.

1.  **First Linear Layer:** Expands the dimension from $d_{\text{model}}$ to an inner-layer dimension $d_{ff}$.
2.  **Activation Function:** Applies a non-linearity (e.g., ReLU).
3.  **Second Linear Layer:** Projects the dimension back from $d_{ff}$ to $d_{\text{model}}$.

The inner dimension $d_{ff}$ is typically larger than $d_{\text{model}}$, often $d_{ff} = 4 \times d_{\text{model}}$ (e.g., if $d_{\text{model}}=512$, then $d_{ff}=2048$).

**Formula:**

For an input vector $x$ (representing the output of the attention sub-layer for a single position), the FFN computation is:

$
\text{FFN}(x) = \text{Linear}_2(\text{Activation}(\text{Linear}_1(x)))
$

Using specific weight matrices ($W_1, W_2$) and bias vectors ($b_1, b_2$), and ReLU as the activation:

$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$

Where:
-   $x \in \mathbb{R}^{d_{\text{model}}}$ is the input vector for a specific position.
-   $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$ is the weight matrix of the first linear layer.
-   $b_1 \in \mathbb{R}^{d_{ff}}$ is the bias vector of the first linear layer.
-   $W_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$ is the weight matrix of the second linear layer.
-   $b_2 \in \mathbb{R}^{d_{\text{model}}}$ is the bias vector of the second linear layer.
-   $\max(0, \cdot)$ represents the ReLU activation function.

**Important Note:** This *exact same* FFN (with the same $W_1, b_1, W_2, b_2$) is applied independently to the vector corresponding to *each position* in the input sequence.

**Code Example (using NumPy):**

```python
import numpy as np

def relu(x):
  """ReLU activation function."""
  return np.maximum(0, x)

class PositionWiseFFN:
    def __init__(self, d_model, d_ff):
        self.d_model = d_model
        self.d_ff = d_ff

        # Simulate linear layers with random weights and biases
        # In a real model, these are learned parameters
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.random.randn(d_ff) # Bias for the first layer
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.random.randn(d_model) # Bias for the second layer

    def call(self, x):
        """
        Apply the feed-forward network position-wise.
        Args:
          x: Input tensor (batch_size, seq_len, d_model)
        Returns:
          Output tensor (batch_size, seq_len, d_model)
        """
        # First linear transformation + bias
        # (batch_size, seq_len, d_model) @ (d_model, d_ff) -> (batch_size, seq_len, d_ff)
        linear1_output = np.matmul(x, self.W1) + self.b1 # Bias b1 is broadcasted

        # Apply activation function (ReLU)
        relu_output = relu(linear1_output)

        # Second linear transformation + bias
        # (batch_size, seq_len, d_ff) @ (d_ff, d_model) -> (batch_size, seq_len, d_model)
        linear2_output = np.matmul(relu_output, self.W2) + self.b2 # Bias b2 is broadcasted

        return linear2_output

# --- Example Usage ---
batch_size = 2
seq_len = 10
d_model = 64
d_ff = 256 # Typically 4 * d_model

# Random input data (output from attention sub-layer)
attention_output = np.random.rand(batch_size, seq_len, d_model)

# Create FFN instance
ffn = PositionWiseFFN(d_model, d_ff)

# Apply FFN
ffn_output = ffn.call(attention_output)

print("Input Shape (Attention Output):", attention_output.shape)
print("FFN Output Shape:", ffn_output.shape) # Should be the same as input shape

```

This FFN component, along with multi-head attention and residual connections/layer normalization, forms the core building blocks of the Transformer architecture.
