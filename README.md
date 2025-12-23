# Neural Network from Scratch

My first neural network implementation built entirely from scratch in NumPy — no frameworks, just fundamentals.

## 🎯 What It Does

Trains a single-layer network to predict the sum of 3D vector components. Given inputs like `[2, 3, 4]`, the network learns to output `9`.

## 🔄 The Training Loop

1. **Forward Pass**: Multiply inputs by weights, add bias (`Wx + b`)
2. **Loss Computation**: Calculate mean squared error
3. **Optimization**: Nudge each weight up/down, keep changes that reduce loss
4. **Repeat**: Train over multiple epochs

## 🧠 Key Concepts Learned

- Matrix multiplication for batched operations
- Forward propagation through linear layers
- Loss computation (MSE)
- Basic gradient descent (numerical approximation)
- Weight vs bias roles

## 📊 Architecture

```
Input: (100, 3)  →  Weights: (3, 1)  →  Output: (100, 1)
                        ↓
                    Bias: (1,)
```

**Training Data**: 100 random 3D vectors  
**Target**: Sum of each vector's components  
**Parameters**: 3 weights + 1 bias

## 🚀 Usage

```bash
python neural_network_one.py
```

Expected output:
```
Epoch 0 | Loss: ...
[[w_x w_y w_z]]
Epoch 1 | Loss: ...
[[w_x w_y w_z]]
...
```

## 💡 Why This Matters

Every complex neural network — from GPT to diffusion models — follows this same training loop. This implementation strips away abstractions to reveal the mathematical core.

## 📝 Notes

- **Status**: Intentionally minimal
- **Next Steps**: Multiple layers, activation functions, analytical gradients
- **Purpose**: Build concrete understanding, one operation at a time

## 📄 License

MIT

---

*First step in understanding how neural networks actually work under the hood.*