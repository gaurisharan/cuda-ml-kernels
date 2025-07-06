# CUDA ML Kernels Practice

This repository contains my initial practice implementations of **GPU-accelerated kernels using CUDA with C++**, focused on building fundamental intuition for machine learning and high-performance computing.

---

## 🗂️ Current Contents

### `intro.cu`

- **Matrix multiplication kernel** (basic matmul using CUDA threads and blocks)

### `kernels/vector_add.cu`

- **Vector addition kernel** – elementwise addition of two vectors in parallel

### `kernels/dot_product.cu`

- **Dot product kernel** – parallel reduction to compute dot product efficiently

### `kernels/relu.cu`

- **ReLU activation kernel** – applies ReLU activation function elementwise

---

## 🚀 Setup & Running

1. **Compile with nvcc:**

   ```bash
   nvcc <filename>.cu -o <output>.exe
   ```

2. **Run:**
```bash
.\<output>.exe
```

## 🎯 Learning Goals:
- Understand CUDA thread indexing and memory management
- Implement fundamental ML computation kernels
- Prepare for building full neural network forward passes

## 💡 Next Steps
- Implement softmax activation
- Build a full MLP forward pass with GPU-only bias addition
- Benchmark CPU vs GPU performance

## 📜 License 
MIT License