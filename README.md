# CUDA ML Kernels

## 🚀 Motivation

This repository implements **custom CUDA kernels for common ML operations**, benchmarked against PyTorch's highly optimized cuBLAS/cuDNN kernels. The goal is to:

- Understand GPU parallelization patterns
- Compare naive kernel performance vs. library implementations
- Build intuition for ML Systems performance engineering

---

## 📁 Repository Structure

```.
├── kernels/
│   ├── matrix\_multiply.cu
│   ├── vector\_add.cu
│   ├── relu.cu
│   ├── dot\_product.cu
│   ├── intro.cu
├── benchmarks/
│   └── benchmark.py
└── .gitignore
```

- `kernels/`: CUDA C++ kernel implementations
- `benchmarks/`: Python script to benchmark kernels vs. PyTorch

---

## ⚡ Kernels Implemented

| Kernel            | Description                       |
|-------------------|-----------------------------------|
| `matrix_multiply` | Matrix multiplication (1024x1024) |
| `vector_add`      | Elementwise vector addition       |
| `relu`            | ReLU activation function          |
| `dot_product`     | Vector dot product reduction      |
| `intro`           | 4x4 matrix multiplication demo    |

---

## 🔧 Build Instructions

1. **Ensure NVIDIA CUDA Toolkit is installed.**

2. **Compile each `.cu` file:**

```bash
cd kernels

nvcc -o matrix_multiply.exe matrix_multiply.cu
nvcc -o vector_add.exe vector_add.cu
nvcc -o relu.exe relu.cu
nvcc -o dot_product.exe dot_product.cu
nvcc -o intro.exe intro.cu
````

> Replace `.exe` with no extension if on Linux/Mac.

---

## 🧪 Running Benchmarks

From the repo root:

```bash
cd benchmarks
python benchmark.py
```

---

## 📊 Results Summary

| Kernel             | PyTorch Time (ms) | Custom CUDA Time (ms) | Speedup |
| ------------------ | ----------------- | --------------------- | ------- |
| matrix\_multiply   | 14.28             | 6.95                  | 2.06x   |
| vector\_add        | 2.39              | 1.35                  | 1.77x   |
| relu               | 2.97              | 0.56                  | 5.35x   |
| dot\_product       | \~0               | 1.33                  | Slower  |
| intro (4x4 matmul) | 2.25              | 1.08                  | 2.09x   |

---

## 💡 Key Insights

* **Matrix multiplication and ReLU** kernels show significant speedups, demonstrating effective GPU thread parallelization.
* **Vector addition** gains are modest, as PyTorch uses cuBLAS kernels optimized near theoretical peak.
* **Dot product** is slower due to naive reduction implementation vs. PyTorch's warp-level optimized reductions.
* **Small matmul (intro)** demonstrates kernel launch overhead optimization benefits.

---

## 📝 Future Improvements

* Implement **warp-level reductions** for dot product
* Integrate **unit tests** comparing kernel outputs with PyTorch for correctness validation
* Extend to **batched kernels** relevant for end-to-end ML pipeline acceleration

---

## 👤 Author

Gauri Sharan

---

## 📜 License

MIT License