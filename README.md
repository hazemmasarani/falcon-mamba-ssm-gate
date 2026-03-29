# 🚀 Falcon-Mamba SSM–Gate Parallelism

A **model-parallel implementation** of Falcon-Mamba where the architecture is split into:

- **SSM Model**
- **Gate Model**

Each runs on a **separate GPU** using **PyTorch Distributed (NCCL)**.

---

## 📁 Structure
```
├── model/
│ ├── falcon_gate_modeling.py
│ ├── falcon_ssm_modeling.py
├── output/
│ └── logits.pt
├── run_test.py
└── README.md
```


---

## ⚙️ Requirements

- Python 3.10+
- PyTorch (CUDA)
- transformers
- 2 GPUs

```bash
pip install torch transformers
```

---

▶️ Run
```
python run_test.py \
  -dev0 cuda:0 \
  -dev1 cuda:1 \
  -batch_size 2 \
  -seq_len 32
  -num_iter 10
```

---

## 🔄 How It Works

- **Rank 0 (GPU 0):** Loads the Gate model  
- **Rank 1 (GPU 1):** Loads the SSM model  
- Uses `torch.distributed` with NCCL backend  
- Executes forward pass across GPUs  
- Saves output to `output/logits.pt`  

---
## 📌 Notes

- Designed for **multi-GPU experimentation**  
- Prototype for **architecture-level parallelism**  
- Requires **CUDA + NCCL** 