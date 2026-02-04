# AUSteer

**Fine-Grained Activation Steering: Steering Less, Achieving More**  
**ICLR 2026** · [Paper](https://openreview.net/forum?id=guSVafqhrB)

Breaking LLM blocks into **fine-grained atomic units (AUs)** for targeted intervention—**steer less, achieve more**.

---

## Table of Contents

- [Preparation](#preparation)
- [Data](#data)
- [Quick Start](#quick-start)
- [Reproducing Results](#reproducing-results)
- [Key Modules](#key-modules)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Preparation

### Environment

- Python **3.10**
- Dependencies: `requirements.txt` and `requirements2.txt`

Create and activate a conda environment:

```bash
conda create -n austeer python=3.10
conda activate austeer
pip install -r requirements.txt
pip install -r requirements2.txt
```

---

## Data
[Download Dataset Here](https://drive.google.com/file/d/1BV3eUSoluUiCtWcQ5M52dM1HfBNe5w-D/view?usp=sharing)
[Download AU Ranks Here (Optional)](https://drive.google.com/file/d/13EpYY25lMj1nqmDQ_I27TvJ73talbs48/view?usp=sharing)
- Datasets are located in the `datasets/` directory.
- AU ranks (per model and task) are provided in the `AU_ranks/` directory.

---

## Quick Start


```bash
python main_test.py --model meta-llama/Llama-2-7b-chat-hf --data_name boolq --k 100 --alpha 15 --applied_module attention
```

---

## Reproducing Results

Run the following command to reproduce the experimental results:

```bash
nohup bash reproduce.sh > reproduce_result.log 2>&1 &
```

Logs will be written to `reproduce_result.log`.

---



## Key Modules

- **Atomic Unit Localization**  
  `MFU_screening` is used to rank atomic units (AUs) for each model and task.

- **Adaptive Steering**  
  `set_MFU` is used to adaptively steer the selected AUs.

- **Model Modification**  
  Model implementations are modified under the `rmodels/` directory.  
  To apply **AUSteer** to other models, modify the corresponding modeling code in a similar manner.

---

## Citation

For questions, please contact **feng0119 AT e.ntu.edu.sg**.

If you find this work useful, please cite:

```bibtex
@inproceedings{austeer2026,
  title     = {Fine-Grained Activation Steering: Steering Less, Achieving More},
  author    = {Zijian, Feng and Tianjiao, Li and Zixiao, Zhu and Hanzhang, Zhou and Junlang, Qian and Li, Zhang and Jia Jim Deryl, Chua and Lee Onn, Mak and Gee Wah, Ng and Kezhi, Mao},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  note      = {ICLR 2026}
}
```

---

## Acknowledgements

```bibtex
@inproceedings{wangsemantics,
  title     = {Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors},
  author    = {Wang, Weixuan and Yang, Jingyuan and Peng, Wei},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```
