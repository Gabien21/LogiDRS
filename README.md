# 📘 LogiDRS: Logical Reasoning using Discourse Relation Sense

**LogiDRS** is a model designed to tackle **logical reasoning** in multiple-choice **Machine Reading Comprehension (MRC)** tasks. Logical reasoning is crucial for recognizing hidden connectives and semantic relationships in text, a capability that many previous models still struggle with.

This work introduces a novel approach that integrates **discourse relation sense** between textual units to improve logical structure comprehension and final predictions.

---

## 🧾 Abstract

Logical Reasoning plays an important role in machine reading comprehension (MRC) tasks by enabling the recognition of hidden connectives and relations in text. However, previous studies still struggle to recognize logical structures in text and do not fully utilize the relationship between textual units in these structures. 

In this paper, we propose **LogiDRS**, a novel approach that leverages the sense of relationship between textual units to predict an answer to tackle this challenge. 

We evaluate **LogiDRS** on two logical reasoning benchmarks: **ReClor** and **LogiQA**, and show strong performance against competitive baselines.

---

## 📊 Experiment Results

**LogiDRS** achieves competitive results among RoBERTa-Large single model methods and the **DAGN** model.

---

## ⚙️ How to Run

###  Obtain Datasets:
Download the benchmark datasets:
- [ReClor dataset](https://github.com/yuweihao/reclor)
- [LogiQA dataset](https://github.com/lgw863/LogiQA-dataset)

### Run the Model:
To run the model, use the command below:
```bash
sh run_logidrs.sh
```
---
## 🙏 Acknowledgement
The implementation of LogiDRS is inspired by [DAGN](https://github.com/Eleanor-H/DAGN) and supported by [Huggingface Toolkit](https://huggingface.co/docs/transformers). This research is funded by University of Science, VNU-HCM under grant number CNTT 2024-17. This research used the GPUs provided by the Intelligent Systems Lab at the Faculty of Information Technology, University of Science,VNU-HCM. Moreover,the datasets [PDTB 2.0](https://catalog.ldc.upenn.edu/LDC2008T05) used in this study were generously supported by the Linguistic Data Consortium (LDC).