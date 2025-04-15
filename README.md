# LogiDRS: Logical Reasoning using Discourse Relation Sense

**LogiDRS** is a model designed to tackle **logical reasoning** in multiple-choice **Machine Reading Comprehension (MRC)** tasks. Logical reasoning is crucial for recognizing hidden connectives and semantic relationships in text, a capability that many previous models still struggle with.

This work introduces a novel approach that integrates **discourse relation sense** between textual units to improve logical structure comprehension and final predictions.

---

## 🧠 Abstract

Logical Reasoning plays an important role in machine reading comprehension (MRC) tasks by enabling the recognition of hidden connectives and relations in text. However, previous studies still struggle to recognize logical structures in text and do not fully utilize the relationship between textual units in these structures. 

In this paper, we propose **LogiDRS**, a novel approach that leverages the sense of relationship between textual units to predict an answer to tackle this challenge. 

We evaluate **LogiDRS** on two logical reasoning benchmarks: **ReClor** and **LogiQA**, and show strong performance against competitive baselines.

---

## 🧪 Experiment Results

**LogiDRS** achieves competitive results among RoBERTa-Large single model methods and **DAGN** model.

---

## 🔧 How to Run

To run the model, use this command:

```bash
sh run_logiformer.sh
```
---
## 🙏 Acknowledgement
The implementation of LogiDRS is inspired by [DAGN]() and [LReasoner](), and supported by [Huggingface Toolkit]().