# <img width="70" height="70" alt="image" src="https://github.com/user-attachments/assets/662c5483-bb81-43db-96f8-6cb3b1c9eb5b" /> XCR-Bench: Cross-Cultural Reasoning Benchmark

XCR-Bench is a human-annotated, multi-task benchmark for evaluating the cross-cultural reasoning capabilities of large language models (LLMs). It provides high-quality parallel data annotated with Culture-Specific Items (CSIs) and mapped to Hall’s Triad of Culture, enabling systematic evaluation of cultural competence beyond surface-level artifacts.

This repository contains the dataset and evaluation scripts introduced in the paper:

**[XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs](https://arxiv.org/abs/2601.14063)**


---

## 🌍 Overview

Cross-cultural competence in LLMs requires the ability to:

- Identify culture-specific elements in text  
- Predict appropriate cultural references  
- Adapt them across different cultural contexts  

Existing evaluations mostly rely on machine translation or intrinsic knowledge probing. XCR-Bench goes beyond this by framing cultural competence as a reasoning problem grounded in realistic scenarios.

### What XCR-Bench Provides

- 4,100+ parallel sentences  
- 1,098 unique Culture-Specific Items (CSIs)  
- Annotations across:  
  - CSI categories (based on Newmark’s framework)  
  - Hall’s Triad of Culture  
  - Intra-lingual and inter-lingual adaptations  
- Data for four target cultures:  
  - Chinese  
  - Arabic  
  - Bengali (West Bengal)  
  - Bengali (Bangladesh)

---

## 📌 Designed Tasks

XCR-Bench enables three core evaluation tasks.

### 1. CSI Identification  
Detect culture-specific items in Western (US/UK) sentences.

- **Input:** Plain sentence  
- **Output:** Identified CSI span(s)

### 2. CSI Prediction  
Predict appropriate Western CSIs given masked contexts.

- **Input:** Sentence with `<CSI>[MASK]</CSI>`  
- **Output:** Predicted CSI

### 3. CSI Adaptation  
Adapt CSIs from Western culture to a target culture.

Settings:
- Intra-lingual (English → English cultural adaptation)  
- Inter-lingual (English → target language)

Output:
- Adapted sentence  
- Adaptation strategy (based on Newmark’s taxonomy)

---

## 🗂 Repository Structure

This repository organized as follows:

```
xcr_bench/
├── data/
│   ├── xcr_bench_base_corpus.csv
│   ├── xcr_bench_chinese_adaptation.csv
│   ├── xcr_bench_arabic_adaptation.csv
│   ├── xcr_bench_bengali_bangladesh_adaptation.csv
│   └── xcr_bench_bengali_west_bengal_adaptation.csv
│
├── code/
│   ├── Identification/
│   │   ├── identification_prompt.txt
│   │   └── Evaluation/
│   │
│   ├── Prediction/
│   │   ├── prediction_prompt.txt
│   │   └── Evaluation/
│   │
│   └── Adaptation/
│       ├── adaptation_prompt.txt
│       └── Evaluation/
│
├── README.md
└── LICENSE
```

Each data instance contains:

- Original sentence  
- Cultural Context
- CSI category
- CSI Hall Mapping
- Hall cultural level (Visible / Semi-visible / Invisible)  
- Adapted equivalents for each culture  

---

## 📊 Evaluation Metrics

XCR-Bench includes both hard and soft evaluation metrics.

| Task | Hard Metric | Soft Metric |
|-----|-------------|-------------|
| CSI Identification | Exact span match | Levenshtein-based similarity |
| CSI Prediction | Exact match | Sentence-BERT semantic similarity |
| CSI Adaptation | – | CSI-BERT and SENT-BERT scores |

Evaluation scripts implementing these metrics are provided in the `Evaluation/` directory.

## 📜 License

This dataset is released under the **CC BY-NC-ND: Creative Commons Attribution-NonCommercial-NoDerivatives** license.  
Please ensure appropriate attribution when using the data.

---

## 🤗 Hugging Face Dataset

The XCR-Bench corpus is publicly available on Hugging Face:

https://huggingface.co/datasets/mohsinulkabir14/xcr_bench

## 📎 Citation

If you use XCR-Bench in your work, please cite:
```
@article{kabir2026xcr,
  title={XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs},
  author={Kabir, Mohsinul and Ahmed, Tasnim and Rahman, Md Mezbaur and Ji, Shaoxiong and Alhuzali, Hassan and Ananiadou, Sophia},
  journal={arXiv preprint arXiv:2601.14063},
  year={2026}
}
```



## 🤝 Contact

For questions, suggestions, or collaborations:

- **Mohsinul Kabir** – mdmohsinul.kabir@manchester.ac.uk  
- **Sophia Ananiadou** – sophia.ananiadou@manchester.ac.uk  
