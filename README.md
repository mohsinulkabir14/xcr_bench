# 🧩 XCR-Bench: Cross-Cultural Reasoning Benchmark

XCR-Bench is a human-annotated, multi-task benchmark for evaluating the cross-cultural reasoning capabilities of large language models (LLMs). It provides high-quality parallel data annotated with Culture-Specific Items (CSIs) and mapped to Hall’s Triad of Culture, enabling systematic evaluation of cultural competence beyond surface-level artifacts.

This repository contains the dataset and evaluation scripts introduced in the paper:

**XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs**  
Mohsinul Kabir, Tasnim Ahmed, Md Mezbaur Rahman, Shaoxiong Ji, Hassan Alhuzali, Sophia Ananiadou


---

## 🌍 Overview

Cross-cultural competence in LLMs requires the ability to:

- Identify culture-specific elements in text  
- Predict appropriate cultural references  
- Adapt them across different cultural contexts  

Existing evaluations mostly rely on machine translation or intrinsic knowledge probing. XCR-Bench goes beyond this by framing cultural competence as a reasoning problem grounded in realistic scenarios.

### What XCR-Bench Provides

- 4,900+ parallel sentences  
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

## 📌 Tasks Supported

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

## 🗂 Dataset Structure

A typical release of XCR-Bench is organized as follows:

