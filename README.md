# Extractive Question Answering with SQuAD v1.1

This repository contains implementations and evaluations of various neural network models for extractive question answering (QA) tasks using the Stanford Question Answering Dataset (SQuAD v1.1).

## Project Overview

We explored several neural network architectures to tackle extractive QA, where models must identify answer spans directly from given context passages. Models evaluated include:

* Recurrent Neural Networks (RNNs): Plain BiLSTM and Bi-directional Attention Flow (BiDAF)-style LSTM
* Transformer-based models: BERT and DistilBERT
* Encoder-decoder models: T5
* Generative Language Models: GPT-2, FLAN-T5, DeepSeek

## Dataset

* SQuAD v1.1: Over 100,000 questions from Wikipedia articles, with annotated answer spans.

## Evaluation Metrics

Models were evaluated using:

* **Exact Match (EM)**: Percentage of predictions exactly matching ground truth.
* **F1 Score**: Token-level precision and recall harmonic mean.

## Contributions

### Personal Work:

I trained and evaluated three transformer-based models:

### 1. **BERT (Bidirectional Encoder Representations from Transformers)**

* **Data Pre-processing**:

  * Tokenized question-context pairs using Hugging Face's AutoTokenizer.
  * Context managed within BERT’s 512-token window with 128-token stride.
  * Handled partial answers by marking incomplete spans as "no answer".

* **Training Details**:

  * Model: BERT-base (110M parameters)
  * Epochs: 5
  * Optimizer: AdamW, learning rate of 1e-5
  * Scheduler: Linear warm-up
  * Hardware: NVIDIA RTX 4090 GPU
  * Duration: Approx. 1 hour (15 mins/epoch)

* **Performance**:

  * Exact Match (EM): **78.97%**
  * F1 Score: **86.77%**

### 2. **DistilBERT**

* **Data Pre-processing**: Similar approach to BERT.

* **Training Details**:

  * Model: distilBERT-base (67M parameters)
  * Epochs: 6
  * Optimizer: AdamW, learning rate of 1e-5
  * Scheduler: Linear warm-up
  * Hardware: NVIDIA RTX 4090 GPU
  * Duration: Approx. 46.2 minutes (9.25 mins/epoch)

* **Performance**:

  * Exact Match (EM): **75.7%**
  * F1 Score: **84.4%**

DistilBERT trained approximately 25% faster than BERT while maintaining competitive performance.

### 3. **GPT-2**

* **Data Pre-processing**:

  * Tokenized context-question pairs, managing them within the GPT-2 token limit.
  * Post-processed predictions to extract answer spans from logits, applying softmax normalization and selection criteria based on scores.

* **Training Details**:

  * Model: GPT-2
  * Epochs: 3
  * Optimizer: AdamW, learning rate of 5e-5
  * Scheduler: Linear warm-up
  * Hardware: NVIDIA RTX 4090 GPU
  * Duration: Approx. 1.5 hour (30 mins/epoch)

* **Performance**:

  * Exact Match (EM): **74.75%**
  * F1 Score: **84.30%**

## Other Models Evaluated

* **RNNs (BiLSTM and BiDAF)**: Demonstrated moderate performance; BiDAF with attention outperformed plain BiLSTM significantly.
* **T5**: Achieved robust performance with a slightly higher computational overhead.
* **Generative models (FLAN-T5, DeepSeek)**: Varied performance with FLAN-T5 performing moderately well and DeepSeek demonstrating high effectiveness.

## Comparative Analysis

* Transformer models (BERT, DistilBERT, T5) significantly outperformed RNN-based models in accuracy.
* Higher embedding sizes consistently improved performance.
* Transformer-based models showed superior context comprehension and generalization.
* RNN models provided faster training and lower computational requirements but struggled in complex context-question interactions.

## Future Improvements

* Incorporation of character-level embeddings and deep contextualized embeddings (e.g., ELMo).
* Exploration of larger model variants to further boost accuracy.

## Conclusion

Transformer models, particularly BERT, provided the best balance between performance and computational demands, making them ideal for real-world extractive QA tasks.

## Team Members

Jake Cox, Nolan DeBord, Alan Huang, Nina Luong, **Arya Rahmanian**, Chenjie Yu
