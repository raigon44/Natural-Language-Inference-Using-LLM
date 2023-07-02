# Natural-Language-Inference-Using-LLM

This repository contains the source code for fine-tuning BERT for the natural language inference task.

Natural Language Inference (NLI) is a fundamental task in natural language processing (NLP) that involves determining the logical relationship between two given sentences. The goal is to determine whether the relationship is "entailment," "contradiction," or "neutral."

The Stanford Natural Language Inference (SNLI) dataset is a widely used benchmark for NLI tasks. It consists of a large collection of sentence pairs, each labeled with one of the three aforementioned relationships. The dataset is manually annotated, making it a reliable resource for evaluating NLI models. In the SNLI dataset, each sentence pair consists of a premise and a hypothesis. The premise is a statement or a sentence that serves as the context, while the hypothesis is another sentence that needs to be evaluated against the premise. The task is to determine whether the hypothesis can be inferred from the premise (entailment), contradicts the premise (contradiction), or has no relationship with the premise (neutral).

For example, consider the following sentence pair from the SNLI dataset:

Premise: "A young boy playing soccer in the park."
Hypothesis: "A child is outside playing a sport."

In this case, the relationship between the premise and hypothesis is entailment, as the hypothesis can be inferred from the premise. NLI models aim to learn patterns and linguistic cues to accurately classify such relationships.

In this project, BERT was fine-tuned on the SNLI dataset.

Language Model used: BERT-BASE-UNCASED

Modeling the task for training:

**[CLS] Premise [SEP] Hypothesis [SEP]**

A classification head was added on top of the BERT model to make the prediction.


