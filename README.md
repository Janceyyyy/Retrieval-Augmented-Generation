
# Retrieval-Augmented Generation (RAG)
![alt text](pic/flowchart.png)
This repository contains Python scripts developed as part of our research project focused on enhancing machine learning models, specifically T5 and GPT. As this project is part of ongoing research, specific application codes on GPT & T5, along with detailed results, are not included in this repository.

## Overview

Our work employs Retrieval-Augmented Generation (RAG) and advanced embedding techniques to improve the efficiency and accuracy of the T5 and GPT models. The core components of our project include:

1. **RAG.py** - Implements the Retrieval-Augmented Generation process by leveraging `RoBERTa` embeddings to find and utilize similar historical data points that enhance the training data for our models.

2. **Embedding_Generation.py** - Generates embeddings for the entire training dataset using the `RoBERTa` model. This script is fundamental to the retrieval process in `RAG.py`, enabling the matching of similar data points based on cosine similarity.

## Technologies Used

- **Transformers Library**: For accessing pretrained models like RoBERTa for embeddings and tokenizer functionalities.
- **PyTorch**: Serving as the backbone for model interactions and tensor operations.
- **Scikit-Learn**: Specifically, the cosine similarity measure from `sklearn.metrics.pairwise` to identify similar data points.
- **NumPy**: For efficient array and mathematical operations essential in handling embeddings.

## Requirements

- Python 3.8 or higher
- torch
- transformers
- numpy
- sklearn

## Dataset

The project uses data from the "2014 i2b2/UTHealth shared task", focusing on clinical patient data for enhancing model predictions. For more details on the dataset and its acquisition, refer to the Department of Biomedical Informatics at Harvard Medical School.

## Note

As this project is part of ongoing research, specific application codes on GPT & T5, along with detailed results, are not included in this repository. These will be discussed in our forthcoming paper.

