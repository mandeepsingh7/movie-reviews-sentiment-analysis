# Sentiment Analysis on Movie Reviews

## Overview

In this project, we perform sentiment analysis on the IMDB movie review dataset. The dataset comprises 50,000 reviews, evenly distributed between positive and negative sentiments, making it an ideal resource for sentiment analysis tasks.

### Dataset

- **Source**: [IMDB movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)
- **Description**: The dataset comprises 50,000 reviews, equally divided into positive and negative sentiments.

### Data Preprocessing

- **Cleaning**: Removal of HTML tags, URLs, and special characters to ensure only textual data remains.
- **Tokenization**: Breaking down the text into individual words or tokens to facilitate further processing.
- **Stop Words Removal**: Elimination of common words (stop words) that do not add significant meaning to the text.

### Machine Learning Models

We implemented several traditional machine learning classification models to classify movie reviews into positive or negative sentiments. These models include:

- **SVM (Support Vector Machine)**: Achieved an accuracy of 90.8%.
- **Logistic Regression**: Achieved an accuracy of 90.2%.
- **Naive Bayes**: Achieved an accuracy of 88.2%.
- **Random Forest**: Achieved an accuracy of 82.6%.
- **K-Nearest Neighbors (KNN)**: Achieved an accuracy of 79.8%.

### Deep Learning Models

Additionally, Long Short-Term Memory (LSTM) neural networks were employed for sentiment analysis. Two distinct approaches were compared:

- **LSTM without GloVe Embeddings**: In this approach, a vocabulary was constructed from the dataset, and each review was numerically encoded to feed into the LSTM model. Achieved an accuracy of 88.5% 
- **LSTM with GloVe Embeddings**: Pre-trained GloVe word embeddings were utilized to initialize the embedding layer of the LSTM model, enhancing its ability to capture semantic information. Achieved an accuracy of 87.6%.
- **DistilBERT**: Fine-tuned for 5 epochs, yielding an accuracy of 92.1%.

### Results

The performance of each model was evaluated using accuracy metrics. The following notable results were achieved:

- **SVM**: Achieved an accuracy of 90.8%.
- **DistilBERT**: got an accuracy of 92.1%.



### Project Files

- `data_extraction.ipynb`: Responsible for data preprocessing and saving the preprocessed data into a CSV file.
- `ml_training.ipynb`: Conducts the training of ML models (SVM, Logistic Regression, Naive Bayes, Random Forest, KNN) using the preprocessed CSV file.
- `lstm_training.ipynb`: Handles the training of LSTM model without GloVe embeddings, involving the creation of a vocabulary and data processing.
- `lstm_training_glove.ipynb`: Implements LSTM training with GloVe embeddings.
- `distilbert.ipynb`: Performs fine-tuning of DistilBERT for sentiment analysis.
- `kaggle_distilbert.ipynb`, `kaggle_glove.ipynb`, `kaggle_lstm.ipynb`: Utilizes Kaggle GPU resources for training and displays the training process.
