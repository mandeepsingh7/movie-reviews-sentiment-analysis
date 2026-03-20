# Movie Reviews Sentiment Analysis
Sentiment classification of movie reviews using both Classical Machine Learning (TF-IDF + ML models) and Transformer-based models (DistilBERT) with a modular pipeline for training, evaluation, and inference.


## Overview

This project builds a sentiment analysis system to classify movie reviews as Positive or Negative.

The pipeline includes:
- Data loading and preprocessing from IMDB dataset
- Training multiple ML models using TF-IDF features
- Training a transformer model using DistilBERT
- Evaluation with test accuracy

## Dataset
- **Source** : IMDB Dataset (via Hugging Face)
- **Total samples:** 50,000 
- **Train/Valid/Test split:** 20,000 / 5,000 / 25,000
- **Classes** : 2 (Positive, Negative)

## Models

### Classical Machine Learning

- TF-IDF vectorization
- Models:
    - Logistic Regression
    - Linear SVM
    - Naive Bayes

### Transformer Model

- Model: DistilBERT (distilbert-base-uncased)
- Fine-tuned on IMDB dataset using Hugging Face Trainer API

## Results

### Transformer (DistilBERT)
- Accuracy: 90.86%

### ML Models
- Logistic Regression: 88.03%
- Linear SVM: 87.41%
- Naive Bayes: 85.20%

Transformer models achieve higher accuracy, while ML models provide faster inference.

## Project Structure

```text
movie-reviews-sentiment-analysis/
|
├── models/
│   ├── ml/
│   │   |── linear_svc.joblib
│   │   |── multinomial_nb.joblib
│   │   └── logistic_regression.joblib
│   │
│   └── transformers/
│       └── <run_name>/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── training_log.json
│           ├── evaluation_results.json
│           └── evaluation/
│               ├── test_metrics.json
│               └── training_curves.png
|
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── ml_utils.py
│   ├── model_utils.py
│   ├── train_utils.py
│   ├── eval_utils.py
│   └── scripts/
│       ├── download_test_data.py
│       ├── evaluate.py
│       └── train.py
|
└── notebooks/
    |── ml_training.ipynb
    |── ml_eval.ipynb
    └── inference.ipynb

```

## Tech Stack

### Machine Learning & Deep Learning
-   PyTorch
-   Scikit-learn
-   Hugging Face Transformers 

### Backend
-   FastAPI
-   Docker 

### Infrastructure
-   AWS EC2
-   Vercel

## Setup

### Install dependencies
```sh
pip install -r requirements.txt
```

### Download test data 
```sh
python src/scripts/download_test_data.py 
```

## Usage 

### 1. Train Transformer Model
```sh
python src/scripts/train.py --run_name v1 --epochs 5
```
Arguments ( `data-fraction`, `lr`, `weight_decay` ) can be configured via CLI, with defaults already provided.

### 2. Evaluate Transformer Model 
```sh
python src/scripts/evaluate.py --model_dir models/transformers/v1
```

Outputs:
- Accuracy 
- Training curves

### 3. ML Training

Train and experiment with ML models
```sh
notebooks/ml_training.ipynb
```
Includes:
- TF-IDF Feature extraction
- Training multiple models 
- Hyperparameter tuning 

### 4. ML Evaluation 
Evaluate trained ML models
```sh
notebooks/ml_eval.ipynb
```
Includes:
- Test set evaluation

### 5. Inference
Run predictions using both ML and Transformer models
```sh
notebooks/inference.ipynb
```

This repository focused on model development and inference pipelines. <br>
Backend API and frontend are maintained separately.

### Backend ─ Projects Hub

https://github.com/mandeepsingh7/projects-hub

- Centralized backend for all ML/AI projects 
- Contains FastAPI endpoints for multiple projects, including the Movie Reviews Sentiment Analysis system
- Handles :
  - Inference pipelines
  - Routing 

**Deployment:** AWS EC2

### Frontend ─ Portfolio

https://github.com/mandeepsingh7/portfolio

- Unified frontend for all projects
- Provides UI to interact with different APIs from Projects Hub 
- Handles :
  - User Interface 
  - API Integration 

**Deployment:** Vercel 

## API Example

### Get Random Sample

**Endpoint**

```
GET /movie-reviews-sentiment-analysis/random-sample
```

**Response**
```json
{
  "text": "Technically speaking, this movie sucks...lol. However, it's also hilarious. Whether or not it's intentionally funny I don't know. Horrible in every aspect, it also is the only movie I know of that has 1) a fat kid being played by a slim actor in a (very obvious) fat suit, 2) an attractive 30-something actress playing a character who's supposed to be in her late 60's, and 3) the most compliments for plastic yard daisies ever. Don't take this film seriously, just watch it for laughs....a great party movie.",
  "label": "Negative"
}
```

### Predict Sentiment

**Endpoint**

```
POST /movie-reviews-sentiment-analysis/predict/
```
**Request**
```json
{
  "text": "Technically speaking, this movie sucks...lol. However, it's also hilarious. Whether or not it's intentionally funny I don't know. Horrible in every aspect, it also is the only movie I know of that has 1) a fat kid being played by a slim actor in a (very obvious) fat suit, 2) an attractive 30-something actress playing a character who's supposed to be in her late 60's, and 3) the most compliments for plastic yard daisies ever. Don't take this film seriously, just watch it for laughs....a great party movie."
}
```

**Response**
```json
{
  "prediction": "Negative"
}
```

## Demo

Accessible through the portfolio:

https://mandeeps.in