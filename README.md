# Sentiment-Analysis
This project demonstrates the implementation of sentiment analysis using the RoBERTa model, a state-of-the-art NLP model from Hugging Face's Transformers library. The model is fine-tuned on a Twitter sentiment dataset and is capable of classifying text into positive, neutral, or negative sentiment categories.
# Sentiment Analysis with RoBERTa

## Description

This project demonstrates the implementation of sentiment analysis using the RoBERTa model, a state-of-the-art NLP model from Hugging Face's Transformers library. The model is fine-tuned on a Twitter sentiment dataset and is capable of classifying text into positive, neutral, or negative sentiment categories.

## Objectives

1. **Understand and implement sentiment analysis** using the RoBERTa model.
2. **Learn to work with Hugging Face Transformers** and PyTorch to perform NLP tasks.
3. **Develop skills in data preprocessing and model evaluation** for sentiment analysis.

## Features

- **Data Preprocessing**: Tokenizes input text using the `AutoTokenizer` from Hugging Face.
- **Model Inference**: Utilizes `AutoModelForSequenceClassification` for sentiment classification.
- **Softmax Scoring**: Applies the softmax function to model outputs to obtain probability scores for sentiment classes.
- **User-friendly**: Easy-to-understand code and well-commented for educational purposes.

## Technologies Used

- **Python**: The programming language used for the entire project.
- **Hugging Face Transformers**: For model and tokenizer loading.
- **PyTorch**: For running the RoBERTa model.
- **Jupyter Notebook**: For interactive development and demonstration of the project.

## Code Example

```python
# ROBERTA PRETRAINED MODEL
!pip install torch torchvision torchaudio
!pip install transformers

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get model predictions
    outputs = model(**inputs)
    scores = outputs[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Define labels
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Get the sentiment with the highest score
    sentiment = labels[scores.argmax()]
    return sentiment, scores

# Example usage
text = "I love using the RoBERTa model for NLP tasks!"
sentiment, scores = analyze_sentiment(text)
print(f"Sentiment: {sentiment}")
print(f"Scores: {scores}")
```

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Sanasharma24/sentiment-analysis.git
    cd sentiment-analysis
    ```

2. **Install dependencies**:
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers
    ```

3. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

4. **Execute the notebook cells** to see the sentiment analysis in action.

## Contact

For any queries or suggestions, feel free to contact me at sanasharma0424@gmail.com.

---

This project showcases my ability to implement advanced NLP techniques using modern libraries and tools. I am eager to apply these skills in a challenging internship where I can contribute and grow further.
