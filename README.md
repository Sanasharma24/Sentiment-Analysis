# 🚦 Sentiment Analysis: Comparing VADER and RoBERTa

Welcome! This project explores and compares two powerful approaches to sentiment analysis:  
- **VADER** (a fast, rule-based model ideal for social media and short texts)  
- **RoBERTa** (a state-of-the-art deep learning transformer model for nuanced understanding)

My goal is to **analyze the strengths, weaknesses, and practical differences** between these models when classifying text as **Positive**, **Neutral**, or **Negative**.

---

## 🎯 Main Goal

To **compare the performance, accuracy, and behavior of VADER and RoBERTa** on the same sentiment analysis tasks, and to provide insights into which model is better suited for different scenarios.

---

## 🧭 How Does It Work?

1. **Dataset Selection:**  
   We use the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), which contains thousands of real-world text reviews and their star ratings.

2. **Preprocessing:**  
   - Clean and prepare the text data for analysis.

3. **Sentiment Analysis:**  
   - **VADER:** Uses NLTK’s VADER lexicon for fast, rule-based sentiment scoring.
   - **RoBERTa:** Uses Hugging Face’s `cardiffnlp/twitter-roberta-base-sentiment` for deep learning-based sentiment classification.

4. **Comparison & Evaluation:**  
   - Run both models on the same reviews.
   - Compare their predictions, probability scores, and overall accuracy.
   - Visualize results with charts and tables.

5. **Interpretation:**  
   - Discuss where each model excels or struggles, and why.

---

## ✨ Features

- **Side-by-side sentiment predictions** from VADER and RoBERTa.
- **Probability/confidence scores** for each sentiment class.
- **Clear visualizations** to highlight differences in model behavior.
- **Well-commented, beginner-friendly code** in a Jupyter Notebook.

---

## 🛠️ Technologies Used

| Technology             | Purpose                                  |
|-----------------------|------------------------------------------|
| Python                | Programming language                      |
| NLTK (VADER)          | Rule-based sentiment analysis             |
| Hugging Face Transformers (RoBERTa) | Deep learning sentiment analysis |
| PyTorch               | Model execution for RoBERTa               |
| Jupyter Notebook      | Interactive coding and visualization      |
| Matplotlib/Seaborn    | Data visualization                        |

---

## 🚀 How to Run

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Sanasharma24/Sentiment-Analysis.git
    cd Sentiment-Analysis
    ```

2. **Install dependencies:**
    ```bash
    pip install pandas nltk torch transformers matplotlib scikit-learn jupyter
    ```

3. **Download NLTK VADER lexicon (first time only):**
    ```python
    import nltk
    nltk.download('vader_lexicon')
    ```

4. **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5. **Open `Sentiment_Analysis.ipynb` and run the cells to see the comparison in action!**

---

## 📝 Example Output

| Review Text                        | VADER Prediction | RoBERTa Prediction |  
|------------------------------------|------------------|--------------------|  
| "I love this product!"             | Positive         | Positive           |  
| "It was okay, nothing special."    | Neutral          | Neutral            |  
| "Terrible experience, never again" | Negative         | Negative           |  

You’ll see detailed charts and tables in the notebook!

---

## 🔍 What Will You Learn?

- The practical differences between rule-based and deep learning sentiment models.
- When to use a fast, interpretable model (VADER) vs. a powerful, nuanced model (RoBERTa).
- How to evaluate and visualize model performance on real-world data.

---

## 📬 Contact

Questions or feedback?  
📧 Email: sanasharma0424@gmail.com

---

**This project demonstrates my ability to implement, compare, and interpret modern NLP models. I’m excited to apply these skills in real-world data science and AI challenges!**

---

Let me know if you want to add screenshots, sample charts, or further customize this README!

---
Answer from Perplexity: pplx.ai/share
