# 📌 Consumer Complaint Classification with Naive Bayes

This project focuses on building a **multi-class text classification model** using **Naive Bayes** to automatically categorize consumer complaints into predefined product-related classes. It utilizes a large-scale dataset of complaints submitted to the Consumer Financial Protection Bureau (CFPB).


## 📊 Dataset Overview

The dataset includes over **2.3 million records**, of which **~800,000** records with valid complaint narratives are used for analysis.

Each record consists of:
- **Product** (Target class): e.g., `credit_report`, `loan`, `card`, `mortgage`, etc.
- **Consumer Complaint Narrative** (Text): Descriptions of issues faced by consumers.



## 🧹 Data Preprocessing

Steps performed:
- Selected only `Product` and `Consumer complaint narrative` columns
- Dropped rows with missing narratives
- Mapped similar product names to unified categories
- Tokenized the text using `NLTK`
- Removed:
  - Stop words
  - Punctuation
  - Placeholder tokens like `xxxx`, `000`
- Created a new cleaned column `clean_complaints`



## 📈 Exploratory Data Analysis

- Visualized class distribution using seaborn
- Observed `credit_report`, `debt_collection`, and `mortgage` to be the most frequent categories
- Word frequency analysis was done for different complaint types like `loan`, `card`



## 🧠 Model Building

- Used `CountVectorizer` to transform text data into numerical features
  - Applied `min_df=200` to limit vocabulary size
- Split the data into train-test sets (80-20 split, stratified)
- Trained a **Multinomial Naive Bayes** classifier



## 🧪 Evaluation

- Achieved **~78.7% accuracy** on the test set
- Visualized results using a **confusion matrix heatmap**
- Observed misclassifications among semantically similar classes


## 🔍 Prediction Example

Sample inputs were passed through the preprocessing and model pipeline:

```python
test_complaint = [
  "I have a debt of Bank of America which was written off...",
  "Delta American Express charge card. I suddenly lost privileges..."
]
````

**Predicted Classes:**

* `debt_collection`
* `card`

## 💾 Model Deployment

* The trained **vectorizer** and **Naive Bayes model** are saved as `.pkl` files for future use:

  * `Output/count_vect.pkl`
  * `Output/nb.pkl`

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/NishatTasnim01/App-Reviews-Sentiment-Analysis.git
   cd App-Reviews-Sentiment-Analysis
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:

   ```bash
   jupyter notebook App_Reviews_Sentiment_Analysis.ipynb
   ```


## 📁 Project Structure

```
├── App_Reviews_Sentiment_Analysis.ipynb
├── Output/
│   ├── count_vect.pkl
│   └── nb.pkl
├── complaints.csv
├── requirements.txt
└── README.md
```

## 👩‍💻 Developed By

🔗 [GitHub: Nishat Tasnim](https://github.com/NishatTasnim01)

## 📌 Acknowledgment

* Data sourced from the [Consumer Financial Protection Bureau (CFPB)](https://www.consumerfinance.gov/data-research/consumer-complaints/).


⭐ *If you found this project helpful, please consider giving it a star!*
