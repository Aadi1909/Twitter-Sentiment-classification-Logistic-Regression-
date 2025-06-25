# Twitter Sentiment Classification with Logistic Regression

This repository contains a Python project to perform sentiment classification on Twitter data using machine learning techniques, primarily focusing on a Logistic Regression model. The workflow includes text preprocessing, feature extraction, model training, hyperparameter optimization, and evaluation. The project is centered around the main script: `sentiment.py`.

## Features

- **Data Loading:** Loads training and test data from CSV files.
- **Text Preprocessing:** 
  - Removes stop words and special characters
  - Handles emoticons and standardizes text
  - Provides custom tokenization and stemming
- **Exploratory Data Analysis:** 
  - Visualizes word distributions using Bokeh
  - Identifies common and important vocabulary
- **Feature Engineering:** 
  - Implements Bag-of-Words and TF-IDF vectorization
- **Modeling:**
  - Trains a Logistic Regression classifier for sentiment analysis
  - Grid search for hyperparameter optimization with cross-validation
- **Evaluation:**
  - Reports best model parameters and accuracy on validation and test sets
- **Model Persistence:** Saves the trained model using pickle for future inference
- **Example Predictions:** Shows how to use the trained model for sentiment prediction on sample tweets

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- bokeh

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Aadi1909/Twitter-Sentiment-classification-Logistic-Regression-.git
   cd Twitter-Sentiment-classification-Logistic-Regression-
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is not present, install the above libraries manually.)*

3. **Prepare the data**
   - Place your `train.csv` and `test.csv` files in the `data/` directory.
   - The CSV files should contain at least the columns: `SentimentText` and `Sentiment`.

4. **Run the main script**
   ```bash
   python sentiment.py
   ```

## Project Structure

- `sentiment.py` &mdash; Main script for preprocessing, training, evaluating, and saving the model.
- `data/`
  - `train.csv` &mdash; Training data (CSV)
  - `test.csv` &mdash; Test data (CSV)
  - `logisticRegression.pkl` &mdash; Serialized trained model (generated after training)

## Notable Code Highlights

- **Preprocessing:** Uses NLTK for stopword removal and stemming.
- **Visualization:** Bokeh is employed for word frequency histograms.
- **Pipeline:** Utilizes `scikit-learn`'s Pipeline and GridSearchCV for efficient model building and tuning.
- **Model Saving:** Stores the trained model as a pickle file for later use.

## Example: Predicting Sentiment

After training, the model can be used to predict sentiment on new tweets:
```python
twits = [
    "This is really bad, I don't like it at all",
    "I love this!",
    ":)",
    "I'm sad... :("
]
preds = clf.predict(twits)
for twit, sentiment in zip(twits, preds):
    print(f"{twit} --> {sentiment}")
```

## Acknowledgements

- The project leverages open-source libraries including scikit-learn, NLTK, pandas, and Bokeh.
- Twitter datasets for sentiment analysis.

## License

This project is licensed under the MIT License.
