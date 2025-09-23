# Sentiment-analysis-machine-learning-model

## Overview
My project explored sentiment analysis, which detects the emotional tone of text to better understand users. While sentiment analysis is widely researched, I undertook this project to introduce myself to machine learning and to evaluate the performance of different approaches.

I built a sentiment-analysis pipeline to compare supervised probabilistic and non-probabilistic classifiers, applying established methods and best practices. The goal was to evaluate model performance and highlight insights that could inform future AI-driven text analytics.

![Screenshot](/assets/Images/MLAINLP.png)


## My Research 
This project evaluates and compares supervised learning classifiers for sentiment analysis—a core area of natural language processing and AI. Drawing on academic and online sources, the review looks at how machine learning methods are applied to text data, focusing on supervised algorithms for classifying sentiment while noting the broader ethical and professional context.

### Sentiment Analysis

Social platforms generate huge amounts of unstructured text that reveal user opinions and emotions. Sentiment analysis, or opinion mining, detects the polarity of this text—positive, negative, or neutral—and can also be extended to finer scales or combined with emotion detection for richer insight.

![Screenshot](/assets/Images/SSpectrum.png)

Key steps include:

- Data sourcing: from social media, forums, or benchmark datasets such as SST or SemEval.

- Text processing: cleaning and normalising text (tokenising, lemmatising, removing noise) to improve model accuracy.

- Feature extraction: using methods like Bag-of-Words, TF-IDF, and N-grams to turn text into usable numeric features.




### Modelling Approaches

Three main strategies are common:

- Lexicon-based: simple and dictionary-driven but less automated.

- Machine learning: uses labelled data with algorithms such as Naive Bayes, Logistic Regression, SVMs, or Random Forests.

- Deep learning: neural networks (CNNs, RNNs) that learn complex patterns automatically and often deliver the best accuracy.

Hybrid methods can combine these for improved results.


![Screenshot](/assets/Images/SAMDLpipelines.png)


### Supervised Learning Focus

This project compares probabilistic (e.g. Naive Bayes, Logistic Regression) and non-probabilistic (e.g. SVM, Random Forest) classifiers, highlighting how each handles uncertainty and noise. Performance is assessed with standard metrics such as accuracy, precision, recall, and F1 score.

## What is requred 

To compare probabilistic and non-probabilistic machine-learning approaches for sentiment analysis, I built a system that applies both types of classifiers using established sentiment-analysis methods.

Key considerations:

- Dataset: must be rich in lexicon, accessible, and pre-processed to support accurate sentiment detection and avoid domain bias.

- Feature extraction: techniques such as TF-IDF, Bag-of-Words, N-grams, or word embeddings are evaluated—individually and in combination—to see how they influence model performance.

- Model selection: probabilistic and non-probabilistic algorithms are chosen not just for reported accuracy but also for data characteristics, interpretability, and reliability.

- Evaluation: data are split into training and test sets, with cross-validation and tests on unseen data to ensure robust, unbiased results.

This setup enables a fair comparison of the two modelling approaches and supports meaningful conclusions about their effectiveness in AI-driven sentiment analysis.

## My Design 
The workflow follows key steps: data collection, data processing, feature extraction, model training, and evaluation. The goal is to identify which classifiers perform best for text-based sentiment classification.

I selected a dataset with textual reviews linked to ratings, providing both input (text) and discrete output (sentiment label). Reviews were chosen for their richness in emotional cues and diversity, helping reduce domain dependence while maintaining enough examples for training (5,000–20,000 instances).

### Data Processing 
Raw data is user-generated and inconsistent, so pre-processing was applied to improve model performance. Ratings were mapped to three classes—negative, neutral, positive—to simplify classification. Text processing included: Replacing contractions, Tokenisation, Partial stop-word removal, Lemmatisation and stemming.

These steps reduce noise while preserving semantic meaning, supporting better feature extraction and model learning.

| **Process** | **Description**    | **Example**    |
| :---   | :--- | :--- |
| **Replacing contractions** | This process separates combinations of words and replaces them with their standard form, e.g. “can’t” becomes “can not”.   | [Fast machine , but another apple con . Costs you a fortune and you can not do anything. You have to buy apps, which you can get more freely ones on iPhone. I never boaught anything on my iphone. I can do more on my phone for free. Far to expensive! I will return it ASAP.]   |
| **Tokenisation** | Tokenisation is a process that involves breaking down text into single word named tokens.   | ['Fast', 'machine', ',', 'but', 'another', 'apple', 'con', '.', 'Costs', 'you', 'a', 'fortune', 'and', 'you', 'can', 'not', 'do', 'anything', '.', 'You', 'have', 'to', 'buy', 'apps', ',', 'which', 'you', 'can', 'get', 'more', 'freely', 'ones', 'on', 'iPhone', '.', 'I', 'never', 'boaught', 'anything', 'on', 'my', 'iphone', '.', 'I', 'can', 'do', 'more', 'on', 'my', 'phone', 'for', 'free', '.', 'Far', 'to', 'expensive', '!', 'I', 'will', 'return', 'it', 'ASAP', '.']   |
| **Partial stop-words removal** | This process involves removing stop words such as "the", "of", "and", "a", "to", and "in", but keeping stop words such as “not” and “but”.  | ['Fast', 'machine', ',', 'another', 'apple', 'con', '.', 'Costs', 'fortune', '.', 'buy', 'apps', ',', 'freely', 'ones', 'iPhone', '.', 'never', 'boaught', 'iphone', '.', 'phone', 'free', '.', 'Far', 'expensive', '!', 'return', 'ASAP', '.']   |
| **Lemmatisation** | Lemmatisation converts words to their base or root form (lemma)  | ['Fast', 'machine', ',', 'another', 'apple', 'con', '.', 'cost', 'fortune', '.', 'buy', 'apps', ',', 'freely', 'one', 'iPhone', '.', 'never', 'boaught', 'iPhone', '.', 'phone', 'free', '.', 'Far', 'expensive', '!', 'return', 'ASAP', '.']   |
| **Stemming** | Stemming involves truncating suffixes to convert words into their root form.  | ['Fast', 'machin', ',', 'anoth', 'appl', 'con', '.', 'cost', 'fortun', '.', 'buy', 'app', ',', 'free', 'one', 'iPhone', '.', 'never', 'boaught', 'iPhone', '.', 'phone', 'free', '.', 'Far', 'expens', '!', 'return', 'ASAP', '.']   |

### Feature Extraction

Feature extraction transforms text into numeric representations for the models. Techniques like TF-IDF, Bag-of-Words (BoW), N-grams, or word embeddings are common. For this project, BoW was used to avoid introducing biases across classifiers, while keeping the preprocessing consistent.

### Classification Models

To fairly evaluate both approaches, we selected two probabilistic classifiers (Naive Bayes, Logistic Regression) and two non-probabilistic classifiers (Random Forest, SVM). These were chosen based on their strong performance in prior research, prevalence in sentiment analysis tasks, and compatibility with reviews-based datasets.

### Model Evaluation

Models were assessed using accuracy, precision, recall, F1-score, and confusion matrices. Additional validation included cross-validation and testing on unseen or external data to ensure robust comparisons. This framework allows conclusions about the effectiveness of probabilistic vs. non-probabilistic approaches in AI-driven sentiment analysis.

![Screenshot](/assets/Images/SApipeline.png)

### Privacy and Ethical Considerations

Working with real user data requires strict attention to privacy, data protection, and ethics. In the UK, the Data Protection Act 2018 (which implements GDPR) sets key rules: data must be collected lawfully and transparently, used only for explicit purposes, stored securely, and kept no longer than necessary.

For this project, only the essential data—text reviews and ratings—was collected. No personal identifiers were stored, and all data was held securely on an external drive and deleted after the research concluded.

We also respected each data source’s Terms of Service. For example, platforms like Facebook prohibit automated scraping without permission. All data gathering methods followed these requirements to ensure the project remained compliant and ethical.

## How I implemented it 

### Software and Hardware
The sentiment analysis pipeline was implemented in **Jupyter Notebook**. We recommend using a similar environment to run the notebook and CSV files.

**Hardware used:**  

| Component | Specification |
| --- | --- |
| Processor | 2.6 GHz 6-Core Intel Core i7 |
| Graphics | Intel UHD 630 1536 MB |
| RAM | 16 GB |
| OS | MacOS Sonoma |
| Storage | 500 GB |

---

### Data Collection
Data was collected from **Google Maps reviews**, which provide text (input) and ratings (output) for sentiment classification. Reviews covered diverse domains, such as hospitals, cinemas, airports, and pubs, with a total of ~8,000 reviews. The dataset was split into:

* **Training & testing set** – to train models and evaluate performance  
* **Unseen dataset** – to test model generalisation  

CSV files were processed using **Pandas**.

---

### Data Processing
Ratings were converted from a 5-point scale to a 3-point sentiment scale:

* Ratings 1–2 → Negative (-1)  
* Rating 3 → Neutral (0)  
* Ratings 4–5 → Positive (1)  

Text processing included:  

* Replacing contractions  
* Tokenisation  
* Partial stop-word removal  
* Lemmatisation and stemming  
* Lowercasing  

These steps were implemented using **Contractions** and **NLTK**, with functions applied sequentially to preserve effectiveness. Stop-word removal was customised to retain polarity-relevant words like “not”.

---

### Feature Extraction
We used **Bag-of-Words (BoW)** via `CountVectorizer()` from Scikit-learn to convert text into numerical features for classification.

---

### Classification Models
Four supervised learning classifiers were applied:  

* **Probabilistic:** Multinomial Naïve Bayes, Logistic Regression  
* **Non-probabilistic:** Random Forest, Support Vector Machine  

Minimal optimisation was applied to obtain general performance estimates for comparison.

---

### Model Evaluation
Models were evaluated using:  

* **Metrics:** Accuracy, Precision, Recall, F1-score  
* **Visualisation:** Confusion matrices  
* **Validation:** 4:1 train-test split and 5-fold **StratifiedKFold** cross-validation to handle class imbalance  
* **Unseen data testing:** to assess generalisation  

This methodology ensures a fair comparison of probabilistic and non-probabilistic classifiers for AI-driven sentiment analysis.

# Results & Discussion

The main goal of this project was to compare the performance of probabilistic and non-probabilistic algorithms for sentiment analysis. We built a sentiment analysis pipeline using best practices identified from our literature review. Overall, the pipeline performed well on the chosen dataset, suggesting that our implementation was effective.

## Challenges

One of the main challenges was **data collection**. We observed an uneven distribution of sentiment classes, particularly a low number of neutral reviews. This likely reflects both the structure of the 5-star rating scale and user behaviour—people are more likely to leave strongly positive or negative reviews than neutral ones. This imbalance created additional complexity when training the models.

## Outcomes

We evaluated the classifiers using both **seen** and **unseen** data, generating metrics such as accuracy, precision, recall, and F1-score. We also applied **k-fold cross-validation** for a more robust assessment.

**Seen data accuracy scores:**  
- Naïve Bayes: 88%  
- Logistic Regression: 88%  
- Random Forest: 85%  
- Support Vector Machine: 86%  

Results indicate that both probabilistic and non-probabilistic classifiers performed well on balanced data, with probabilistic classifiers slightly outperforming the others.  

**Unseen data accuracy scores:**  
- Naïve Bayes: 77%  
- Logistic Regression: 76%  
- Random Forest: 81%  
- Support Vector Machine: 82%  

When tested on unseen data, probabilistic classifiers showed a notable drop in performance, while non-probabilistic classifiers maintained higher accuracy. This aligns with research showing probabilistic models are more sensitive to noisy or unbalanced datasets, while non-probabilistic models handle skewed distributions more robustly.

**Conclusion:** Both probabilistic and non-probabilistic algorithms are valid for sentiment analysis, but non-probabilistic models may be preferable for datasets with uneven class distributions.


## Room for Improvement

While the current pipeline achieved strong results, there are several ways to enhance future work:

* **Larger and more diverse datasets** – Including data from multiple platforms could help reduce domain bias and improve generalisation.  
* **Balancing techniques** – Applying methods such as SMOTE or class-weight adjustments may address class imbalance and improve neutral sentiment detection.  
* **Advanced feature extraction** – Incorporating modern techniques like word embeddings (Word2Vec, GloVe) or transformer-based embeddings could capture richer semantic relationships.  
* **Hyperparameter tuning** – Systematic optimisation of classifier parameters may boost accuracy beyond the baseline results.  
* **Real-time deployment** – Extending the pipeline into a production-ready application (e.g., a web service or dashboard) would make it more practical for business use.

## Next Steps

Building on these findings, the following actions could guide the next phase of development:

* **Experiment with deep learning models** such as LSTM or transformer-based architectures to explore their performance on sentiment analysis. 
