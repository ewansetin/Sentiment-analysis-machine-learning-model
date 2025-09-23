# Sentiment-analysis-machine-learning-model

## Overview
This project explores sentiment analysis, a technique that detects the emotional tone of text to better understand user opinions. While sentiment analysis is widely researched, this project serves as an introduction to machine learning and evaluates the performance of different classification approaches.

The project implements a comprehensive sentiment analysis pipeline to compare supervised probabilistic and non-probabilistic classifiers, applying established methods and best practices. The primary goal is to evaluate model performance and highlight insights that could inform future AI-driven text analytics applications.

![Screenshot](/assets/Images/MLAINLP.png)


## Research Context
This project evaluates and compares supervised learning classifiers for sentiment analysis—a core area of natural language processing and AI. Drawing on academic and online sources, the project examines how machine learning methods are applied to text data, focusing on supervised algorithms for classifying sentiment within the broader ethical and professional context.


### What is Sentiment Analysis?

Social platforms generate vast amounts of unstructured text that reveal user opinions and emotions. Sentiment analysis, also known as opinion mining, detects the polarity of this text (positive, negative, or neutral) and can be extended to finer scales or combined with emotion detection for deeper insights.

![Screenshot](/assets/Images/SSpectrum.png)

The sentiment analysis process involves several key steps:

- Data sourcing: Collecting text from social media, forums, or benchmark datasets such as Stanford Sentiment Treebank (SST) or SemEval

- Text processing: Cleaning and normalizing text through tokenization, lemmatization, and noise removal to improve model accuracy

- Feature extraction: Converting text into numerical representations using methods like Bag-of-Words, TF-IDF, and N-grams




### Common Modelling Approaches

Three main strategies are commonly employed in sentiment analysis:

- Lexicon-based: Simple and dictionary-driven but less automated and adaptable

- Machine learning: Utilize labeled data with algorithms such as Naive Bayes, Logistic Regression, Support Vector Machines, or Random Forests

- Deep learning: Employ neural networks (CNNs, RNNs, Transformers) that automatically learn complex patterns and often achieve superior accuracy

Hybrid methods can combine these approaches for enhanced performance.


![Screenshot](/assets/Images/SAMDLpipelines.png)


### Project Focus

This project specifically compares probabilistic (e.g., Naive Bayes, Logistic Regression) and non-probabilistic (e.g., SVM, Random Forest) classifiers, examining how each approach handles uncertainty and noise. Performance is assessed using standard metrics including accuracy, precision, recall, and F1-score.

## Key Considerations

- Dataset: The dataset must be lexically rich, accessible, and appropriately preprocessed to support accurate sentiment detection while avoiding domain bias

- Feature extraction: Various techniques including TF-IDF, Bag-of-Words, N-grams, and word embeddings are evaluated both individually and in combination to assess their impact on model performance

- Model selection: Probabilistic and non-probabilistic algorithms are chosen based on reported accuracy, data characteristics, interpretability, and reliability

- Evaluation: Data is split into training and test sets, with cross-validation and testing on unseen data to ensure robust, unbiased results



## Design & Workflow
The workflow follows key steps:  
data collection ⮕ data processing ⮕ feature extraction ⮕ model training ⮕ evaluation. 

### Data Collection 
The project utilizes a dataset containing textual reviews paired with numerical ratings, providing both input (text) and discrete output (sentiment labels). Reviews were selected for their emotional richness and contest diversity, helping reduce domain dependence while maintaining sufficient examples for training (5,000–20,000 instances).

### Data Processing 
Raw user-generated data exhibits inherent inconsistency, necessitating comprehensive pre-processing to optimize model performance. The rating scale was simplified from five points to three sentiment classes—negative, neutral, and positive—to streamline classification tasks.

The text processing pipeline includes sequential application of:
- Contraction replacement and standardization
- Tokenization into individual word units
- Selective stop-word removal (retaining sentiment-relevant terms)
- Lemmatization and stemming for root word extraction
- Case normalization

These steps reduce noise while preserving semantic meaning, supporting better feature extraction and model learning.

| **Process** | **Description**    | **Example**    |
| :---   | :--- | :--- |
| **Replacing contractions** | This process separates combinations of words and replaces them with their standard form, e.g. “can’t” becomes “can not”.   | [Fast machine , but another apple con . Costs you a fortune and you can not do anything. You have to buy apps, which you can get more freely ones on iPhone. I never boaught anything on my iphone. I can do more on my phone for free. Far to expensive! I will return it ASAP.]   |
| **Tokenisation** | Tokenisation is a process that involves breaking down text into single word named tokens.   | ['Fast', 'machine', ',', 'but', 'another', 'apple', 'con', '.', 'Costs', 'you', 'a', 'fortune', 'and', 'you', 'can', 'not', 'do', 'anything', '.', 'You', 'have', 'to', 'buy', 'apps', ',', 'which', 'you', 'can', 'get', 'more', 'freely', 'ones', 'on', 'iPhone', '.', 'I', 'never', 'boaught', 'anything', 'on', 'my', 'iphone', '.', 'I', 'can', 'do', 'more', 'on', 'my', 'phone', 'for', 'free', '.', 'Far', 'to', 'expensive', '!', 'I', 'will', 'return', 'it', 'ASAP', '.']   |
| **Partial stop-words removal** | This process involves removing stop words such as "the", "of", "and", "a", "to", and "in", but keeping stop words such as “not” and “but”.  | ['Fast', 'machine', ',', 'another', 'apple', 'con', '.', 'Costs', 'fortune', '.', 'buy', 'apps', ',', 'freely', 'ones', 'iPhone', '.', 'never', 'boaught', 'iphone', '.', 'phone', 'free', '.', 'Far', 'expensive', '!', 'return', 'ASAP', '.']   |
| **Lemmatisation** | Lemmatisation converts words to their base or root form (lemma)  | ['Fast', 'machine', ',', 'another', 'apple', 'con', '.', 'cost', 'fortune', '.', 'buy', 'apps', ',', 'freely', 'one', 'iPhone', '.', 'never', 'boaught', 'iPhone', '.', 'phone', 'free', '.', 'Far', 'expensive', '!', 'return', 'ASAP', '.']   |
| **Stemming** | Stemming involves truncating suffixes to convert words into their root form.  | ['Fast', 'machin', ',', 'anoth', 'appl', 'con', '.', 'cost', 'fortun', '.', 'buy', 'app', ',', 'free', 'one', 'iPhone', '.', 'never', 'boaught', 'iPhone', '.', 'phone', 'free', '.', 'Far', 'expens', '!', 'return', 'ASAP', '.']   |

### Feature Extraction

Feature extraction transforms processed text into numeric representations suitable for machine learning algorithms. Common techniques include TF-IDF, Bag-of-Words (BoW), N-grams, and word embeddings. This project employs Bag-of-Words to maintain consistency across classifiers while avoiding algorithm-specific biases.

### Classification Models Selection

To ensure fair evaluation across both approaches, the project implements two probabilistic classifiers (Multinomial Naive Bayes, Logistic Regression) and two non-probabilistic classifiers (Random Forest, Support Vector Machine). These algorithms were selected based on their demonstrated performance in prior research, prevalence in sentiment analysis applications, and compatibility with review-based datasets.

### Model Evaluation

Model assessment employs multiple metrics including accuracy, precision, recall, F1-score, and confusion matrices. Additional validation techniques include cross-validation and testing on unseen external datasets to ensure robust performance comparisons. This comprehensive framework enables meaningful conclusions about the relative effectiveness of probabilistic versus non-probabilistic approaches in AI-driven sentiment analysis.

![Screenshot](/assets/Images/SApipeline.png)

### Privacy and Ethical Considerations

Working with real user-generated data requires strict adherence to privacy, data protection, and ethical guidelines. In the UK, the Data Protection Act 2018 (implementing GDPR) establishes key requirements: data must be collected lawfully and transparently, used only for explicit purposes, stored securely, and retained no longer than necessary.

This project collected only essential data—text reviews and associated ratings—without storing personal identifiers. All data was maintained securely on external storage and deleted upon research completion. The implementation also respected each data source's Terms of Service, ensuring all data gathering methods remained compliant and ethical.

## How I implemented it 

### Software and Hardware Environment
The sentiment analysis pipeline was implemented using Jupyter Notebook in a Python environment. The following hardware configuration supported the development and testing process:

| Component | Specification |
| --- | --- |
| Processor | 2.6 GHz 6-Core Intel Core i7 |
| Graphics | Intel UHD 630 1536 MB |
| RAM | 16 GB |
| OS | MacOS Sonoma |
| Storage | 500 GB |

---

### Data Collection
Data collection focused on Google Maps reviews, which provide paired text input and rating outputs ideal for sentiment classification. The dataset encompasses diverse domains including hospitals, cinemas, airports, and pubs, totaling approximately 8,000 reviews.

The dataset was strategically divided into:

* **Training & testing set** – to train models and evaluate performance  
* **Unseen dataset** – to test model generalisation  

All data processing utilized Pandas for efficient CSV file handling and manipulation.

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

These steps were implemented using **Contractions** and **NLTK**, with functions applied sequentially to preserve effectiveness. Stop-word removal was customised to retain polarity-relevant words like “not” and "never".

---

### Feature Extraction
The project implemented Bag-of-Words (BoW) using Scikit-learn's CountVectorizer() to convert processed text into numerical feature vectors suitable for machine learning classification.

---

### Classification Models
Four supervised learning classifiers were applied:  

* **Probabilistic:** Multinomial Naïve Bayes, Logistic Regression  
* **Non-probabilistic:** Random Forest, Support Vector Machine  

Minimal hyperparameter optimization was applied to obtain baseline performance estimates suitable for comparative analysis.

---

### Model Evaluation
Models were evaluated using:  

* **Metrics:** Accuracy, Precision, Recall, F1-score  
* **Visualisation:** Confusion matrices  
* **Validation:** 4:1 train-test split and 5-fold **StratifiedKFold** cross-validation to handle class imbalance  
* **Generalization Testing:** Performance assessment on completely unseen datasets

This methodology ensures a fair comparison of probabilistic and non-probabilistic classifiers for AI-driven sentiment analysis.

# Results & Discussion

The primary objective of this project was comparing the performance of probabilistic and non-probabilistic algorithms for sentiment analysis tasks. The implemented pipeline, based on best practices identified through literature review, demonstrated strong performance on the selected dataset, validating the effectiveness of the chosen approach.

## Challenges

Data Collection and Class Distribution: The most significant challenge involved managing uneven sentiment class distribution, particularly the scarcity of neutral reviews. This imbalance likely reflects both the inherent structure of 5-star rating systems and typical user behavior—users tend to leave strongly positive or negative reviews rather than neutral assessments. This class imbalance introduced additional complexity during model training and evaluation phases.

## Outcomes

The evaluation process assessed all classifiers using both seen and unseen datasets, generating comprehensive metrics including accuracy, precision, recall, and F1-scores. K-fold cross-validation provided additional robustness to the performance assessment.

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
