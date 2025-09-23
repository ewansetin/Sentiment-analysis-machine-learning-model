# Sentiment-analysis-machine-learning-model

## Overview
This project explores sentiment analysis, a technique used to detect the emotional tone of text in order to better understand user opinions. Although sentiment analysis is widely researched, this work serves as an introduction to machine learning and evaluates the performance of different classification approaches.

The project implements a comprehensive sentiment analysis pipeline to compare supervised probabilistic and non-probabilistic classifiers, drawing on established methods and best practice. The primary aim is to assess model performance and highlight insights that may inform future applications of AI-driven text analytics.

![Screenshot](/assets/Images/MLAINLP.png)


## Research Context
This project evaluates and compares supervised learning classifiers for sentiment analysis an essential area within natural language processing and AI. Drawing on academic and online sources, it examines the application of machine learning methods to text data, with a particular focus on supervised algorithms for classifying sentiment, set within a broader ethical and professional context.


### What is Sentiment Analysis?

Social platforms produce vast quantities of unstructured text that reflect user opinions and emotions. Sentiment analysis, often referred to as opinion mining, identifies the polarity of this text (positive, negative, or neutral) and may be extended to finer gradations or combined with emotion detection to provide deeper insights.

![Screenshot](/assets/Images/SSpectrum.png)

The sentiment analysis process involves several key steps:

- Data sourcing: Collecting text from social media, online forums, or benchmark datasets such as the Stanford Sentiment Treebank (SST) or SemEval.
  
- Text processing: Cleaning and normalising text through tokenisation, lemmatisation, and noise reduction in order to improve model accuracy.
  
- Feature extraction: Transforming text into numerical representations using methods such as Bag-of-Words, TF-IDF, and N-grams.

### Common Modelling Approaches

Three principal strategies are commonly employed in sentiment analysis:

- Lexicon-based: Simple and dictionary-driven, though less automated and adaptable.
  
- Machine learning: Uses labelled data with algorithms such as Naïve Bayes, Logistic Regression, Support Vector Machines, or Random Forests.
  
- Deep learning: Employs neural networks (CNNs, RNNs, Transformers) that automatically learn complex patterns and often achieve higher accuracy.
  
  
Hybrid approaches may also combine these methods to enhance performance.


![Screenshot](/assets/Images/SAMDLpipelines.png)


### Project Focus

This project specifically compares probabilistic classifiers (e.g. Naïve Bayes, Logistic Regression) with non-probabilistic classifiers (e.g. SVM, Random Forest), examining how each approach manages uncertainty and noise. Performance is evaluated using standard metrics, including accuracy, precision, recall, and the F1-score.

---

## Key Considerations

- Dataset: The dataset should be lexically rich, readily accessible, and appropriately pre-processed to support accurate sentiment detection while minimising domain bias.
  
- Feature extraction: Techniques such as TF-IDF, Bag-of-Words, N-grams, and word embeddings are evaluated both individually and in combination to assess their influence on model performance.
  
- Model selection: Probabilistic and non-probabilistic algorithms are selected on the basis of reported accuracy, data characteristics, interpretability, and reliability.
  
- Evaluation: Data is divided into training and test sets, with cross-validation and testing on unseen data employed to ensure robust and unbiased results.


---
## Design & Workflow
The workflow follows key steps:  
Data Collection ⮕ Data Processing ⮕ Feature Extraction ⮕ Model Training ⮕ Model Evaluation. 

### Data Collection 
The project makes use of a dataset comprising textual reviews paired with numerical ratings, providing both input (text) and discrete output (sentiment labels). The reviews were selected for their emotional richness and contextual diversity, helping to reduce domain dependence while ensuring sufficient examples for training (5,000–20,000 instances).

### Data Processing 
Raw user-generated data is inherently inconsistent, necessitating comprehensive pre-processing to optimise model performance. The original five-point rating scale was reduced to three sentiment classes—negative, neutral, and positive—in order to streamline the classification task.

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

Feature extraction converts processed text into numerical representations suitable for machine learning algorithms. Common techniques include TF-IDF, Bag-of-Words (BoW), N-grams, and word embeddings. For this project, Bag-of-Words was employed to maintain consistency across classifiers while avoiding algorithm-specific biases.

### Classification Models Selection

To ensure fair evaluation across both approaches, the project employs two probabilistic classifiers (Multinomial Naïve Bayes, Logistic Regression) and two non-probabilistic classifiers (Random Forest, Support Vector Machine). These algorithms were selected on the basis of their demonstrated performance in prior research, their prevalence in sentiment analysis applications, and their suitability for review-based datasets.

### Model Evaluation

Model assessment employs multiple metrics, including accuracy, precision, recall, the F1-score, and confusion matrices. Further validation techniques comprise cross-validation and testing on unseen external datasets to ensure robust performance comparisons. This comprehensive framework supports meaningful conclusions regarding the relative effectiveness of probabilistic and non-probabilistic approaches in AI-driven sentiment analysis.

![Screenshot](/assets/Images/SApipeline.png)

### Privacy and Ethical Considerations

Working with real user-generated data requires strict adherence to privacy, data protection, and ethical standards. In the United Kingdom, the Data Protection Act 2018 (which incorporates the GDPR) sets out key requirements: data must be collected lawfully and transparently, used only for explicit purposes, stored securely, and retained no longer than necessary.

For this project, only essential data—textual reviews and associated ratings—was collected, with no personal identifiers stored. All data was kept securely on external storage and deleted upon completion of the research. The implementation also complied with the Terms of Service of each data source, ensuring that all methods of data collection remained both lawful and ethical.

--- 

## Implementation 

### Software and Hardware Environment
The sentiment analysis pipeline was implemented using Jupyter Notebook in a Python environment. The following hardware configuration supported the development and testing process:

| Component | Specification |
| --- | --- |
| Processor | 2.6 GHz 6-Core Intel Core i7 |
| Graphics | Intel UHD 630 1536 MB |
| RAM | 16 GB |
| OS | MacOS Sonoma |
| Storage | 500 GB |



### Data Collection
Data collection focused on Google Maps reviews, which provide paired text input and rating outputs ideal for sentiment classification. The dataset encompasses diverse domains including hospitals, cinemas, airports, and pubs, totaling approximately 8,000 reviews.

The dataset was strategically divided into:

* **Training & testing set** – to train models and evaluate performance  
* **Unseen dataset** – to test model generalisation  

All data processing was carried out using Pandas for efficient handling and manipulation of CSV files.



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


### Feature Extraction
The project employed the Bag-of-Words (BoW) approach using scikit-learn’s CountVectorizer() to convert processed text into numerical feature vectors suitable for machine learning classification.


### Classification Models
Four supervised learning classifiers were applied:  

* **Probabilistic:** Multinomial Naïve Bayes, Logistic Regression  
* **Non-probabilistic:** Random Forest, Support Vector Machine  

Minimal hyperparameter optimisation was conducted to obtain baseline performance estimates suitable for comparative analysis.


### Model Evaluation
Models were evaluated using:  

* **Metrics:** Accuracy, Precision, Recall, F1-score  
* **Visualisation:** Confusion matrices  
* **Validation:** 4:1 train-test split and 5-fold **StratifiedKFold** cross-validation to handle class imbalance  
* **Generalization Testing:** Performance assessment on completely unseen datasets

This methodology ensures a fair comparison of probabilistic and non-probabilistic classifiers for AI-driven sentiment analysis.

---
# Results & Discussion

The primary objective of this project was to compare the performance of probabilistic and non-probabilistic algorithms for sentiment analysis tasks. The implemented pipeline, informed by best practice identified through the literature review, demonstrated strong performance on the selected dataset, thereby validating the effectiveness of the chosen approach.

## Challenges

Data Collection and Class Distribution:The most significant challenge lay in managing uneven sentiment class distribution, particularly the scarcity of neutral reviews. This imbalance likely reflects both the inherent structure of five-star rating systems and typical user behaviour, as individuals are more inclined to leave strongly positive or negative reviews than neutral assessments. Such imbalance introduced additional complexity during both model training and evaluation.

## Outcomes

The evaluation process assessed all classifiers on both seen and unseen datasets, generating comprehensive metrics such as accuracy, precision, recall, and the F1-score. K-fold cross-validation was employed to provide additional robustness to the performance assessment.

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

When evaluated on unseen data, probabilistic classifiers exhibited a notable decline in performance, whereas non-probabilistic classifiers maintained comparatively higher accuracy. This finding aligns with prior research indicating that probabilistic models are more sensitive to noisy or unbalanced datasets, while non-probabilistic models demonstrate greater robustness when handling skewed distributions.

**Conclusion:** Both probabilistic and non-probabilistic algorithms are valid for sentiment analysis, but non-probabilistic models may be preferable for datasets with uneven class distributions.


## Room for Improvement

While the current pipeline achieved strong results, there are several ways to enhance future work:

* **Larger and more diverse datasets** – Including data from multiple platforms could help reduce domain bias and improve generalisation.  
* **Balancing techniques** – Applying methods such as SMOTE or class-weight adjustments may address class imbalance and improve neutral sentiment detection.  
* **Advanced feature extraction** – Incorporating modern techniques like word embeddings (Word2Vec, GloVe) or transformer-based embeddings could capture richer semantic relationships.  
* **Hyperparameter tuning** – Systematic optimisation of classifier parameters may boost accuracy beyond the baseline results.  
* **Real-time deployment** – Extending the pipeline into a production-ready application (e.g., a web service or dashboard) would make it more practical for business use.
* **Further Testing** - Additional testing could help identify inconsistencies in the results.

## Next Steps

Building on these findings, the following actions could guide the next phase of development:

* **Experiment with deep learning models** such as LSTM or transformer-based architectures to explore their performance on sentiment analysis. 
