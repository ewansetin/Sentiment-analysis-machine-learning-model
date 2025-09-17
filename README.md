# Sentiment-analysis-machine-learning-model

## Overview
This project explored sentiment analysis, which detects the emotional tone of text to better understand users. While widely researched, there is still room to assess how different machine-learning approaches perform.

I built a sentiment-analysis pipeline to compare supervised probabilistic and non-probabilistic classifiers, applying established methods and best practices. The goal was to evaluate model performance and highlight insights that could inform future AI-driven text analytics.


## My Research 
This project evaluates and compares supervised learning classifiers for sentiment analysis—a core area of natural language processing and AI. Drawing on academic and online sources, the review looks at how machine learning methods are applied to text data, focusing on supervised algorithms for classifying sentiment while noting the broader ethical and professional context.

### Sentiment Analysis

Social platforms generate huge amounts of unstructured text that reveal user opinions and emotions. Sentiment analysis, or opinion mining, detects the polarity of this text—positive, negative, or neutral—and can also be extended to finer scales or combined with emotion detection for richer insight.

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

## Process 
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