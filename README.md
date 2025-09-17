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

Attempt | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

| Attempt | #1    | #2    |
| :---:   | :---: | :---: |
| Seconds | 301   | 283   |

First Header  | Second Header
------------- | -------------
Content Cell  | Content Cell
Content Cell  | Content Cell
