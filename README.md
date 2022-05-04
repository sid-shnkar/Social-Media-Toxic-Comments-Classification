# Social-Media-Toxic-Comments-Classification
A Machine learning project on NLP to detect different types of toxicity like threats, obsenity, insults, and identity-based hate. in the comments given in the dataset.

## Description
The dataset used in this project consists of three files present in the 'data' folder : train.csv, test.csv and test_labels.csv
The data in the training set is in the form of comments which have been labelled by human raters for toxic behavior. These comments are classified into six types of toxicity: **toxic**, **severe_toxic**, **obscene**, **threat**, **insult** and **identity_hate** .

## Implementation

Following are the phases in which this project has been implemented:

## Explorative Data Analysis

The train data has 159571 observations with 8 columns and the test data has 153164 observations with 2 columns. A plot showing the count of each of the six labels was plotted and it was observed that the label 'toxic' has the most observations in the training dataset while 'threat' label has the least observations.

![smtcc-p1](https://user-images.githubusercontent.com/104520126/166691347-d289bf6c-5463-4f33-ae57-afa787c464b3.jpg)
 
 A **cross-correlation matrix** for each label was plotted to see which labels are likely to appear together with a comment and it was observed that 'obscene' label had a higher chance to be 'insulting' at the same time.
Further, to visualize the most common words contributing to different labels, separate **word cloud** was generated for each label.

## Feature Engineering

In order to fit the comments properly into the model, **tokenization** was used to remove punctuations, special characters and non-ascii characters from the comments. Then  another technique called as **lemmatization** was used and all the comments with length less than three were filtered out. Next **TFIDF** vectorizer was used to scale down the impact of tokens that occur very frequently in a given corpus which are empirically less informative than features that occur in a small fraction of the training corpus.

## Model Selection



## Hyperparameter tuning

## Ensembling

## Results
