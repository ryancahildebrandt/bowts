# **B**ag **O**f **W**ords + **T**erm **S**imilarity

---

[*Open*](https://gitpod.io/#https://github.com/ryancahildebrandt/bowts) *in gitpod*
[*Open*](https://mybinder.org/v2/gh/ryancahildebrandt/bowts/HEAD) *in binder*
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ryancahildebrandt/bowts/HEAD)

## Purpose
This project is an experiment on the effectiveness of using pretrained word embeddings to alter the embedding vector spaces created by bag of word embedding models. Much of the work in the NLP/NLU space relies on pretrained word/sentence embeddings or larger transformer-based models, but the lightweight and relatively quick implementation of bag of words models continues to hold value for many routine NLP tasks. One drawback of bag of words embedding models is the variability of their embedding vector spaces, with the length of any embedding vector depending on the vocabulary contained in the embedded documents. Another drawback of bag of words embeddings is that they are largely unable to account for the meaning of different words in the way pretrained word embeddings are. Take for example the sentence "I need a chicken tender, but a chicken nugget would do.". Pretrained embeddings are able to account for the different senses of words and words that are similar, such that both the individual word embeddings and the sentence embeddings for this example sentence would consider "chicken nugget" and "chicken tender" to be quite similar. By contrast, bag of words models can only encode the individual words or n-grams within the sentence, and will necessarily treat them as unique regardless of how sematically similar they may be. So in a bag of words model, "chicken nugget" and "chicken tender" would be treated as two values in the embedding space that are just as different as "I" and "chicken" or any other two words. This is where pretrained embeddings may be able to help by accounting for similar terms in the documents to be embedded via bag of words models. So for this project, I'll be looking into the process and benefits of combining these two embedding approaches, and looking to answer the following question:
- **Can we use pretrained word embeddings to reduce a bag of words model embedding vector space by combining semantically similar terms, and if so, does this offer any accuracy benefit in a text classification task?**

---

## Datasets
The datasets used for the current project were pulled from the following: 
- [Bitext Customer Support](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants), for shorter documents
- [Multilabel Classification from Analytics Vidhya Hackathon, Abstracts](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset?select=train.csv), for longer documents
- [Multilabel Classification from Analytics Vidhya Hackathon, Titles](https://www.kaggle.com/datasets/shivanandmn/multilabel-classification-dataset?select=train.csv), for mid-length documents

---

## Outputs
- The results [table](./outputs/results_max_df.csv) for all datasets for the maximum subset of processable documents
- The results [table](./outputs/results_sample_df.csv) for all datasets for the sample of 3000 documents per dataset
- The report outlining the project approach and results, in [jmd](./bowts.jmd) and [html](./bowts.html) formats
- The interactive Pluto.jl [notebook](./bowts_pl.jl), for playing around with and visuzlizing algorithm parameters
