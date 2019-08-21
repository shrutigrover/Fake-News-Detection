import os
import re
import nltk
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

import pandas as pd
from sklearn import feature_extraction
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_distances
from time import time

_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def gen_or_load_feats(feat_fn, headlines, bodies, feature_file):
    if not os.path.isfile(feature_file):
        feats = feat_fn(headlines, bodies)
        np.save(feature_file, feats)
    return np.load(feature_file)

def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X

#refuting_features for each headline checks if the the headline contains any refuting word
def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X

#polarity_features
def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 3)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X

## New Features Added Below ##
## Implements non negative matrix factorization. Calculates the cos distance between the resulting head and body vector.
def NMF_cos_50(headlines, bodies):
    ## Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    head_and_body = [clean(headline) + " " + clean(body) for i, (headline, body) in enumerate(zip(headlines, bodies))]

    vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
    X_all = vectorizer_all.fit_transform(head_and_body)
    vocab = vectorizer_all.vocabulary_
    print("NMF_topics: complete vocabulary length=" + str(len(list(vocab.keys()))))

    # calculates n most important topics of the bodies. Each topic contains all words but ordered by importance. The
    # more important topic words a body contains of a certain topic, the higher its value for this topic
    nfm = NMF(n_components=50, random_state=1, alpha=.1)
    print("NMF_topics: fit and transform body")
    t0 = time()
    nfm.fit_transform(X_all)
    print("done in %0.3fs." % (time() - t0))
    vectorizer_head = TfidfVectorizer(vocabulary=vocab, norm='l2')
    X_train_head = vectorizer_head.fit_transform(headlines)

    vectorizer_body = TfidfVectorizer(vocabulary=vocab, norm='l2')
    X_train_body = vectorizer_body.fit_transform(bodies)

    print("NMF_topics: transform head and body")
    print("X_train_head.shape ::",X_train_head.shape)
    nfm_head_matrix = nfm.transform(X_train_head)
    nfm_body_matrix = nfm.transform(X_train_body)
    #cosine distanced
    X = []
    for i in range(len(nfm_head_matrix)):
        X_head_vector = np.array(nfm_head_matrix[i]).reshape((1, -1))  # 1d array is deprecated
        X_body_vector = np.array(nfm_body_matrix[i]).reshape((1, -1))
        cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
        X.append(cos_dist.tolist())

    return X

### Implements LDA. Calculates the cos distance between the resulting head and body vector.
def LDA_cos_25(headlines, bodies):
    head_and_body = [clean(headline) + " " + clean(body) for i, (headline, body) in enumerate(zip(headlines, bodies))]
    #Bag of words
    vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), stop_words='english', use_idf=True, norm='l2')
    X_all = vectorizer_all.fit_transform(head_and_body)
    vocab = vectorizer_all.vocabulary_

    vectorizer_head = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
    X_train_head = vectorizer_head.fit_transform(headlines)

    vectorizer_body = TfidfVectorizer(vocabulary=vocab, use_idf=False, norm='l2')
    X_train_body = vectorizer_body.fit_transform(bodies)
    lda_body = LatentDirichletAllocation(n_topics=25, learning_method='online', random_state=0, n_jobs=3)

    print("latent_dirichlet_allocation_cos: fit and transform body")
    t0 = time()
    lda_body_matrix = lda_body.fit_transform(X_train_body)
    print("done in %0.3fs." % (time() - t0))

    print("latent_dirichlet_allocation_cos: transform head")
    # use the lda trained for body topcis on the headlines => if the headlines and bodies share topics
    # their vectors should be similar
    lda_head_matrix = lda_body.transform(X_train_head)

    print('latent_dirichlet_allocation_cos: calculating cosine distance between head and body')
    # calculate cosine distance between the body and head
    X = []
    for i in range(len(lda_head_matrix)):
        X_head_vector = np.array(lda_head_matrix[i]).reshape((1, -1)) #1d array is deprecated
        X_body_vector = np.array(lda_body_matrix[i]).reshape((1, -1))
        cos_dist = cosine_distances(X_head_vector, X_body_vector).flatten()
        X.append(cos_dist.tolist())
    return X
