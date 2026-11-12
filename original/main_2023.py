from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
from collections import defaultdict

# Download scikit-learn library
# !pip install scikit-learn

from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier

# Download NLTK library
# !pip install nltk


import nltk
from nltk.stem import PorterStemmer

import string
import numpy as np

# Download TextBlob library
# !pip install textblob

from textblob import TextBlob

stop_words = ['a', 'all', 'an', 'and', 'any', 'are', 'as', 'be', 'been', 'but', 'by', 'few', 'for', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i', 'in', 'is', 'it', 'its', 'many', 'me', 'my', 'none', 'of', 'on', 'or', 'our', 'she', 'some', 'the', 'their', 'them', 'there', 'they', 'that', 'this', 'us', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'will', 'with', 'you', 'your']

def get_pos_tags(text):
    """
    Get the part-of-speech (POS) tags for each word in the given text.
    
    Args:
        text (str): The input text to process.
    
    Returns:
        str: A string representation of the POS tags for the text.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tagged_text = ' '.join([tag for word, tag in pos_tags])
    
    return pos_tagged_text


def get_sentiment_score(text):
    """
    Get the sentiment score for the given text using TextBlob.
    
    Args:
        text (str): The input text to analyze.
    
    Returns:
        str: A formatted string representation of the sentiment score.
    """
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    return f"sentiment_{sentiment_score:.2f}"

def read_files(path, label):
    """
    Reads all text files located in the 'path' and assigns them to 'label' class.
    
    Args:
        path (str): The path to the folder containing the text files.
        label (str): The label/class assigned to the text files.
        
    Returns:
        list: A list of tuples containing the text content and corresponding label.
    """
    files = glob.glob(path + os.sep + label + os.sep + '*.txt')
    texts = []
    for i, v in enumerate(files):
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        f.close()
        
    return texts


def read_files_pos(path, label):
    """
    Read files from the specified path and associate them with the given label.
    
    Args:
        path (str): The path where the files are located.
        label (str): The label to associate with the files.
    
    Returns:
        list: A list of tuples containing the processed text and corresponding label.
    """
    filelist = glob.glob(path + os.sep + label + os.sep + '*.txt')
    documents = []
    
    for filename in filelist:
        with open(filename, 'r', encoding='utf-8') as f:
            
            text = f.read()
            pos_tags = get_pos_tags(text)
            sentiment_score = get_sentiment_score(text)
            text_with_features = f"{text} {pos_tags} {sentiment_score}"
            documents.append((text_with_features, label))
            
    return documents

def extract_vocabulary(texts, n, ft):
    """
    Extracts all character 'n'-grams occurring at least 'ft' times in a set of 'texts'.

    Args:
        texts (list): List of tuples where each tuple contains a text and its label.
        n (int): The length of the character n-grams to extract.
        ft (int): The minimum frequency threshold for the character n-grams.

    Returns:
        list: List of character n-grams (vocabulary) that occur at least 'ft' times.
    """
    # Dictionary to store the occurrences of character n-grams
    occurrences = defaultdict(int)

    # Count the occurrences of character n-grams in each text
    for text, label in texts:
        text_occurrences = represent_text(text, n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]

    # Extract the character n-grams that occur at least 'ft' times
    vocabulary = []
    
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)

    return vocabulary


def represent_text(text, n):
    """
    Extracts all character 'n'-grams and POS 'n'-grams from a 'text'.

    Args:
        text (str): The input text.
        n (int): The length of the character n-grams and POS n-grams.

    Returns:
        dict: A dictionary containing the frequency count of each extracted n-gram.
    """
    tokens = []
    if n > 0:
        tokens.extend([text[i:i+n] for i in range(len(text)-n+1)])
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
        
    return frequency


def remove_punctuations_from_docs(train_docs):
    """
    Remove punctuations from documents in the train_docs list.

    Args:
        train_docs (list): List of tuples where each tuple contains a document and its author.

    Returns:
        list: List of tuples where each tuple contains the document with punctuations removed and its author.
    """
    # Get punctuation list
    punctuations = set(string.punctuation+'–')

    # List to store documents with punctuations removed
    docs_without_punctuations = []

    for doc, author in train_docs:
        # Remove punctuations from document
        doc_without_punctuations = ''.join([char for char in doc if char not in punctuations])

        # Append tuple of document with punctuations removed and its author to docs_without_punctuations list
        docs_without_punctuations.append((doc_without_punctuations, author))

    return docs_without_punctuations


def case_folding(train_docs):
    """
    Convert all documents in train_docs list to lowercase.

    Args:
        train_docs (list): List of tuples where each tuple contains a document and its author.

    Returns:
        list: List of tuples where each tuple contains a lowercase document and its author.
    """
    # List to store documents converted to lowercase
    docs_lowercase = []

    for doc, author in train_docs:
        # Convert document to lowercase
        doc_lower = doc.lower()

        # Append tuple of lowercase document and its author to docs_lowercase list
        docs_lowercase.append((doc_lower, author))

    return docs_lowercase


def stem_documents(train_docs):
    """
    Apply PorterStemmer to each document in train_docs list.

    Args:
        train_docs (list): List of tuples where each tuple contains a document and its author.

    Returns:
        list: List of tuples where each tuple contains a stemmed document and its author.
    """
    stemmed_docs = []
    porter = PorterStemmer()

    for doc, author in train_docs:
        words = doc.split()  # Split document into words
        # Apply PorterStemmer to each word
        stemmed_words = [porter.stem(word) for word in words]
        # Merge stemmed words by putting spaces between them
        stemmed_doc = ' '.join(stemmed_words)
        # Append stemmed document and author to stemmed_docs list
        stemmed_docs.append((stemmed_doc, author))

    return stemmed_docs


def remove_stopwords_from_docs(train_docs, stop_words):
    """
    Remove stopwords from documents in the train_docs list.

    Args:
        train_docs (list): List of tuples where each tuple contains a document and its author.
        stop_words (list): List of stopwords to remove.

    Returns:
        list: List of tuples where each tuple contains the document with stopwords removed and its author.
    """
    # List to store documents with stopwords removed
    docs_without_stopwords = []

    for doc, author in train_docs:
        # Tokenize document into words
        words = doc.split()

        # Remove stopwords
        words_filtered = [word for word in words if word.lower() not in stop_words]

        # Join filtered words back into a single document string
        doc_without_stopwords = ' '.join(words_filtered)

        # Append tuple of document with stopwords removed and its author to docs_without_stopwords list
        docs_without_stopwords.append((doc_without_stopwords, author))

    return docs_without_stopwords


def baseline(path, outpath, n=4, ft=5, pt=0.1):
    """
    Runs a single SVM classifier on a set of texts, and saves the predicted authorship for each text to a file.
    
    Args:
        path (str): The path where the input texts are located.
        outpath (str): The path where the output file will be saved.
        n (int): The length of the character n-grams to extract.
        ft (int): The minimum frequency threshold for the character n-grams.
        pt (float): The threshold for rejecting predicted authorship if the difference between the two highest probabilities is less than `pt`.
    """
    start_time = time.time()
    
    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    
    # Iterate over each problem in the collection
    for index, problem in enumerate(problems[0:5]):
        print(problem)
        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        
        # Building training set
        train_docs = []
        
        for candidate in candidates:
            train_docs.extend(read_files(path + os.sep + problem, candidate))
            
        train_docs = remove_punctuations_from_docs(train_docs)
        train_docs = stem_documents(train_docs)
        train_docs = remove_stopwords_from_docs(train_docs, stop_words)
        
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs, n, ft)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False, vocabulary=vocabulary)
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        
        # Building test set
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_docs = remove_punctuations_from_docs(test_docs)
        test_docs = stem_documents(test_docs)
        test_docs = remove_stopwords_from_docs(test_docs, stop_words)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')
        
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        
        clf = CalibratedClassifierCV(OneVsRestClassifier(SVC(C=1)))
        clf.fit(scaled_train_data, train_labels)
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)

        # Reject option (used in open-set cases)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count = count + 1
        print('\t', count, 'texts left unattributed')

        # Saving output data
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
            
        with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        print('\t', 'answers saved to file', 'answers-' + problem + '.json')

    print('elapsed time:', time.time() - start_time)

def baseline_bagging(path, outpath, n=4, ft=5, pt=0.01):
    """
    Runs the baseline algorithm with bagging on a set of texts and saves the predicted authorship for each text to a file.

    Args:
        path (str): The path where the input texts are located.
        outpath (str): The path where the output file will be saved.
        n (int): The length of the character n-grams to extract.
        ft (int): The minimum frequency threshold for the character n-grams.
        pt (float): The threshold for rejecting predicted authorship if the difference between the two highest probabilities is less than `pt`.
    """
    start_time = time.time()

    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    # Iterate over each problem in the collection
    for index, problem in enumerate(problems[0:5]):
        print(problem)

        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])

        # Building training set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(path + os.sep + problem, candidate))
        train_docs = remove_punctuations_from_docs(train_docs)
        train_docs = stem_documents(train_docs)
        train_docs = remove_stopwords_from_docs(train_docs, stop_words)
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs, n, ft)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False, vocabulary=vocabulary)
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))

        # Building test set
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_docs = remove_punctuations_from_docs(test_docs)
        test_docs = stem_documents(test_docs)
        test_docs = remove_stopwords_from_docs(test_docs, stop_words)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')

        # Applying SVM with bagging
        # Scale the data
        max_abs_scaler = MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)

        # Create a bagging classifier with SVM base estimator
        clf = CalibratedClassifierCV(BaggingClassifier(SVC(C=1), n_estimators=10, random_state=42))
        clf.fit(scaled_train_data, train_labels)

        # Make predictions and reject option
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count += 1
        print('\t', count, 'texts left unattributed')

        # Save output data
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        print('\t', 'answers saved to file', 'answers-' + problem + '.json')

    print('elapsed time:', time.time() - start_time)
    

def baseline_boosting_kernel(path, outpath, n=4, ft=5, pt=0.025):
    """
    Runs a boosting classifier with SVM base estimator on a set of texts, and saves the predicted authorship for each text to a file.
    
    Args:
        path (str): The path where the input texts are located.
        outpath (str): The path where the output file will be saved.
        n (int): The length of the character n-grams to extract.
        ft (int): The minimum frequency threshold for the character n-grams.
        pt (float): The threshold for rejecting predicted authorship if the difference between the two highest probabilities is less than `pt`.
    """
    start_time = time.time()
    
    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    
    # Iterate over each problem in the collection
    for index, problem in enumerate(problems[0:5]):
        print(problem)
        
        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        
        # Building training set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files_pos(path + os.sep + problem, candidate))
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs, n, ft)
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(n, n),
            lowercase=False,
            vocabulary=vocabulary,
            token_pattern=r"(?u)\b\w+\b|sentiment_[-+]?\d*\.\d{2}|[A-Z]+"
        )
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        
        # Building test set
        test_docs = read_files_pos(path + os.sep + problem, unk_folder)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')
        
        # Scale the data
        max_abs_scaler = MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)

        # Create a boosting classifier with SVM base estimator
        clf = CalibratedClassifierCV(
            AdaBoostClassifier(SVC(C=1, kernel='linear'), algorithm='SAMME', n_estimators=50, random_state=42)
        )
        clf.fit(scaled_train_data, train_labels)

        # Make predictions and reject option
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count += 1
        print('\t', count, 'texts left unattributed')

        # Save output data
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
            
        with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        print('\t', 'answers saved to file', 'answers-' + problem + '.json')

    print('elapsed time:', time.time() - start_time)
    
    
def KNN_algo(path, outpath, n=4, ft=5, pt=0.025):
    """
    Runs the k-Nearest Neighbors (KNN) algorithm on a set of texts and saves the predicted authorship for each text to a file.

    Args:
        path (str): The path where the input texts are located.
        outpath (str): The path where the output file will be saved.
        n (int): The length of the character n-grams to extract.
        ft (int): The minimum frequency threshold for the character n-grams.
        pt (float): The threshold for rejecting predicted authorship if the difference between the two highest probabilities is less than `pt`.
    """
    start_time = time.time()

    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    # Iterate over each problem in the collection
    for index, problem in enumerate(problems[0:5]):
        print(problem)

        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])

        # Building training set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(path + os.sep + problem, candidate))
        train_docs = remove_punctuations_from_docs(train_docs)
        train_docs = stem_documents(train_docs)
        train_docs = remove_stopwords_from_docs(train_docs, stop_words)
        train_docs = case_folding(train_docs)
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs, n, ft)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False, vocabulary=vocabulary)
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))

        # Building test set
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_docs = remove_punctuations_from_docs(test_docs)
        test_docs = stem_documents(test_docs)
        test_docs = remove_stopwords_from_docs(test_docs, stop_words)
        test_docs = case_folding(test_docs)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')

        # Parameters
        k = 6  # Number of nearest neighbors
        pt = 0.05  # Probability threshold for rejection

        # Scaling data
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)

        # Training classifier
        clf = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=k))
        clf.fit(scaled_train_data, train_labels)

        # Making predictions
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)

        # Reject option (used in open-set cases)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count += 1
        print('\t', count, 'texts left unattributed')

        # Saving output data
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        print('\t', 'answers saved to file', 'answers-' + problem + '.json')

    print('elapsed time:', time.time() - start_time)


def random_forest(path, outpath, n=4, ft=5, pt=0.01):
    """
    Runs the random forest algorithm on a set of texts for authorship attribution.

    Args:
        path (str): The path where the input texts are located.
        outpath (str): The path where the output file will be saved.
        n (int, optional): The length of the character n-grams to extract. Default is 4.
        ft (int, optional): The minimum frequency threshold for the character n-grams. Default is 5.
        pt (float, optional): The threshold for rejecting predicted authorship. Default is 0.01.

    Returns:
        None

    """
    start_time = time.time()
    # Reading information about the collection
    infocollection = path + os.sep + 'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    for index, problem in enumerate(problems[0:5]):
        print(problem)
        # Reading information about the problem
        infoproblem = path + os.sep + problem + os.sep + 'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        train_docs = []
        for candidate in candidates:
            train_docs.extend(read_files(path + os.sep + problem, candidate))
        train_docs = remove_punctuations_from_docs(train_docs)
        train_docs = stem_documents(train_docs)
        train_docs = remove_stopwords_from_docs(train_docs, stop_words)
        train_texts = [text for i, (text, label) in enumerate(train_docs)]
        train_labels = [label for i, (text, label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs, n, ft)
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), lowercase=False, vocabulary=vocabulary)
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i, v in enumerate(train_texts):
            train_data[i] = train_data[i] / len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        # Building test set
        test_docs = read_files(path + os.sep + problem, unk_folder)
        test_docs = remove_punctuations_from_docs(test_docs)
        test_docs = stem_documents(test_docs)
        test_docs = remove_stopwords_from_docs(test_docs, stop_words)
        test_texts = [text for i, (text, label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i, v in enumerate(test_texts):
            test_data[i] = test_data[i] / len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')

        # Random Forest
        # Scale the data
        max_abs_scaler = MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)

        n_est = 100
        max_depth = 20
                # Create a Random Forest classifier
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
        clf = CalibratedClassifierCV(rf)

        # Train the classifier
        clf.fit(scaled_train_data, train_labels)

        # Make predictions and reject option
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)
        count = 0
        for i, p in enumerate(predictions):
            sproba = sorted(proba[i], reverse=True)
            if sproba[0] - sproba[1] < pt:
                predictions[i] = u'<UNK>'
                count += 1
        print('\t', count, 'texts left unattributed')

        # Save output data
        out_data = []
        unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
        pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
        for i, v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
            json.dump(out_data, f, indent=4)

        print('\t', 'answers saved to file', 'answers-' + problem + '.json')
    print('elapsed time:', time.time() - start_time)


def main():
    """
    Main function to execute the baseline authorship attribution method.
    """
    # Argument parsing
    parser = argparse.ArgumentParser(description='PAN-19 Baseline Authorship Attribution Method')
    parser.add_argument('-i', type=str, help='Path to the main folder of a collection of attribution problems')
    parser.add_argument('-o', type=str, help='Path to an output folder')
    parser.add_argument('-n', type=int, default=3, help='n-gram order (default=3)')
    parser.add_argument('-ft', type=int, default=5, help='frequency threshold (default=5)')
    parser.add_argument('-pt', type=float, default=0.1, help='probability threshold for the reject option (default=0.1)')
    args = parser.parse_args()

    # # Check if input and output folders are provided
    # if not args.i:
    #     print('ERROR: The input folder is required')
    #     parser.exit(1)
    # if not args.o:
    #     print('ERROR: The output folder is required')
    #     parser.exit(1)

    # Execute baseline_boosting_kernel method
    # baseline_boosting_kernel(args.i, args.o, args.n, args.ft, args.pt)
    baseline_boosting_kernel(os.path.join('.', 'pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'), '.\output')
if __name__ == '__main__':
    main()