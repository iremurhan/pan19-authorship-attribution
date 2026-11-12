# -*- coding: utf-8 -*-

"""
 A baseline authorship attribution method 
 based on a character n-gram representation
 and a linear SVM classifier.
 It has a reject option to leave documents unattributed
 (when the probabilities of the two most likely training classes are too close)
 
 Questions/comments: stamatatos@aegean.gr

 It can be applied to datasets of PAN-19 cross-domain authorship attribution task
 See details here: http://pan.webis.de/clef19/pan19-web/author-identification.html
 Dependencies:
 - Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
 - scikit-learn


 Usage from command line: 
    > python pan19-cdaa-baseline.py -i EVALUATION-DIRECTORY -o OUTPUT-DIRECTORY [-n N-GRAM-ORDER] [-ft FREQUENCY-THRESHOLD] [-pt PROBABILITY-THRESHOLD]
 EVALUATION-DIRECTORY (str) is the main folder of a PAN-19 collection of attribution problems
 OUTPUT-DIRECTORY (str) is an existing folder where the predictions are saved in the PAN-19 format
 Optional parameters of the model:
   N-GRAM-ORDER (int) is the length of character n-grams (default=3)
   FREQUENCY-THRESHOLD (int) is the cutoff threshold used to filter out rare n-grams (default=5)
   PROBABILITY-THRESHOLD (float) is the threshold for the reject option assigning test documents to the <UNK> class (default=0.1)
                                 Let P1 and P2 be the two maximum probabilities of training classes for a test document. If P1-P2<pt then the test document is assigned to the <UNK> class.
   
 Example:
     > python pan19-cdaa-baseline.py -i "mydata/pan19-cdaa-development-corpus" -o "mydata/pan19-answers"
"""

from __future__ import print_function
import os
import glob
import json
import argparse
import time
import codecs
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
import nltk
from textblob import TextBlob
from sklearn.ensemble import AdaBoostClassifier

def represent_text(text, n):
    # Extracts all character 'n'-grams and POS 'n'-grams from a 'text'
    tokens = []
    if n > 0:
        tokens.extend([text[i:i+n] for i in range(len(text)-n+1)])
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return frequency

def read_files(path, label):
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
    # Extracts all character 'n'-grams, POS 'n'-grams, and sentiment occurring at least 'ft' times in a set of 'texts'
    occurrences = defaultdict(int)
    for (text, label) in texts:
        text_occurrences = represent_text(text, n)
        for ngram in text_occurrences:
            if ngram in occurrences:
                occurrences[ngram] += text_occurrences[ngram]
            else:
                occurrences[ngram] = text_occurrences[ngram]
    vocabulary = []
    for i in occurrences.keys():
        if occurrences[i] >= ft:
            vocabulary.append(i)
    return vocabulary

def get_pos_tags(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    pos_tagged_text = ' '.join([tag for word, tag in pos_tags])
    return pos_tagged_text


def get_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return f"sentiment_{sentiment_score:.2f}"

def baseline_adaboost(path,outpath,n=4,ft=5,pt=0.01):
    start_time = time.time()
    # Reading information about the collection
    infocollection = path+os.sep+'collection-info.json'
    problems = []
    language = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])
    for index,problem in enumerate(problems[0:5]):
        print(problem)
        # Reading information about the problem
        infoproblem = path+os.sep+problem+os.sep+'problem-info.json'
        candidates = []
        with open(infoproblem, 'r') as f:
            fj = json.load(f)
            unk_folder = fj['unknown-folder']
            for attrib in fj['candidate-authors']:
                candidates.append(attrib['author-name'])
        # Building training set
        train_docs=[]
        for candidate in candidates:
            train_docs.extend(read_files(path+os.sep+problem,candidate))
        #train_docs = case_folding(train_docs)
        #train_docs = remove_punctuations_from_docs(train_docs)
        #train_docs = stem_documents(train_docs)
        #train_docs = remove_stopwords_from_docs(train_docs, stop_words)
        train_texts = [text for i,(text,label) in enumerate(train_docs)]
        train_labels = [label for i,(text,label) in enumerate(train_docs)]
        vocabulary = extract_vocabulary(train_docs,n,ft)
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=(n, n),
            lowercase=False,
            vocabulary=vocabulary,
            token_pattern=r"(?u)\b\w+\b|sentiment_[-+]?\d*\.\d{2}|[A-Z]+"
        )
        train_data = vectorizer.fit_transform(train_texts)
        train_data = train_data.astype(float)
        for i,v in enumerate(train_texts):
            train_data[i]=train_data[i]/len(train_texts[i])
        print('\t', 'language: ', language[index])
        print('\t', len(candidates), 'candidate authors')
        print('\t', len(train_texts), 'known texts')
        print('\t', 'vocabulary size:', len(vocabulary))
        # Building test set
        test_docs=read_files(path+os.sep+problem,unk_folder)
        #test_docs = case_folding(test_docs)
        #test_docs = remove_punctuations_from_docs(test_docs)
        #test_docs = stem_documents(test_docs)
        #test_docs = remove_stopwords_from_docs(test_docs, stop_words)
        test_texts = [text for i,(text,label) in enumerate(test_docs)]
        test_data = vectorizer.transform(test_texts)
        test_data = test_data.astype(float)
        for i,v in enumerate(test_texts):
            test_data[i]=test_data[i]/len(test_texts[i])
        print('\t', len(test_texts), 'unknown texts')
        # Applying SVM
        max_abs_scaler = preprocessing.MaxAbsScaler()
        scaled_train_data = max_abs_scaler.fit_transform(train_data)
        scaled_test_data = max_abs_scaler.transform(test_data)
        # Create a boosting classifier with SVM base estimator
        clf = CalibratedClassifierCV(AdaBoostClassifier(SVC(C=10, kernel='linear'), algorithm='SAMME', n_estimators=100, random_state=42))
        clf.fit(scaled_train_data, train_labels)

        # Make predictions and reject option
        predictions = clf.predict(scaled_test_data)
        proba = clf.predict_proba(scaled_test_data)
        count = 0
        for i,p in enumerate(predictions):
            sproba=sorted(proba[i],reverse=True)
            if sproba[0]-sproba[1]<pt:
                predictions[i]=u'<UNK>'
                count=count+1
        print('\t',count,'texts left unattributed')
        # Saving output data
        out_data=[]
        unk_filelist = glob.glob(path+os.sep+problem+os.sep+unk_folder+os.sep+'*.txt')
        pathlen=len(path+os.sep+problem+os.sep+unk_folder+os.sep)
        for i,v in enumerate(predictions):
            out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        with open(outpath+os.sep+'answers-'+problem+'.json', 'w') as f:
            json.dump(out_data, f, indent=4)
        print('\t', 'answers saved to file','answers-'+problem+'.json')
    print('elapsed time:', time.time() - start_time)
    
def main():
    # parser = argparse.ArgumentParser()
    # parser = argparse.ArgumentParser(
    #     description='PAN-19 Baseline Authorship Attribution Method')
    # parser.add_argument(
    #     '-i', type=str, help='Path to the main folder of a collection of attribution problems')
    # parser.add_argument('-o', type=str, help='Path to an output folder')
    # parser.add_argument('-n', type=int, default=3,
    #                     help='n-gram order (default=3)')
    # parser.add_argument('-ft', type=int, default=5,
    #                     help='frequency threshold (default=5)')
    # parser.add_argument('-pt', type=float, default=0.1,
    #                     help='probability threshold for the reject option (default=0.1')
    # args = parser.parse_args()
    # if not args.i:
    #     print('ERROR: The input folder is required')
    #     parser.exit(1)
    # if not args.o:
    #     print('ERROR: The output folder is required')
    #     parser.exit(1)

    # baseline(args.i, args.o, args.n, args.ft, args.pt)
    i = 'C:\\Users\\user\\Desktop\\school\\terms\\spring23\\cmpe493\\Assignments\\TermProject\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    o = 'C:\\Users\\user\\Desktop\\school\\terms\\spring23\\cmpe493\\Assignments\\TermProject\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    baseline_adaboost(i, o, n=4, pt=0.015)

if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Evaluation script for the Cross-Domain Authorship Attribution task @PAN2019.
We use the F1 metric (macro-average) as implemented in scikit-learn:
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
We include the following ad hoc rules:
- If authors are predicted which were not seen during training,
  these predictions will count as false predictions ('<UNK>' class)
  and they will negatively effect performance.
- If texts are left unattributed they will assigned to the ('<UNK>'
  class) and they will negatively effect performance.
- The <UNK> class is excluded from the macro-average across classes.
- If multiple test attributions are given for a single unknown document,
  only the first one will be taken into consideration.

Dependencies:
- Python 2.7 or 3.6 (we recommend the Anaconda Python distribution)
- scikit-learn

Usage from the command line:
>>> python pan19-cdaa-evaluator.py -i COLLECTION -a ANSWERS -o OUTPUT
where
    COLLECTION is the path to the main folder of the evaluation collection
    ANSWERS is the path to the answers folder of a submitted method
    OUTPUT is the path to the folder where the results of the evaluation will be saved

Example: 
>>> python pan19-cdaa-evaluator.py -i "/mydata/pan19-cdaa-development-corpus" -a "/mydata/pan19-answers" -o "/mydata/pan19-evaluation"

# References:
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}
"""

import argparse
import os
import json
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def eval_measures(gt, pred):
    """Compute macro-averaged F1-scores, macro-averaged precision, 
    macro-averaged recall, and micro-averaged accuracy according the ad hoc
    rules discussed at the top of this file.
    Parameters
    ----------
    gt : dict
        Ground truth, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    pred : dict
        Predicted attribution, where keys indicate text file names
        (e.g. `unknown00002.txt`), and values represent
        author labels (e.g. `candidate00003`)
    Returns
    -------
    f1 : float
        Macro-averaged F1-score
    precision : float
        Macro-averaged precision
    recall : float
        Macro-averaged recall
    accuracy : float
        Micro-averaged F1-score
    """

    actual_authors = list(gt.values())
    encoder = LabelEncoder().fit(['<UNK>'] + actual_authors)

    text_ids, gold_authors, silver_authors = [], [], []
    for text_id in sorted(gt):
        text_ids.append(text_id)
        gold_authors.append(gt[text_id])
        try:
            silver_authors.append(pred[text_id])
        except KeyError:
            # missing attributions get <UNK>:
            silver_authors.append('<UNK>')

    assert len(text_ids) == len(gold_authors)
    assert len(text_ids) == len(silver_authors)

    # replace non-existent silver authors with '<UNK>':
    silver_authors = [a if a in encoder.classes_ else '<UNK>' 
                      for a in silver_authors]

    gold_author_ints = encoder.transform(gold_authors)
    silver_author_ints = encoder.transform(silver_authors)

    # get F1 for individual classes (and suppress warnings):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        labels=list(set(gold_author_ints))
        # Exclude the <UNK> class
        for x in labels:
            if x == 0:
                labels.remove(x)
        f1 = f1_score(gold_author_ints,
                  silver_author_ints,
                  labels,
                  average='macro')
        precision = precision_score(gold_author_ints,
                  silver_author_ints,
                  labels,
                  average='macro')
        recall = recall_score(gold_author_ints,
                  silver_author_ints,
                  labels,
                  average='macro')
        accuracy = accuracy_score(gold_author_ints,
                  silver_author_ints)

    return f1,precision,recall

def evaluate(ground_truth_file,predictions_file):
    # Calculates evaluation measures for a single attribution problem
    gt = {}
    with open(ground_truth_file, 'r') as f:
        for attrib in json.load(f)['ground_truth']:
            gt[attrib['unknown-text']] = attrib['true-author']

    pred = {}
    with open(predictions_file, 'r') as f:
        for attrib in json.load(f):
            if attrib['unknown-text'] not in pred:
                pred[attrib['unknown-text']] = attrib['predicted-author']
    f1,precision,recall =  eval_measures(gt,pred)
    return round(f1,3), round(precision,3), round(recall,3)

def evaluate_all(path_collection,path_answers,path_out):
    # Calculates evaluation measures for a PAN-18 collection of attribution problems
    infocollection = path_collection+os.sep+'collection-info.json'
    problems = []
    data = []
    with open(infocollection, 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
    scores=[];
    for problem in problems:
        f1,precision,recall=evaluate(path_collection+os.sep+problem+os.sep+'ground-truth.json',path_answers+os.sep+'answers-'+problem+'.json')
        scores.append(f1)
        data.append({'problem-name': problem, 'macro-f1': round(f1,3), 'macro-precision': round(precision,3), 'macro-recall': round(recall,3)})
        print(str(problem),'Macro-F1:',round(f1,3))
    test_score=(scores[0] + scores[2] + scores[4])/3
    # Saving data to output files (out.json and evaluation.prototext)
    with open(path_out+os.sep+'out.json', 'w') as f:
        json.dump({'problems': data, 'test_score': round(test_score,3)}, f, indent=4, sort_keys=True)
    print('Test score:', round(test_score,3))
    prototext='measure {\n key: "mean macro-f1"\n value: "'+str(round(test_score,3))+'"\n}\n'
    with open(path_out+os.sep+'evaluation.prototext', 'w') as f:
        f.write(prototext)
        
def main():
    # parser = argparse.ArgumentParser(description='Evaluation script AA@PAN2019')
    # parser.add_argument('-i', type=str,
    #                     help='Path to evaluation collection')
    # parser.add_argument('-a', type=str,
    #                     help='Path to answers folder')
    # parser.add_argument('-o', type=str, 
    #                     help='Path to output files')
    # args = parser.parse_args()
    # if not args.i:
    #     print('ERROR: The collection path is required')
    #     parser.exit(1)
    # if not args.a:
    #     print('ERROR: The answers folder is required')
    #     parser.exit(1)
    # if not args.o:
    #     print('ERROR: The output path is required')
    #     parser.exit(1)

    # evaluate_all(args.i,args.a,args.o)
    i = 'C:\\Users\\user\\Desktop\\school\\terms\\spring23\\cmpe493\\Assignments\\TermProject\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    a = 'C:\\Users\\user\\Desktop\\school\\terms\\spring23\\cmpe493\\Assignments\\TermProject\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    o = 'C:\\Users\\user\\Desktop\\school\\terms\\spring23\\cmpe493\\Assignments\\TermProject\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23\\pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23'
    evaluate_all(i, a, o)
if __name__ == '__main__':
    main()
