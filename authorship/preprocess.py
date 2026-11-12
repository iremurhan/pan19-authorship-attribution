"""Optional text preprocessing and POS / sentiment augmentation."""
import string

STOP_WORDS = frozenset(
    "a all an and any are as be been but by few for have he her here him his how i in is it its "
    "many me my none of on or our she some the their them there they that this us was what when "
    "where which who why will with you your".split())

_PUNCTUATION = set(string.punctuation + "–")


def remove_punctuation(text):
    return "".join(ch for ch in text if ch not in _PUNCTUATION)


def remove_stopwords(text):
    return " ".join(w for w in text.split() if w.lower() not in STOP_WORDS)


def stem(text):
    from nltk.stem import PorterStemmer
    porter = PorterStemmer()
    return " ".join(porter.stem(w) for w in text.split())


def case_fold(text):
    return text.lower()


# Applied in this order; stop words are removed before stemming so that
# stemmed forms (e.g. "was" -> "wa") cannot slip past the stop-word list.
STEPS = {
    "punctuation": remove_punctuation,
    "stopwords": remove_stopwords,
    "stem": stem,
    "casefold": case_fold,
}


def preprocess(text, steps):
    for name in STEPS:
        if name in steps:
            text = STEPS[name](text)
    return text


def augment_pos_sentiment(text):
    """Append the POS-tag sequence and a polarity token to the text.

    Character n-grams are later taken over the whole string, so the tags
    contribute n-grams such as "NN V" alongside the raw text.
    """
    import nltk
    from textblob import TextBlob

    tags = " ".join(tag for _, tag in nltk.pos_tag(nltk.word_tokenize(text)))
    polarity = TextBlob(text).sentiment.polarity
    return f"{text} {tags} sentiment_{polarity:.2f}"
