"""TF-IDF + LinearSVC baselines (the 'yardstick' of Lab 3a)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


def fit_tfidf(train_texts, train_y, kind: str = "word"):
    if kind == "word":      # exactly the Lab 3a baseline
        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    else:                   # stronger char n-gram variant (extra, to be fair to the baseline)
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=200000, sublinear_tf=True)
    X = vec.fit_transform(train_texts)
    clf = LinearSVC().fit(X, train_y)
    return vec, clf


def predict_tfidf(vec, clf, texts):
    return clf.predict(vec.transform(texts))
