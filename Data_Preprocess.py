import numpy as np
import spacy
import pickle
import spacy.cli
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from warnings import simplefilter
from scipy.sparse import hstack,save_npz
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) #quality of life change
pd.set_option('display.max_columns', None)


class BoW_parser:
    def __init__(self, return_df=False):
        self.nlp = spacy.load('en_core_web_sm')
        self.return_df = return_df
        self.vectorizer = None

    def parse(self, raw_data, k=10):
        self.processed = []
        self.processed_tokens = []

        for doc in self.nlp.pipe(raw_data, batch_size=1000):
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not any((t.is_stop, t.is_punct, t.is_space, t.like_num))
            ]
            self.processed_tokens.append(tokens)  #raw token lists for coherence model
            self.processed.append(" ".join(tokens))  #joined strings for vectorizer

        processed = self.processed
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                binary=True,
                min_df=k,
                max_df=0.85,
                max_features=20000
            ) #Removes extremely common tokens like in BoW
            X = self.vectorizer.fit_transform(processed)
        else:
            X = self.vectorizer.transform(processed)

        if self.return_df:
            return pd.DataFrame(
                X.toarray(),
                columns=self.vectorizer.get_feature_names_out(),
                index=raw_data.index
            )

        return X

    def save_vectorizer(self, path="vectorizer.pkl"): #Used later in the demo
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Vectorizer saved to {path}")

    def load_vectorizer(self, path="vectorizer.pkl"):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        print(f"Vectorizer loaded from {path}")


class TF_IDF_parser(BoW_parser):

    def __init__(self, return_df: bool = False):
        super().__init__(return_df)

    def parse(self, raw_data, k=10):
        self.processed = []
        self.processed_tokens = []

        for doc in self.nlp.pipe(raw_data, batch_size=1000):
            tokens = [
                t.lemma_.lower()
                for t in doc
                if not any((t.is_stop, t.is_punct, t.is_space, t.like_num))
            ]
            self.processed_tokens.append(tokens)  #raw token lists for coherence model
            self.processed.append(" ".join(tokens))  #joined strings for vectorizer

        processed = self.processed
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=20000)
            X = self.vectorizer.fit_transform(processed)
        else:
            X = self.vectorizer.transform(processed)

        if self.return_df:
            return pd.DataFrame(
                X.toarray(),
                columns=self.vectorizer.get_feature_names_out(),
                index=raw_data.index
            )

        return X

if __name__ == "__main__":

    def build_label(df): #Encodes and aligns labels with their corrosponding position in the matrix
        return df["class_label"].map({True: 1, False: 0}).values.reshape(-1, 1)

    social_media = pd.read_csv('social-media-release.csv').set_index('id').dropna()

    #Split data as immediate first step to prevent any data leaks
    train_data, test_data = train_test_split(
        social_media,
        test_size=0.3,
        random_state=1,
        stratify=social_media['class_label']
    )
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")

    parsers = [
        (BoW_parser(), "BoW", "vectorizer_bow.pkl", {"k": 10}),
        (TF_IDF_parser(), "TF_IDF", "vectorizer_tfidf.pkl", {})
    ]

    for parser, name, vec_path, k in parsers:
        print(f"\nParsing {name} training data...")

        X_train_text = parser.parse(train_data['post'], **k)
        parser.save_vectorizer(vec_path)

        print(f"Parsing {name} test data...")
        X_test_text = parser.parse(test_data['post'])

        train_matrix = hstack((X_train_text, build_label(train_data)))
        test_matrix  = hstack((X_test_text, build_label(test_data)))

        print(f"Saving {name}...") #Vectorizers are saved to preserve vocabulary for demo
        save_npz(f"{name}_train.npz", train_matrix)
        save_npz(f"{name}_test.npz",  test_matrix)
        print(f"Saved {name}_train.npz and {name}_test.npz")