from Data_Preprocess import BoW_parser, TF_IDF_parser, pd, hstack, np
from sklearn.metrics import classification_report, confusion_matrix
from keras_preprocessing.sequence import pad_sequences
from scipy.sparse import csr_matrix
from keras.models import load_model
from Topic_Discovery import topic_label_analysis, compare_topics, print_top_words, extract_corpus, topic_overlap
import pickle

def pre_process(raw_data): #prepare data for NN
    parser = BoW_parser()
    parser.load_vectorizer("vectorizer_bow.pkl")

    posts = raw_data['post']
    class_label = raw_data["class_label"].map({True: 1, False: 0}).values.reshape(-1, 1)

    processed_posts = parser.parse(posts)
    return hstack((processed_posts, csr_matrix(class_label)))

#Load functions had to be adjusted from the NN file to take vectors as input
def load_data(sparse_vector):
    data = sparse_vector.tocsr()

    X = data[:, :-1]
    y = data[:, -1].toarray().ravel()

    return X, y

#Does not need to return max len or find vocab size
def load_for_CNN(sparse_vector, percentile=95):
    sparse = sparse_vector.tocsr()

    X_text = sparse[:, :-1]
    y = np.array(sparse[:, -1].toarray()).flatten().astype(int)

    X_dense = X_text.toarray()
    sequences = [np.where(row > 0)[0] for row in X_dense]

    lengths = [len(seq) for seq in sequences]
    max_len = int(np.percentile(lengths, percentile))

    X_padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

    return X_padded, y


def evaluate(y_test, y_pred, model_name):
    print("\n=========================================================")
    print(f"{model_name} Results")
    print("=========================================================")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    raw_data_path = input("Enter the path of raw data: ")

    try:
        raw_data = pd.read_csv(raw_data_path).dropna(subset=['post', 'news_headline', 'class_label']).reset_index(drop=True)
        processed = pre_process(raw_data)
    except FileNotFoundError:
        print("Raw data not found. Please check the path.")
        exit()

    while True:
        program = input("Select Program:\n 1. NN Demo\n 2. Topic Discovery Demo\nChoice: ")

        if program == '1':
            model_choice = input("Select Model [MLP, CNN]: ").upper()

            if model_choice == 'MLP':
                try:
                    with open("best_mlp.pkl", "rb") as f:
                        mlp = pickle.load(f)
                except FileNotFoundError:
                    print("No saved MLP found. Please run NN.py first.")
                    exit()

                X, y = load_data(processed)
                y_pred = mlp.predict(X)
                evaluate(y, y_pred, "MLP")

            elif model_choice == 'CNN':
                try:
                    cnn = load_model("best_cnn.keras")

                except FileNotFoundError:
                    print("No saved CNN found. Please run NN.py first.")
                    exit()

                X, y = load_for_CNN(processed)
                y_pred = (cnn.predict(X) > 0.5).astype(int).flatten()
                evaluate(y, y_pred, "CNN")

            else:
                print("Please enter MLP or CNN.")
                exit()

        elif program == '2': #LDA Topic Discovery
            source = input("Analyse [posts, headlines, both]: ").lower()

            try:
                with open("lda_posts.pkl", "rb") as f:
                    lda_posts = pickle.load(f)
                with open("lda_headlines.pkl", "rb") as f:
                    lda_headlines = pickle.load(f)
            except FileNotFoundError:
                print("No saved LDA models found. Please run Topic_Discovery.py first.")
                exit()

            if source in ('posts', 'both'):
                parser_posts = TF_IDF_parser()
                parser_posts.load_vectorizer("vectorizer_posts.pkl") #Use saved vectoriser to keep vocab
                corpus_posts, vect_posts = extract_corpus(parser_posts, raw_data['post'])
                n_topics = lda_posts.n_components

                print("\n--- LDA Topics (Posts) ---")
                for t in range(n_topics):
                    print_top_words(lda_posts, vect_posts, t, 10)

                topic_label_analysis(lda_posts, corpus_posts, raw_data, "Posts")

            if source in ('headlines', 'both'):
                parser_headlines = TF_IDF_parser()
                parser_headlines.load_vectorizer("vectorizer_headlines.pkl")
                corpus_headlines, vect_headlines = extract_corpus(parser_headlines, raw_data['news_headline'])
                n_topics = lda_headlines.n_components

                print("\n--- LDA Topics (Headlines) ---")
                for t in range(n_topics):
                    print_top_words(lda_headlines, vect_headlines, t, 10)

                topic_label_analysis(lda_headlines, corpus_headlines, raw_data, "Headlines")

            if source == 'both':
                print("\n--- Topic Alignment (Posts vs Headlines) ---")
                compare_topics(lda_posts, lda_headlines, vect_posts, vect_headlines, n_topics)
                print("\n--- Topic Overlap (Posts vs Headlines) ---")
                topic_overlap(lda_posts, lda_headlines, vect_posts, vect_headlines, n_topics)

            if source not in ('posts', 'headlines', 'both'):
                print("Please enter posts, headlines, or both.")
                exit()

        else:
            print("Please enter 1 or 2.")
            exit()