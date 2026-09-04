import numpy as np
import pickle
import itertools
import json
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout, Embedding, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam


def load_data(train, test):
    print("Loading MLP data...")
    train = train.tocsr()
    test  = test.tocsr()

    X_train = train[:, :-1]
    y_train = train[:, -1].toarray().ravel()

    X_test = test[:, :-1]
    y_test = test[:, -1].toarray().ravel()

    #Scale
    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def load_for_CNN(train, test, percentile=95): #Requires conversion into dense sequence matrix
    print("Loading CNN data...")
    train_sparse = train.tocsr()
    test_sparse  = test.tocsr()

    def to_sequences(sparse): #create flat sequences as a series of np arrays instead of a sparse matrix
        X_text = sparse[:, :-1]
        y = np.array(sparse[:, -1].toarray()).flatten().astype(int)
        X_dense = X_text.toarray()
        sequences = [np.where(row > 0)[0] for row in X_dense]
        return sequences, y, X_text.shape[1] + 1

    train_seqs, y_train, vocab_size = to_sequences(train_sparse) #Vocab is derived from training set
    test_seqs, y_test, _ = to_sequences(test_sparse)


    #calculate max_len from training data
    lengths = [len(seq) for seq in train_seqs]
    max_len = int(np.percentile(lengths, percentile)) #sequence length capped at 95th percentile of document length
    print(f"Mean: {np.mean(lengths):.0f}, Median: {np.median(lengths):.0f}, "
          f"{percentile}th percentile (max_len): {max_len}, Max: {np.max(lengths):.0f}")

    X_train = pad_sequences(train_seqs, maxlen=max_len, padding="post", truncating="post") #Ensure all sequences are the same length
    X_test = pad_sequences(test_seqs,  maxlen=max_len, padding="post", truncating="post")

    return X_train, X_test, y_train, y_test, vocab_size, max_len


def tune_and_save_MLP(X_train, X_test, y_train, y_test):
    param_grid = {
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "learning_rate_init": [0.001, 0.01],
        "alpha": [0.0001, 0.001]
    }

    #Save results and best models for evaluation
    best_accuracy = 0
    best_params = None
    best_model = None
    results = []

    #Grid search
    for hidden, lr, alpha in itertools.product(
        param_grid["hidden_layer_sizes"],
        param_grid["learning_rate_init"],
        param_grid["alpha"]
    ):
        print(f"Trying MLP: layers={hidden}, lr={lr}, alpha={alpha}")
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden,
            learning_rate_init=lr,
            alpha=alpha,
            max_iter=10000,
            activation='tanh',
            solver='sgd',
            verbose=False,
            random_state=1
        )
        mlp.fit(X_train, y_train.ravel())
        accuracy = mlp.score(X_test, y_test)
        print(f"  Accuracy: {accuracy:.4f}")
        results.append({"hidden": str(hidden), "lr": lr, "alpha": alpha, "accuracy": accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"hidden": hidden, "lr": lr, "alpha": alpha}
            best_model = mlp

    print(f"\nBest MLP — Accuracy: {best_accuracy:.4f}, Params: {best_params}")
    print(confusion_matrix(y_test, best_model.predict(X_test)))
    print(classification_report(y_test, best_model.predict(X_test)))

    #Save best model for demo and results
    with open("best_mlp.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("mlp_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved best MLP to best_mlp.pkl")
    return best_model, best_params


def build_CNN(vocab_size, max_len, filters, kernel_size, learning_rate):
    #Construct CNN structure with ADAM optimizer
    model = Sequential([
        Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=max_len),
        Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def tune_and_save_CNN(X_train, X_test, y_train, y_test, vocab_size, max_len):
    param_grid = {
        "filters": [64, 128, 256],
        "kernel_size": [3, 5, 7],
        "learning_rate": [0.001, 0.0001]
    }

    #save best for evaluation
    best_accuracy = 0
    best_params = None
    best_model = None
    results = []

    #early stopping in learning to prevent overfitting of models
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    for filters, kernel_size, lr in itertools.product(
        param_grid["filters"],
        param_grid["kernel_size"],
        param_grid["learning_rate"]
    ):
        print(f"Trying CNN: filters={filters}, kernel={kernel_size}, lr={lr}")
        model = build_CNN(vocab_size, max_len, filters, kernel_size, lr)

        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"  Accuracy: {accuracy:.4f}")
        results.append({"filters": filters, "kernel_size": kernel_size, "lr": lr, "accuracy": accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {"filters": filters, "kernel_size": kernel_size, "lr": lr}
            best_model = model

    print(f"\nBest CNN — Accuracy: {best_accuracy:.4f}, Params: {best_params}")

    best_model.save("best_cnn.keras")
    with open("cnn_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved best CNN to best_cnn.keras")
    return best_model, best_params


if __name__ == '__main__':
    #Load saved files from Data_Preprocess.py
    train_sparse = load_npz('BoW_train.npz')
    test_sparse = load_npz('BoW_test.npz')

    #MLP tuning
    X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = load_data(
        train_sparse, test_sparse
    )

    best_mlp, mlp_params = tune_and_save_MLP(X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp)

    #CNN tuning
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, vocab_size, max_len = load_for_CNN(
        train_sparse, test_sparse
    )

    best_cnn, cnn_params = tune_and_save_CNN(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, vocab_size, max_len)