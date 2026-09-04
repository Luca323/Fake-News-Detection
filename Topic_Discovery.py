import pandas as pd
import matplotlib.pyplot as plt
import pickle
from Data_Preprocess import TF_IDF_parser, BoW_parser, np
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
pd.set_option('display.max_columns', None)
np.set_printoptions(linewidth=200) #for debugging

def extract_corpus(parser, raw_data):
    raw_data = raw_data.astype(str)
    corpus = parser.parse(raw_data)
    vect = parser.vectorizer
    return corpus, vect

def build_LDA(corpus, n_topics):
    lda = LatentDirichletAllocation(n_components=n_topics, learning_method='online',
                                    max_iter=50, random_state=SEED)

    lda.fit(corpus)

    return lda

def tune_topic_num(corpus, tokenized_docs, dictionary, vect, k_range, seed):
    perplexities = []
    coherences = []

    for k in k_range:
        print(f"Fitting LDA with {k} topics...")
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method='online',
            max_iter=10,
            random_state=seed
        )
        lda.fit(corpus)

        perplexities.append(lda.perplexity(corpus))

        #convert top 10 topic LDA numerical topics into human-readable words
        topics = [
            [vect.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
            for topic in lda.components_
        ]

        #Coherence extraction via gensim
        coherence = CoherenceModel(
            model=None,
            topics=topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v'
        ).get_coherence()

        coherences.append(coherence)

        print(f"  Perplexity: {perplexities[-1]:.2f}, Coherence: {coherence:.4f}")

    return perplexities, coherences

def topic_label_analysis(lda, corpus, raw_data, source_name):
    doc_topics = lda.transform(corpus)
    dominant_topic = doc_topics.argmax(axis=1)
    raw_data = raw_data.copy()
    raw_data['dominant_topic'] = dominant_topic

    print(f"\nTopic distribution by class label ({source_name}):")
    print(raw_data.groupby(['dominant_topic', 'class_label']).size().unstack(fill_value=0))

    #percentage of each topic is true vs false news
    topic_label_pct = raw_data.groupby('dominant_topic')['class_label'].apply(
        lambda x: (x.map({True: 1, False: 0}).mean())
    )

    print(f"\nProportion of true news per topic ({source_name}):")
    print(topic_label_pct)

def print_top_words(lda_model, vect, topic_index, n_top_words): #print top words in a given topic
    topic_word_mat = lda_model.components_[topic_index]
    vocabulary = vect.get_feature_names_out()
    vocab_comp = zip(vocabulary, topic_word_mat)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:n_top_words]
    print(f"Topic {topic_index}:")
    for word in sorted_words:
        print(word[0], end=" ")
    print("\n")

def compare_topics(lda_posts, lda_headlines, vect_posts, vect_headlines, n_topics, n_words=10):
    #Shows a side-by-side comparison of the top words in each topic for posta and headlines
    print(f"{'POSTS':<40}                   {'HEADLINES':<40}")
    print(f"================================================================================")

    for t in range(n_topics):
        post_words = [vect_posts.get_feature_names_out()[i]
                      for i in lda_posts.components_[t].argsort()[:-n_words - 1:-1]]
        headline_words = [vect_headlines.get_feature_names_out()[i]
                          for i in lda_headlines.components_[t].argsort()[:-n_words - 1:-1]]

        print(f"Topic {t}: {' '.join(post_words):<40}        {' '.join(headline_words):<40}")

def topic_overlap(lda_posts, lda_headlines, vect_posts, vect_headlines, n_topics, n_words=100):
    #Discovers overlap between post and headline topics in %
    scores = []

    for t in range(n_topics):
        post_words = set(vect_posts.get_feature_names_out()[i]
                        for i in lda_posts.components_[t].argsort()[:-n_words-1:-1])
        headline_words = set(vect_headlines.get_feature_names_out()[i]
                            for i in lda_headlines.components_[t].argsort()[:-n_words-1:-1])
        overlap = len(post_words & headline_words) / n_words
        scores.append(overlap)
        print(f"Topic {t} overlap: {overlap:.0%} — shared words: {post_words & headline_words}")

    print(f"\nMean topic overlap: {np.mean(scores):.0%}")

def plot_comparison_metrics(results, k_range):
    #Build graph showing findings

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('LDA Topic Optimisation: BoW vs TF-IDF', fontsize=14, fontweight='bold', y=1)

    BOW_COLOR = 'blue'
    TFIDF_COLOR = 'red'

    bow_perplexities = results['BoW']['perplexities']
    bow_coherences = results['BoW']['coherences']
    tfidf_perplexities = results['TF-IDF']['perplexities']
    tfidf_coherences = results['TF-IDF']['coherences']
    k_list = list(k_range)

    best_bow_k = k_list[bow_coherences.index(max(bow_coherences))]
    best_tfidf_k = k_list[tfidf_coherences.index(max(tfidf_coherences))]

    #Coherence comparison
    ax = axes[0, 0]
    ax.plot(k_list, bow_coherences, marker='o', color=BOW_COLOR,   linewidth=2, markersize=5, label='BoW')
    ax.plot(k_list, tfidf_coherences, marker='s', color=TFIDF_COLOR, linewidth=2, markersize=5, label='TF-IDF')
    ax.axvline(best_bow_k, color=BOW_COLOR, linestyle='--', alpha=0.5)
    ax.axvline(best_tfidf_k, color=TFIDF_COLOR, linestyle='--', alpha=0.5)
    ax.scatter(best_bow_k, max(bow_coherences),   color=BOW_COLOR,   s=100, zorder=5, edgecolors='black', linewidth=1.5)
    ax.scatter(best_tfidf_k, max(tfidf_coherences), color=TFIDF_COLOR, s=100, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(f'k={best_bow_k}\n{max(bow_coherences):.4f}',   (best_bow_k,   max(bow_coherences)),   textcoords='offset points', xytext=(8, -20), fontsize=8, color=BOW_COLOR)
    ax.annotate(f'k={best_tfidf_k}\n{max(tfidf_coherences):.4f}', (best_tfidf_k, max(tfidf_coherences)), textcoords='offset points', xytext=(8, 5),   fontsize=8, color=TFIDF_COLOR)
    ax.set_title('Coherence (c_v) vs Number of Topics', fontweight='bold')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Coherence Score')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    #Perplexity comparison
    ax = axes[0, 1]
    ax.plot(k_list, bow_perplexities, marker='o', color=BOW_COLOR,   linewidth=2, markersize=5, label='BoW')
    ax.plot(k_list, tfidf_perplexities, marker='s', color=TFIDF_COLOR, linewidth=2, markersize=5, label='TF-IDF')
    ax.set_title('Perplexity vs Number of Topics', fontweight='bold')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Perplexity')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    #BoW dual axis
    ax  = axes[1, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(k_list,  bow_coherences, marker='o', color=BOW_COLOR, linewidth=2, markersize=5, label='Coherence')
    l2, = ax2.plot(k_list, bow_perplexities, marker='s', color='#4CAF50', linewidth=2, markersize=5, linestyle='--', label='Perplexity')
    ax.axvline(best_bow_k, color=BOW_COLOR, linestyle=':', alpha=0.6)
    ax.set_title('BoW: Coherence & Perplexity', fontweight='bold')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Coherence Score', color=BOW_COLOR)
    ax2.set_ylabel('Perplexity', color='#4CAF50')
    ax.tick_params(axis='y', labelcolor=BOW_COLOR)
    ax2.tick_params(axis='y', labelcolor='#4CAF50')
    ax.legend(handles=[l1, l2], loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)

    #TF-IDF dual axis
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(k_list, tfidf_coherences, marker='o', color=TFIDF_COLOR, linewidth=2, markersize=5, label='Coherence')
    l2, = ax2.plot(k_list, tfidf_perplexities, marker='s', color='#FF9800', linewidth=2, markersize=5, linestyle='--', label='Perplexity')
    ax.axvline(best_tfidf_k, color=TFIDF_COLOR, linestyle=':', alpha=0.6)
    ax.set_title('TF-IDF: Coherence & Perplexity', fontweight='bold')
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Coherence Score', color=TFIDF_COLOR)
    ax2.set_ylabel('Perplexity', color='#FF9800')
    ax.tick_params(axis='y', labelcolor=TFIDF_COLOR)
    ax2.tick_params(axis='y', labelcolor='#FF9800')
    ax.legend(handles=[l1, l2], loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.show()
    print("Saved lda_comparison.png")

if __name__ == "__main__":
    SEED = 6059
    raw_data = pd.read_csv('social-media-release.csv').dropna(subset=['post', 'news_headline', 'class_label']).reset_index(drop=True)

    parser_posts = BoW_parser()
    parser_headlines = BoW_parser()

    corpus_posts, vect_posts = extract_corpus(parser_posts, raw_data['post'])
    corpus_headlines, vect_headlines = extract_corpus(parser_headlines, raw_data['news_headline'])

    LDA_headlines = build_LDA(corpus_headlines, 14)
    LDA_posts = build_LDA(corpus_posts, 14)

    with open("lda_posts.pkl", "wb") as f: pickle.dump(LDA_posts, f)
    with open("lda_headlines.pkl", "wb") as f: pickle.dump(LDA_headlines, f)
    parser_posts.save_vectorizer("vectorizer_posts.pkl")
    parser_headlines.save_vectorizer("vectorizer_headlines.pkl")

    print("DONE")






'''
    #results = {}

    for name, (parser_posts, parser_headlines) in parsers.items():
        print("\n====================================================================")
        print(f"Running LDA with {name} representation")
        print("====================================================================")

        corpus_posts, vect_posts = extract_corpus(parser_posts, raw_data['post'])
        corpus_headlines, vect_headlines = extract_corpus(parser_headlines, raw_data['news_headline'])

        # Optimise k on headlines
        tokenized_docs = parser_headlines.processed_tokens
        dictionary = Dictionary(tokenized_docs)
        k_range = range(2, 30)

        perplexities, coherences = tune_topic_num(
            corpus_headlines, tokenized_docs, dictionary, vect_headlines, k_range, SEED
        )
        best_k = list(k_range)[coherences.index(max(coherences))]
        print(f"\nOptimal topics for {name}: {best_k} (coherence: {max(coherences):.4f})")

        #train both models with best k
        lda_posts = build_LDA(corpus_posts, best_k)
        lda_headlines = build_LDA(corpus_headlines, best_k)

        #compare
        print(f"\nTopic comparison ({name}):")
        compare_topics(lda_posts, lda_headlines, vect_posts, vect_headlines, best_k)

        #Display Topic overlap
        print(f"\nTopic overlap ({name}):")
        topic_overlap(lda_posts, lda_headlines, vect_posts, vect_headlines, best_k)

        #Link to class labels (True/fake news)
        topic_label_analysis(lda_posts, corpus_posts, raw_data, f"{name} Posts")
        topic_label_analysis(lda_headlines, corpus_headlines, raw_data, f"{name} Headlines")

        #Results
        results[name] = {
            "best_k": best_k,
            "best_coherence": max(coherences),
            "perplexities": perplexities,
            "coherences": coherences,
            "lda_posts": lda_posts,
            "lda_headlines": lda_headlines,
            "vect_posts": vect_posts,
            "vect_headlines": vect_headlines,
            "parser_posts": parser_posts,
            "parser_headlines": parser_headlines
        }

    # Summary
    print("\n=============================================================")
    print("REPRESENTATION COMPARISON SUMMARY")
    print("=============================================================")
    for name, r in results.items():
        print(f"{name}: best_k={r['best_k']}, best_coherence={r['best_coherence']:.4f}")

    # Plot comparison graphs
    plot_comparison_metrics(results, k_range)
    
    

    # Save best representation's models (best balance of perplexity & coherence)
    best_rep = 'BoW'
    print(f"\nSaving best models from {best_rep} representation...")

    with open("lda_posts.pkl", "wb") as f: pickle.dump(results[best_rep]['lda_posts'], f)
    with open("lda_headlines.pkl", "wb") as f: pickle.dump(results[best_rep]['lda_headlines'], f)
    results[best_rep]['parser_posts'].save_vectorizer("vectorizer_posts.pkl")
    results[best_rep]['parser_headlines'].save_vectorizer("vectorizer_headlines.pkl")
'''
