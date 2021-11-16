from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import svds
import pickle
# nltk.download('stopwords')

###############
# Preprocessing
###############
training_data_source = "data/brown.txt"

# Read in training data
with open(training_data_source) as f:
    sentences = f.readlines()

# Lowercase and tokenize training data
sentences = [sentence.lower() for sentence in sentences]
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# Tried removing stopwords, did not really help
# tokenized_sentences_no_stop_words = [word for sentence in sentences for word in sentence if word not in stopwords.words('english')]


##########
# Word2vec
##########
# Saves word2vec ski-gram model variations
def word2vec_models():

    ############################################################################
    # Negative Sampling Options (with default window size 5 and vector_size 100)
    # Train model types and save word vectors
    ############################################################################
    model_neg_sam_1 = Word2Vec(tokenized_sentences, sg=1, negative=1)
    model_neg_sam_5 = Word2Vec(tokenized_sentences, sg=1, negative=5)
    model_neg_sam_15 = Word2Vec(tokenized_sentences, sg=1, negative=15)

    word_vectors_neg_sam_1 = model_neg_sam_1.wv
    word_vectors_neg_sam_5 = model_neg_sam_5.wv
    word_vectors_neg_sam_15 = model_neg_sam_15.wv

    word_vectors_neg_sam_1.save('word_vectors_neg_sam_1.kv')
    word_vectors_neg_sam_5.save('word_vectors_neg_sam_5.kv')
    word_vectors_neg_sam_15.save('word_vectors_neg_sam_15.kv')

    #####################################################
    # Vector length options (with default window size 5)
    #####################################################
    model_vec_len_50 = Word2Vec(tokenized_sentences, sg=1, vector_size=50, negative=5)
    model_vec_len_100 = Word2Vec(tokenized_sentences, sg=1, vector_size=100, negative=5)
    model_vec_len_300 = Word2Vec(tokenized_sentences, sg=1, vector_size=300, negative=5)

    word_vectors_vec_len_50 = model_vec_len_50.wv
    word_vectors_vec_len_100 = model_vec_len_100.wv
    word_vectors_vec_len_300 = model_vec_len_300.wv

    word_vectors_vec_len_50.save('word_vectors_vec_len_50.kv')
    word_vectors_vec_len_100.save('word_vectors_vec_len_100.kv')
    word_vectors_vec_len_300.save('word_vectors_vec_len_300.kv')


    #######################################################
    # Context window options (with default 100 vector_size)
    #######################################################
    model_window_2 = Word2Vec(tokenized_sentences, sg=1,window=2, negative=5)
    model_window_5 = Word2Vec(tokenized_sentences, sg=1, window=5, negative=5)
    model_window_10 = Word2Vec(tokenized_sentences, sg=1, window=10, negative=5)

    word_vectors_window_2 = model_window_2.wv
    word_vectors_window_5 = model_window_5.wv
    word_vectors_window_10 = model_window_10.wv

    word_vectors_window_2.save('word_vectors_window_2.kv')
    word_vectors_window_5.save('word_vectors_window_5.kv')
    word_vectors_window_10.save('word_vectors_window_10.kv')



#####
# SVD
#####
# Creates and saves SVD model variations
def svd_models():

    def create_co_occurance_matrix(window_size=2):
        context_dict = dict()
        vocab = set()

        # Create the co-occurance matrix
        for sentence in tokenized_sentences:
            for i, word in enumerate(sentence):
                # Make sure we stay within bounds of the sentence
                # And skip current word when finding context words
                if i-window_size < 0:
                    window_left = 0
                else:
                    window_left = i-window_size
                if i+1+window_size >= len(sentence):
                    window_right = len(sentence)
                else:
                    window_right = i+1+window_size
                # Context words are all words within window_size of the current word
                context_words = sentence[window_left:i] + sentence[i+1:window_right]
                vocab.add(word)
                # Generate values for word, context_word pairs
                for context_word in context_words:
                    context_dict[(word, context_word)] = context_dict.get((word, context_word), 0) + 1
        
        vocab  = list(sorted(vocab))
        print('generated words and context words dict')
        print(len(vocab))
        # Only needed once
        # vocab_pickle = open('vocab.txt', 'wb')
        # pickle.dump(vocab, vocab_pickle)

        # Build co-occurence matrix from dict of word-context pairs
        context_matrix = lil_matrix((len(vocab), len(vocab)))
        print('created lil matrix, about to fill it')
        for (word, context_word), v in context_dict.items():
            word_index = vocab.index(word)
            context_word_index = vocab.index(context_word)
            context_matrix[word_index, context_word_index] = v
        print('about to save co occurence matrix')
        coo_context_matrix = context_matrix.tocoo()
        
        # Uncomment to save certain files when experimenting
        # save_npz("co_occurance_coo_matrix_win_2.npz", coo_context_matrix)
        # save_npz("co_occurance_coo_matrix_win_5.npz", coo_context_matrix)
        # save_npz("co_occurance_coo_matrix_win_10_new.npz", coo_context_matrix)


    # Transform co_occurance matrix to positive pointwise mutual information matrix
    def co_occurance_to_ppmi(matrix, window):
        matrix = matrix.tocsr()
        
        # Get the total sums of the words and context words
        sum_columns = np.array(matrix.sum(axis=0)).squeeze()
        # Because matrix is symmetric, sum of rows and sum of cols are the same, we can store them together
        sum_rows_and_cols = sum_columns

        total_sum = matrix.sum()

        # Get probabilities
        p_words_and_contexts = sum_rows_and_cols/total_sum
        p_matrix_entries = matrix/total_sum

        # Get pmi matrix
        word_indices, context_indices = matrix.nonzero() # Get indices of non-zero elements in matrix
        log_p_matrix_entries = np.log(p_matrix_entries.data)
        log_p_words = np.log(p_words_and_contexts[word_indices])
        log_p_contexts = np.log(p_words_and_contexts[context_indices])
        pmi_matrix = log_p_matrix_entries - (log_p_words + log_p_contexts)
        
        # Turn pmi into ppmi matrix
        pmi_matrix[pmi_matrix < 0] = 0
        ppmi_matrix = p_matrix_entries
        ppmi_matrix.data = pmi_matrix

        print('saving ppmi matrix file')
        if window == 2:
            save_npz("ppmi_matrix_window_2.npz", ppmi_matrix)
        if window == 5:
            save_npz("ppmi_matrix_window_5.npz", ppmi_matrix)
        if window==10:
            save_npz("ppmi_matrix_window_10.npz", ppmi_matrix)

    # Use SVD to get word embedding matrix
    # Takes as params a ppmi matrix and dimensionality of embeddings (aka "k")
    def factorization_to_embeddings(matrix, dimensionality):
        u, s, v = svds(matrix, dimensionality)
        partial_s = np.power(s, 1/2)
        return(u*partial_s)


    def write_to_embedding_file(output_file, svd_embeddings, vocab):
        print('writing embeddings to file')
        with open(output_file, 'w') as file:
            for i, row in enumerate(svd_embeddings):
                file.write(vocab[i] + ' ' + ' '.join([str(embedding) for embedding in row]) + '\n')




    #############################################################
    # Create/save the co occurance matrices with each window size
    #############################################################
    # create_co_occurance_matrix(window_size=2)
    # create_co_occurance_matrix(window_size=5)
    # create_co_occurance_matrix(window_size=10)
    

    ###############################################################
    # Create/save the ppmi matrices from the co occurrence matrices
    ###############################################################
    # co_occurance_window_2 = load_npz('co_occurance_coo_matrix_win_2.npz')
    # co_occurance_to_ppmi(co_occurance_window_2, window=2)

    # co_occurance_window_5 = load_npz('co_occurance_coo_matrix_win_5.npz')
    # co_occurance_to_ppmi(co_occurance_window_5, window=5)

    # co_occurance_window_10 = load_npz('co_occurance_coo_matrix_win_10_new.npz')
    # co_occurance_to_ppmi(co_occurance_window_10, window=10)


    ##############################################################################
    # Create/save the word embeddings using svd factorization on the ppmi matrices
    ##############################################################################

    #########################################
    # Dimension options with window length 2
    #########################################
    with open('vocab.txt', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)

    # ppmi_matrix_win_2 = load_npz('ppmi_matrix_window_2_test.npz')

    # svd_embeddings_win_2_dim_50 = factorization_to_embeddings(ppmi_matrix_win_2, 50)
    # write_to_embedding_file("svd_embeddings_win_2_dim_50_test.txt", svd_embeddings_win_2_dim_50, vocab)

    # svd_embeddings_win_2_dim_100 = factorization_to_embeddings(ppmi_matrix_win_2, 100)
    # write_to_embedding_file("svd_embeddings_win_2_dim_100_no_stop.txt", svd_embeddings_win_2_dim_100, vocab)

    # svd_embeddings_win_2_dim_300 = factorization_to_embeddings(ppmi_matrix_win_2, 300)
    # write_to_embedding_file("svd_embeddings_win_2_dim_300_no_stop.txt", svd_embeddings_win_2_dim_300, vocab)


    #########################################
    # Dimension options with window length 5
    #########################################
    # ppmi_matrix_win_5 = load_npz('ppmi_matrix_window_5.npz')

    # svd_embeddings_win_5_dim_50 = factorization_to_embeddings(ppmi_matrix_win_5, 50)
    # write_to_embedding_file("svd_embeddings_win_5_dim_50.txt", svd_embeddings_win_5_dim_50, vocab)

    # svd_embeddings_win_5_dim_100 = factorization_to_embeddings(ppmi_matrix_win_5, 100)
    # write_to_embedding_file("svd_embeddings_win_5_dim_100.txt", svd_embeddings_win_5_dim_100, vocab)

    # svd_embeddings_win_5_dim_300 = factorization_to_embeddings(ppmi_matrix_win_5, 300)
    # write_to_embedding_file("svd_embeddings_win_5_dim_300.txt", svd_embeddings_win_5_dim_300, vocab)


    ##########################################
    # Dimension options with window length 10
    ##########################################
    # ppmi_matrix_win_10 = load_npz('ppmi_matrix_window_10.npz')

    # svd_embeddings_win_10_dim_50 = factorization_to_embeddings(ppmi_matrix_win_10, 50)
    # write_to_embedding_file("svd_embeddings_win_10_dim_50.txt", svd_embeddings_win_10_dim_50, vocab)

    # svd_embeddings_win_10_dim_100 = factorization_to_embeddings(ppmi_matrix_win_10, 100)
    # write_to_embedding_file("svd_embeddings_win_10_dim_100.txt", svd_embeddings_win_10_dim_100, vocab)

    # svd_embeddings_win_10_dim_300 = factorization_to_embeddings(ppmi_matrix_win_10, 300)
    # write_to_embedding_file("svd_embeddings_win_10_dim_300.txt", svd_embeddings_win_10_dim_300, vocab)



######################################
# Produce two types of word embeddings
######################################
# word2vec_models()
svd_models()