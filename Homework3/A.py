from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}
    for lexelt, lexelt_info in data.items():
        words = []
        for (instance_id, left_context, head, right_context, sense_id) in lexelt_info:
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)
            words += k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)

        s[lexelt] = words

    return s


def k_nearest_words_vector_from_tokens(left_context, right_context, k):

    left_start = max(len(left_context) - k, 0)
    right_end = min(k, len(right_context))

    left_set = left_context[left_start:]
    right_set = right_context[:right_end]

    joint = left_set + right_set
    return joint


def insert_k_nearest_words_into_dict(left_context, right_context, k, freq_dict):

    left_start = max(len(left_context) - k, 0)
    right_end = min(k, len(right_context))

    for word in left_context[left_start:]:
        freq_dict[word] = freq_dict[word] + 1 if word in freq_dict else 1

    for word in right_context[:right_end]:
        freq_dict[word] = freq_dict[word] + 1 if word in freq_dict else 1


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    # implement your code here

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



