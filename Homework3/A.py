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
        words = set()
        for (instance_id, left_context, head, right_context, sense_id) in lexelt_info:
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)
            words.update(k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size))

        s[lexelt] = list(words)

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
    for (instance_id, left_context, head, right_context, sense_id) in data:
        labels[instance_id] = sense_id
        left_tokens = nltk.word_tokenize(left_context)
        right_tokens = nltk.word_tokenize(right_context)
        words = k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)
        vectors[instance_id] = frequency_vector_from_near_words(s, words)

    return vectors, labels


def frequency_vector_from_near_words(s, words):
    freq_dict = list_frequencies(words)
    vector = empty_vector_of_size(len(s))
    for idx, word in enumerate(s):
        if word in freq_dict:
            vector[idx] = freq_dict[word]
    return vector


def empty_vector_of_size(size):
    return [0] * size


def list_frequencies(list):
    freq_dict = {}
    for obj in list:
        freq_dict[obj] = freq_dict[obj] + 1 if obj in freq_dict else 1
    return  freq_dict


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

    # create x, y lists from training datas
    x_train_list, y_train_list = x_y_lists_from_training(X_train, y_train)

    # train svm
    print 'training svm...'
    svm_clf = svm.LinearSVC()
    svm_clf.fit(x_train_list, y_train_list)

    # predict svm results
    print 'predicting svm...'
    svm_results = predictions_from_data(svm_clf, X_test)

    # train knn
    print 'training knn'
    knn_clf = neighbors.KNeighborsClassifier()
    knn_clf.fit(x_train_list, y_train_list)

    # predict knn
    print 'predicting knn'
    knn_results = predictions_from_data(knn_clf, X_test)

    return svm_results, knn_results


def predictions_from_data(classifier, x_test):
    results = []
    for x in x_test:
        prediction = classifier.predict(x_test[x])
        result = (x, prediction)
        results.append(result)
    return results


def x_y_lists_from_training(x_train, y_train):
    x_train_list = []
    y_train_list = []
    for obj in x_train:
        x_train_list.append(x_train[obj])
        y_train_list.append(y_train[obj])

    return x_train_list, y_train_list


# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    # line =      active.v     active.v.bnc.00123123 38201
    #             lexeltItem   instance_id          sense_id
    sentences = []
    for lexel in results:
        predictions = results[lexel]
        for instance_id, label in predictions:
            sentence = lexel + ' ' + instance_id + ' ' + label + '\n'
            sentences.append(sentence)

    # Sort alphabetically
    sentences = sorted(sentences)

    # write to file
    f = open(output_file, 'w')
    f.writelines(sentences)
    f.close()
    print ''


# run part A
def run(train, test, language, knn_file, svm_file):

    print 'building training data...'
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:

        print 'vectorizing training data...'
        X_train, y_train = vectorize(train[lexelt], s[lexelt])

        print 'vectorizing testing data'
        X_test, _ = vectorize(test[lexelt], s[lexelt])

        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



