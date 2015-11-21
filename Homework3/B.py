import A
from sklearn.feature_extraction import DictVectorizer
import lib.Universal_tagger as UniversalTagger
import nltk
from sklearn import svm
import pickle

# You might change the window size
window_size = 10

# controls the word features
WORD_WINDOW = 2
WORD_HEAD = True

# controls the POS features
POS_WINDOW = 0
POS_HEAD = True


# B.1.a,b,c,d
def extract_features(data, cached_pos_tags):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}


    # implement your code here
    for (instance_id, left_context, head, right_context, sense_id) in data:
        vector = {}
        left_tags, head_tag, right_tags = cached_pos_tags[instance_id]

        # prepare feature information
        left_tokens = nltk.word_tokenize(left_context)
        right_tokens = nltk.word_tokenize(right_context)

        # add features
        word_head = head if WORD_HEAD else None
        pos_head = head_tag if POS_HEAD else None
        add_k_word_features_to_vector(vector, left_tokens, right_tokens, WORD_WINDOW, word_head)
        add_k_word_POS_features_to_vector(vector, left_tags, right_tags, POS_WINDOW, pos_head)

        # track results
        features[instance_id] = vector
        labels[instance_id] = sense_id

    return features, labels


#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_k_word_features_to_vector(vector, left_tokens, right_tokens, window_size, head=None):
    words = A.k_nearest_words_vector_from_tokens(left_tokens, right_tokens, window_size)
    mid = len(words)/2
    left = words[:mid]
    right = words[mid:]
    for idx, word in enumerate(left):
        key = 'w_b' + str(len(left) - idx)
        vector[key] = word

    for idx, word in enumerate(right):
        key = 'w_a' + str(idx+1)
        vector[key] = word

    if head:
        key = 'w_head'
        vector[key] = head


#  Adds wb1 for 1st word before head and wa1 for first word after head... to +-n words
def add_k_word_POS_features_to_vector(vector, left_tags, right_tags, window_size, head_tag=None):

    words = A.k_nearest_words_vector_from_tokens(left_tags, right_tags, window_size)
    mid = len(words)/2
    left = words[:mid]
    right = words[mid:]

    for idx, (word, tag) in enumerate(left):
        key = 'pos_b' + str(len(left_tags) - idx)
        vector[key] = tag

    for idx, (word, tag) in enumerate(right):
        key = 'pos_a' + str(idx+1)
        vector[key] = tag

    # add POS tag for head
    if head_tag:
        key = 'pos_head'
        word, tag = head_tag[0]
        vector[key] = tag




# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

# create x, y lists from training datas

    x_train_list, y_train_list = A.x_y_lists_from_training(X_train, y_train)

    # train svm
    print 'training svm...'
    svm_clf = svm.LinearSVC()
    svm_clf.fit(x_train_list, y_train_list)

    # predict svm results
    print 'predicting svm...'
    svm_results = A.predictions_from_data(svm_clf, X_test)

    return svm_results


# saves tagges corpus for faster processing
def tag_and_save(train, test, language):
    train_tagged = {}
    test_tagged = {}
    print 'training tagger'
    tagger = UniversalTagger.EnglishTagger()
    counter = 0
    for lexelt in train:

        # train
        for (instance_id, left_context, head, right_context, sense_id) in train[lexelt]:

            counter += 1
            print 'saving train' + str(counter)

            # prepare feature information
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)

            left_tags = tagger.tag(left_tokens)
            middle = tagger.tag([head])
            right_tags = tagger.tag(right_tokens)

            # add features
            train_tagged[instance_id] = (left_tags, middle, right_tags)

        # test
        for (instance_id, left_context, head, right_context, sense_id) in test[lexelt]:

            counter += 1
            print 'saving test' + str(counter)

            # prepare feature information
            left_tokens = nltk.word_tokenize(left_context)
            right_tokens = nltk.word_tokenize(right_context)

            left_tags = tagger.tag(left_tokens)
            middle = tagger.tag([head])
            right_tags = tagger.tag(right_tokens)

            # add features
            test_tagged[instance_id] = (left_tags, middle, right_tags)

    # save with pickle
    train = language + '-train.p'
    test = language + '-test.p'

    print 'saving train'
    print 'saving test'
    pickle.dump(train_tagged, open(train, 'wb'))
    pickle.dump(test_tagged, open(test, 'wb'))
    print 'saved pickle'


# run part B
def run(train, test, language, answer):
    results = {}

    # tag_and_save(train, test, language)

    # load cached POS tags
    print 'loading cached pos tags...'
    train_name = language + '-train.p'
    test_name = language + '-test.p'
    train_pos_tags = pickle.load(open(train_name, 'rb'))
    test_pos_tags = pickle.load(open(test_name, 'rb'))

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt], train_pos_tags)
        test_features, _ = extract_features(test[lexelt], test_pos_tags)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)