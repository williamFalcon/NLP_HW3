__author__ = 'waf'
import nltk
from nltk.data import load

from nltk.corpus import cess_esp as spa_corpus  # spanish
from nltk.corpus import cess_cat as cat_corpus  # catalan


class SpanishTagger:
    _tagger = None

    def __init__(self, train_percent_size=1):
        """

        :param train_percent_size: 0-1
        :return:
        """
        spanish_sents = spa_corpus.tagged_sents()
        subset = subset_from_corpus(spanish_sents, train_percent_size)
        self._tagger = trained_tagger_with_corpus(subset)

    def tag(self, words):
        return self._tagger.tag(words)


class CatalanTagger:
    _tagger = None

    def __init__(self, train_percent_size=1):
        """

        :param train_percent_size: 0-1
        :return:
        """
        catalan_sents = cat_corpus.tagged_sents()
        subset = subset_from_corpus(catalan_sents, train_percent_size)
        self._tagger = trained_tagger_with_corpus(subset)

    def tag(self, words):
        return self._tagger.tag(words)

class EnglishTagger:
    _tagger = None

    def __init__(self):
        """

        :param train_percent_size: 0-1
        :return:
        """
        _POS_TAGGER = 'taggers/maxent_treebank_pos_tagger/english.pickle'
        self._tagger = load(_POS_TAGGER)

    def tag(self, words):
        return self._tagger.tag(words)


def subset_from_corpus(corpus, percent):
    # Downsample corpus to percentage wanted
    train = int(len(corpus)*percent)

    # Extract subset requested for training
    subset = corpus[:train]
    return subset


def trained_tagger_with_corpus(corpus):
    t0 = nltk.DefaultTagger('NN')
    uni_tag = nltk.UnigramTagger(corpus, backoff=t0)
    bi_tag = nltk.BigramTagger(corpus, backoff=uni_tag)
    t3 = nltk.TrigramTagger(corpus, backoff=bi_tag)
    return t3