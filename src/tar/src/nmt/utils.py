import statistics as stat
import numpy as np
import matplotlib.pyplot as plt


# This function computes the average sentence length for a given corpora
def average_len(file):
    with open(file) as fn:
        sentences = fn.readlines()

    len_sentences = [len(sentence.split()) for sentence in sentences]
    len_avg, len_std, len_max = stat.mean(len_sentences), stat.stdev(len_sentences), max(len_sentences)
    print('Average sentence length\tStandard deviation\tMaximum sentence length\n')
    print('{}\t{}\t{}'.format(len_avg, len_std, len_max))
    return len_avg, len_std, len_max


# Compute the histogram for the sentence lenght in a given corpora
def sentence_len_histogram(file, plot=False):
    with open(file) as fn:
        sentences = fn.readlines()

    len_sentences = [len(sentence.split()) for sentence in sentences]
    hist, bins = np.histogram(len_sentences, bins=range(0, 90, 10))
    print('\t'.join(['bin {}'.format(bins[n]) for n in range(len(bins))]))
    print('\t'.join([str(hist[n]) for n in range(len(bins)-1)]))

    if plot:
        _ = plt.hist(hist, bins=bins)
        plt.title("Sentence length histogram")
        plt.show()
    return hist, bins
