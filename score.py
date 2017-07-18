from matplotlib import (cm, pyplot as plt, mlab)
from my_recognizer import recognize
from asl_utils import SinglesData
import numpy as np

def plot_scores(stats):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    N = len(stats[0][1])
    xs = np.arange(N)
    bar_width = 0.35
    spacing = 2

    bars = []
    for idx, stat in enumerate(stats):
        feature_wers = [ score[1] for score in stat[1] ]
        bar = ax.bar(xs*spacing + bar_width*idx, feature_wers, bar_width)
        bars.append(bar[0])

    ax.set_xlim(-bar_width, (N)*spacing + bar_width*(N-1)) #FIXME: Compute graph width
    ax.set_ylim(0.5, 0.70)
    ax.set_ylabel('Scores')
    ax.set_title('WER by model selectors and feature sets')
    xTickMarks = [ score[0] for score in stats[0][1] ]
    ax.set_xticks((xs + bar_width) * N/2)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)

    ## add a legend
    group_names = tuple( score[0] for score in stats )
    ax.legend(tuple(bars) , group_names)

    plt.show()

def wer_score(guesses: list, test_set: SinglesData):
    S = 0
    N = len(test_set.wordlist)

    if len(guesses) != N:
        print("Size of guesses must equal number of test words ({})!".format(num_test_words))

    for word_id in range(N):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    # WER = (S+I+D)/N  but we have no insertions or deletions for isolated words so WER = S/N
    return float(S) / float(N)

def run(feature_sets, model_selectors, asl, train_func):
    scores = []

    for feature_name, feature_set in feature_sets:
        feature_scores = []

        for selector_name, model_selector in model_selectors:
            models = train_func(feature_set, model_selector)
            test_set = asl.build_test(feature_set)
            probabilities, guesses = recognize(models, test_set)
            feature_scores.append((selector_name, wer_score(guesses, test_set)))

        scores.append((feature_name, feature_scores))

    return scores

# Scores collected previously by calling the `run` function
collected_scores = [
    ('Ground', [
        ('Constant', 0.6685393258426966),
        ('CV', 0.5449438202247191),
        ('BIC', 0.550561797752809),
        ('DIC', 0.6348314606741573)
    ]),
    ('Norm', [
        ('Constant', 0.651685393258427),
        ('CV', 0.5955056179775281),
        ('BIC', 0.6123595505617978),
        ('DIC', 0.6404494382022472)
    ]),
    ('Polar', [
        ('Constant', 0.6179775280898876),
        ('CV', 0.5786516853932584),
        ('BIC', 0.5449438202247191),
        ('DIC', 0.6404494382022472)
    ]),
    ('Delta', [
        ('Constant', 0.6404494382022472),
        ('CV', 0.6797752808988764),
        ('BIC', 0.6179775280898876),
        ('DIC', 0.6067415730337079)
    ]),
    ('Custom', [
        ('Constant', 0.6179775280898876),
        ('CV', 0.6123595505617978),
        ('BIC', 0.6123595505617978),
        ('DIC', 0.6292134831460674)
    ])
]

