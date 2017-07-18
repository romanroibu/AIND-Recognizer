import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # DONE implement the recognizer

    for word_id in range(0, len(test_set.get_all_sequences())):        
        X, lengths = test_set.get_item_Xlengths(word_id)

        best_guess = None
        best_score = None
        probability_dict = {}

        for word, model in models.items():
            try:
                logL = model.score(X, lengths)
            except:
                logL = -float('inf')

            probability_dict[word] = logL

            if not best_score or best_score < logL:
                best_score = logL
                best_guess = word

        probabilities.append(probability_dict)
        guesses.append(best_guess)

    # return probabilities, guesses
    return (probabilities, guesses)
