import math
import numpy as np
from collections import Counter


def similarity_calculation(l1, l2):
    similar = 0
    c1 = Counter(l1)
    c2 = Counter(l2)
    cosine_similarity = cosine_similarity_calculation(c1, c2)
    # length_similarity = length_similarity_calculation(c1, c2)

    similarity = cosine_similarity  # * length_similarity
    if similarity >= 0.995:
        similar = 1

    return similarity, similar


def cosine_similarity_calculation(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0) ** 2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0) ** 2 for k in terms))
    if magA != 0 and magB != 0:
        cosine_similarity = dotprod / (magA * magB)
    else:
        cosine_similarity = 0
    return cosine_similarity


def length_similarity_calculation(c1, c2):
    lenc1 = len(c1)
    lenc2 = len(c2)

    length_similarity = min(lenc1, lenc2) / float(max(lenc1, lenc2))
    return length_similarity


def accuracy_calculation(l1, l2):
    accuracy = 0
    accurate = 0

    s1 = set(l1)
    s2 = set(l2)

    if s1 == s2:
        accurate = 1

    s_similar = s1 & s2
    if len(s2) > 0:
        accuracy = 100 / len(s2) * len(s_similar)

    return accuracy, accurate


def preference_score_calculation(position, l1, l2, preference_order_data, label_list):
    l1_diagnosis_importance = float(0)
    l2_diagnosis_importance = float(0)

    """
    Preference score is resulting out of sum of importance values.
    Importance value is 2 to the power of the reversed preference order number.
    The higher the preference score the more likely a user will accept the diagnosis.
    Vice versa the diagnosis importance has to be as small as possible.
    """

    ranking_score_list = list(map(lambda x: x, range(0, len(label_list))))
    ranking_score_list = sorted(ranking_score_list, reverse=True)

    for item in preference_order_data.keys():
        if '_po' in item:
            variable_name = item.rstrip('_po')
            if variable_name in l1:
                l1_diagnosis_importance += np.float64(np.log(np.longdouble(2 ** ranking_score_list[int(preference_order_data[item][position])])))
            if variable_name in l2:
                l2_diagnosis_importance += np.float64(np.log(np.longdouble(2 ** ranking_score_list[int(preference_order_data[item][position])])))

    return l1_diagnosis_importance, l2_diagnosis_importance
