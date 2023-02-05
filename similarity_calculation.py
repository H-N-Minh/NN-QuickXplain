import math
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
