from sentence_transformers import CrossEncoder

MODEL = CrossEncoder("cross-encoder/stsb-roberta-base")


# Please note
# Implementation is from https://github.com/NC0DER/KeyphraseExtraction/blob/main/KeyExt/metrics.py
# and was slightly modified (Combined with overlap coefficient, threshold of 0.5)
def exact_f1(extracted, references): 
    """Function to compute exact match F-score for a list of extracted keyphrases and a reference list"""
    P = len(set(extracted) & set(references)) / len(extracted)
    R = len(set(extracted) & set(references)) / len(references)
    F = (2*P*R)/(P+R) if (P+R) > 0 else 0 
    return F # modified


def overlap(A: set, B: set):
    return len(A & B) / min(len(A), len(B))


# Please note!
# Implementation is from https://github.com/NC0DER/KeyphraseExtraction/blob/main/KeyExt/metrics.py
# and was modified (Combined with overlap coefficient, threshold of 0.5)
def partial_f1(extracted, references): # name modified
    """
    Computes the exatch match f1 measure at k.
    Arguments
    ---------
    extracted : A list of extracted keyphrases.
    references  : A list of human assigned keyphrases.

    Returned value
    --------------
              : double
    """
    # Exit early, if one of the lists or both are empty.
    if not extracted or not references:
        return 0.0

    # Store the longest keyphrases first.
    references_sets = sorted([set(keyword.split()) for keyword in references], key = len, reverse = True)
    extracted_sets = sorted([set(keyword.split()) for keyword in extracted], key = len, reverse = True)

    # This list stores True, if the assigned keyphrase has been matched earlier.
    # To avoid counting duplicate matches.
    references_matches = [False for references_set in references_sets]

    # For each extracted keyphrase, find the closest match, 
    # which is the assigned keyphrase it has the most words in common.
    for extracted_set in extracted_sets:
        # Modification 1
        all_matches = [(i, overlap(extracted_set, references_set)) for i, references_set in enumerate(references_sets)]
        closest_match = sorted(all_matches, key = lambda x: x[1], reverse = True)[0]
        # Modification 2
        if closest_match[1] >= 0.5:
            references_matches[closest_match[0]] = True
    # Calculate the precision and recall metrics based on the partial matches.
    partial_matches = references_matches.count(True)
    precision_k = partial_matches / len(extracted)
    recall_k = partial_matches / len(references)
    
    return (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if precision_k and recall_k else 0.0
    )


def similarity(extracted, references):
    N = len(references)
    # Find the most similar reference kp for each extracted kp
    most_similar = []
    for kp in extracted:
        pairs = zip([kp for _ in range(N)], references)
        scores = MODEL.predict(list(pairs))
        # Index of the reference keyphrase that was most similar
        best_idx = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[0][0]
        most_similar.append(references[best_idx])
    
    # print(f"Most similar: {most_similar}")
    scores = MODEL.predict(list(zip(extracted, most_similar)))

    return sum(scores) / len(extracted)


def average_metric(extracted, test_references, metric):
    assert(len(extracted)==len(test_references))
    N = len(extracted)

    total_score = 0
    for doc_extracted, doc_references in zip(extracted, test_references):
        total_score += metric(doc_extracted, doc_references)

    return total_score / N


if __name__ == "__main__":
    assigned = ["neural networks", "word embeddings"]
    extracted_a = ["computer science", "ethical considerations"]
    extracted_b = ["neural networks", "ethical considerations"]

    print(partial_f1(references=assigned, extracted=extracted_a))
    print(partial_f1(references=assigned, extracted=extracted_b))


