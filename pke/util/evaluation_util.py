import json
import pke
import spacy

from tqdm.notebook import tqdm
from nltk.stem.snowball import SnowballStemmer as Stemmer
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex


# Load English language model from spacy
nlp = spacy.load("en_core_web_sm")

# Tokenization fix for in-word hyphens (e.g. 'non-linear' would be kept 
# as one token instead of default spacy behavior of 'non', '-', 'linear')
# https://spacy.io/usage/linguistic-features#native-tokenizer-additions

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer


def preprocess_dataset(dataset, split: str):
    train_docs = []
    train_references = []

    for sample in tqdm(dataset[split]):
        train_docs.append(nlp(" ".join(sample["document"])))

        sample_keyphrases = []
        for keyphrase in sample["extractive_keyphrases"]:
            # tokenize keyphrase
            tokens = [token.text for token in nlp(keyphrase)]
            # normalize tokens using Porter's stemming
            stems = [Stemmer('porter').stem(tok.lower()) for tok in tokens]
            sample_keyphrases.append(" ".join(stems))
        train_references.append(sample_keyphrases)
    
    assert(len(train_docs) == len(train_references))

    return train_docs, train_references



def extract_keyphrases(extractor, n: int, params: list, stemming: bool,
    docs: list, df: dict=None
    ):
    if extractor == pke.unsupervised.MultipartiteRank():
        assert(isinstance(params[0], float))
        assert df == None
    elif extractor == pke.unsupervised.KPMiner():
        assert(isinstance(params[0], tuple))
        assert df != None
    else:
        raise ValueError("Unknown extractor")
    
    extracted_keyphrases = []
    for doc in tqdm(docs):
        doc_keyphrases = []
        extractor.load_document(input=doc, language='en')
        extractor.grammar_selection(grammar="NP: {<ADJ>*<NOUN|PROPN>+}")

        if extractor == pke.unsupervised.MultipartiteRank():
            for alpha in params:
                extractor.candidate_weighting(alpha=alpha)

        elif extractor == pke.unsupervised.KPMiner():
            for alpha, sigma in params:
                extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)
            
        param_keyphrases = [
            kp for kp, _ in extractor.get_n_best(n=n, stemming=stemming)
            ]
        if len(params) == 1:
            doc_keyphrases.extend(param_keyphrases)
        else:
            doc_keyphrases.append(param_keyphrases)
        extracted_keyphrases.append(doc_keyphrases)
    
    return extract_keyphrases


def serialize(extracted: dict, path:str):
    n_to_extr_json = json.dumps(extracted)
    with open(path, "w+", encoding="utf") as f:
        f.write(n_to_extr_json)


def deserialize(path: str):
    with open(path, "r", encoding="utf") as f:
        n_to_extr_json = f.read()
        n_to_extr = json.loads(n_to_extr_json)
        # Convert keys (5, 10, 15) back to integers as json.dumps() converted them to strings
        n_to_extr = {int(key): value for key, value in n_to_extr.items()}
    return n_to_extr


def compute_scores(metric, params: list,  
    extracted_keyphrases: list, references: list
    ):
    assert len(extracted_keyphrases) == len(references)
    n_docs = len(references)

    scores = []
    for doc_references, doc_keyphrases in zip(references, extracted_keyphrases):
        doc_scores = []
        for param_keyphrases in doc_keyphrases:
            doc_scores.append(metric(param_keyphrases, doc_references))
        scores.append(doc_scores)

    param_to_score = dict()
    for doc_scores in scores:
        for param, score in zip(params, doc_scores):
            if param in param_to_score.keys():
                param_to_score[param] += score
            else:
                param_to_score[param] = score
    
    param_to_score = {
        param: score / n_docs for (param, score) in param_to_score.items()
        }
    
    return param_to_score


# Stemming of raw extracted keyphrases
def stem(keyphrases: list):
    stemmed_keyphrases = []
    for kp in keyphrases:
        tokens = [token.text for token in nlp(kp)]
        stems = [Stemmer('porter').stem(t.lower()) for t in tokens]
        stemmed_keyphrases.append(" ".join(stems))
    return stemmed_keyphrases


def perform_stemming(extracted_raw: dict):
    n_to_extr = dict()
    for n, extractions in extracted_raw.items():
        extractions_stemmed = []
        for doc_keyphrases in extractions:
            # doc_keyphrases is nested
            if type(doc_keyphrases[0]) == list: 
                doc_keyphrases_stemmed = []
                for param_keyphrases in doc_keyphrases:
                    param_keyphrases_stemmed = stem(param_keyphrases)
                    doc_keyphrases_stemmed.append(param_keyphrases_stemmed)
            # doc_keyphrases is not nested
            elif type(doc_keyphrases[0]) == str: 
                doc_keyphrases_stemmed = stem(doc_keyphrases)
            else:
                raise Exception("Something is wrong...")
            extractions_stemmed.append(doc_keyphrases_stemmed)
        n_to_extr[n] = extractions_stemmed
    return n_to_extr
