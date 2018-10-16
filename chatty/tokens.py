from functools import partial
import spacy


nlp = spacy.load('en_core_web_sm')


def lemma(sep='_', tag='<LEMMA>'):
    "lemmas"
    def inner(doc: spacy.tokens.doc.Doc):
        tokens = []
        for tok in doc:
            tagged = sep.join((tag, tok.lemma_))
            tokens.append(tagged)
        return tokens
    return inner


def pos(sep='_', tag='<POS>'):
    "parts of speech"
    def inner(doc: spacy.tokens.doc.Doc):
        tokens = []
        for tok in doc:
            tagged = sep.join((tag, tok.pos_))
            tokens.append(tagged)
        return tokens
    return inner


def dep(sep='_', tag='<DEP>'):
    "dependency"
    def inner(doc: spacy.tokens.doc.Doc):
        tokens = []
        for tok in doc:
            tagged = sep.join((tag, tok.dep_))
            tokens.append(tagged)
        return tokens
    return inner


def word(sep='_', tag='<WORD>', lower=True):
    "word tokens"
    def inner(doc: spacy.tokens.doc.Doc):
        tokens = []
        for tok in doc:
            tagged = sep.join((tag, tok.text.lower() if lower else tok.text))
            tokens.append(tagged)
        return tokens
    return inner


def chain(tokenizers=[], sep='_'):
    "for pulling out multiple kinds of features per token"
    def inner(doc: spacy.tokens.doc.Doc):
        each_tokenized = []
        for tokenize in tokenizers:
            tokenized = tokenize(doc)
            each_tokenized.append(tokenized)
        return [sep.join(i) for i in zip(*each_tokenized)]
    return inner


def ngramize(tokenizer, ngrams=[1], sep='_'):
    def inner(doc: spacy.tokens.doc.Doc):
        "convert an output of tokens into an ngramized version of it"
        tokens = tokenizer(doc)
        output = []
        for ngram_size in ngrams:
            for i in range(ngram_size, len(tokens) + 1):
                output.append(sep.join(tokens[i - ngram_size: i]))
        return output
    return inner


def tokenize(string: str, tokenizers=[]):
    "all tokenizers come through here"
    doc = nlp(string)
    tokens = []
    for tokenizer in tokenizers:
        for token in tokenizer(doc):
            tokens.append(token)
    return tokens


def build(tokenizers=[]):
    tokenizer = partial(tokenize, tokenizers=tokenizers)
    return tokenizer
