from functools import partial, wraps
import spacy


nlp = spacy.load('en_core_web_sm')


def lazy_tag_spacy_doc(sep: str, tag: str, **kwargs):
    def wrapper(func):
        @wraps(func)
        def inner(**kwargs):
            def spacy_getter(doc: spacy.tokens.doc.Doc):
                tokens = []
                for tok in doc:
                    tagged = sep.join((tag, func(tok, **kwargs)))
                    tokens.append(tagged)
                return tokens
            return spacy_getter
        return inner
    return wrapper


@lazy_tag_spacy_doc(sep='_', tag='<LEMMA>')
def lemma(tok: spacy.tokens.token.Token):
    "get lemmas"
    return tok.lemma_


@lazy_tag_spacy_doc(sep='_', tag='<POS>')
def pos(tok: spacy.tokens.token.Token):
    return tok.pos_


@lazy_tag_spacy_doc(sep='_', tag='<DEP>')
def dep(tok: spacy.tokens.token.Token):
    return tok.dep_


@lazy_tag_spacy_doc(sep='_', tag='<WORD>', lower=True)
def word(tok, lower=True):
    return tok.text.lower() if lower else tok.text


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
