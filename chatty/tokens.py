import spacy


nlp = spacy.load('en_core_web_sm')


def lemma(doc: spacy.tokens.doc.Doc, sep='_', tag='<LEMMA>'):
    "lemmas"
    tokens = []
    for tok in doc:
        tagged = sep.join((tag, tok.lemma_))
        tokens.append(tagged)
    return tokens


def pos(doc: spacy.tokens.doc.Doc, sep='_', tag='<POS>'):
    "parts of speech"
    tokens = []
    for tok in doc:
        tagged = sep.join((tag, tok.pos_))
        tokens.append(tagged)
    return tokens


def word(doc: spacy.tokens.doc.Doc, sep='_', tag='<WORD>', lower=True):
    "word tokens"
    tokens = []
    for tok in doc:
        tagged = sep.join((tag, tok.text.lower() if lower else tok.text))
        tokens.append(tagged)
    return tokens


def chain(doc: spacy.tokens.doc.Doc, tokenizers=[], sep='_'):
    "for pulling out multiple features per token"
    each_tokenized = []
    for tokenize in tokenizers:
        tokenized = tokenize(doc)
        each_tokenized.append(tokenized)
    return [sep.join(i) for i in zip(*each_tokenized)]


def ngramize(tokens, ngrams=[1], sep='_'):
    "convert an output of tokens into an ngramized version of it"
    output = []
    for ngram_size in ngrams:
        for i in range(ngram_size, len(tokens) + 1):
            output.append(sep.join(tokens[i - ngram_size: i]))
    return output


def tokenize(string: str, tokenizers=[], sep='_'):
    "all tokenizers come through here"
    doc = nlp(string)
    tokens = []
    for tokenizer in tokenizers:
        for token in tokenizer(doc, sep=sep):
            tokens.append(token)
    return tokens
