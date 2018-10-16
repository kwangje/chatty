import chatty.tokens as tokens


SENT = " What are the disadvantages of on-line shopping ? "
doc = tokens.nlp(SENT)


def test_lemma():
    expected = [
        '<LEMMA>_ ',
        '<LEMMA>_what',
        '<LEMMA>_be',
        '<LEMMA>_the',
        '<LEMMA>_disadvantage',
        '<LEMMA>_of',
        '<LEMMA>_on',
        '<LEMMA>_-',
        '<LEMMA>_line',
        '<LEMMA>_shopping',
        '<LEMMA>_?'
    ]
    output = tokens.lemma(doc)
    assert output == expected


def test_pos():
    expected = [
        '<POS>_SPACE',
        '<POS>_NOUN',
        '<POS>_VERB',
        '<POS>_DET',
        '<POS>_NOUN',
        '<POS>_ADP',
        '<POS>_ADP',
        '<POS>_PUNCT',
        '<POS>_NOUN',
        '<POS>_NOUN',
        '<POS>_PUNCT'
    ]
    output = tokens.pos(doc)
    assert output == expected


def test_ngramize_pos():
    expected = [
        '<POS>_SPACE_<POS>_NOUN',
        '<POS>_NOUN_<POS>_VERB',
        '<POS>_VERB_<POS>_DET',
        '<POS>_DET_<POS>_NOUN',
        '<POS>_NOUN_<POS>_ADP',
        '<POS>_ADP_<POS>_ADP',
        '<POS>_ADP_<POS>_PUNCT',
        '<POS>_PUNCT_<POS>_NOUN',
        '<POS>_NOUN_<POS>_NOUN',
        '<POS>_NOUN_<POS>_PUNCT'
    ]
    tokenizer = tokens.ngramize(tokens.pos, [2])
    output = tokenizer(doc)
    assert output == expected


def test_chain_pos_lemma():
    expected = [
        '<POS>_SPACE_<LEMMA>_ ',
        '<POS>_NOUN_<LEMMA>_what',
        '<POS>_VERB_<LEMMA>_be',
        '<POS>_DET_<LEMMA>_the',
        '<POS>_NOUN_<LEMMA>_disadvantage',
        '<POS>_ADP_<LEMMA>_of',
        '<POS>_ADP_<LEMMA>_on',
        '<POS>_PUNCT_<LEMMA>_-',
        '<POS>_NOUN_<LEMMA>_line',
        '<POS>_NOUN_<LEMMA>_shopping',
        '<POS>_PUNCT_<LEMMA>_?'
    ]
    tokenizer = tokens.chain(tokenizers=[tokens.pos, tokens.lemma])
    output = tokenizer(doc)
    assert output == expected


def test_build_tokenizer():
    expected = [
        '<POS>_SPACE_<LEMMA>_ ',
        '<POS>_NOUN_<LEMMA>_what',
        '<POS>_VERB_<LEMMA>_be',
        '<POS>_DET_<LEMMA>_the',
        '<POS>_NOUN_<LEMMA>_disadvantage',
        '<POS>_ADP_<LEMMA>_of',
        '<POS>_ADP_<LEMMA>_on',
        '<POS>_PUNCT_<LEMMA>_-',
        '<POS>_NOUN_<LEMMA>_line',
        '<POS>_NOUN_<LEMMA>_shopping',
        '<POS>_PUNCT_<LEMMA>_?'
    ]
