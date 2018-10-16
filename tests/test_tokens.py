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
    tokenizer = tokens.lemma()
    output = tokenizer(doc)
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
    tokenizer = tokens.pos()
    output = tokenizer(doc)
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
    tokenizer = tokens.ngramize(tokens.pos(), [2])
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
    tokenizer = tokens.chain(tokenizers=[tokens.pos(), tokens.lemma()])
    output = tokenizer(doc)
    assert output == expected


def test_build_tokenizer():
    expected = [
        '<LEMMA>_ _<POS>_SPACE_<LEMMA>_what_<POS>_NOUN',
        '<LEMMA>_what_<POS>_NOUN_<LEMMA>_be_<POS>_VERB',
        '<LEMMA>_be_<POS>_VERB_<LEMMA>_the_<POS>_DET',
        '<LEMMA>_the_<POS>_DET_<LEMMA>_disadvantage_<POS>_NOUN',
        '<LEMMA>_disadvantage_<POS>_NOUN_<LEMMA>_of_<POS>_ADP',
        '<LEMMA>_of_<POS>_ADP_<LEMMA>_on_<POS>_ADP',
        '<LEMMA>_on_<POS>_ADP_<LEMMA>_-_<POS>_PUNCT',
        '<LEMMA>_-_<POS>_PUNCT_<LEMMA>_line_<POS>_NOUN',
        '<LEMMA>_line_<POS>_NOUN_<LEMMA>_shopping_<POS>_NOUN',
        '<LEMMA>_shopping_<POS>_NOUN_<LEMMA>_?_<POS>_PUNCT',
        '<WORD>_ ',
        '<WORD>_what',
        '<WORD>_are',
        '<WORD>_the',
        '<WORD>_disadvantages',
        '<WORD>_of',
        '<WORD>_on',
        '<WORD>_-',
        '<WORD>_line',
        '<WORD>_shopping',
        '<WORD>_?'
    ]
    tokenizer = tokens.build([
        tokens.ngramize(
            tokens.chain([
                tokens.lemma(),
                tokens.pos(),
            ]), 
            [2]
        ),
        tokens.word()
    ])
    output = tokenizer(SENT)
    assert output == expected
