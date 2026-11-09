from regrag.text import tokenize


def test_rule_numbers_stay_whole():
    assert "11.2.1" in tokenize("Under Rule 11.2.1, a Firm must report.")


def test_stopwords_are_dropped_and_words_stemmed():
    assert tokenize("The reporting officers") == ["report", "offic"]


def test_stemming_can_be_turned_off():
    assert tokenize("The reporting officers", stem=False) == ["reporting", "officers"]


def test_empty_and_stopword_only_text():
    assert tokenize("") == []
    assert tokenize("the of and") == []
