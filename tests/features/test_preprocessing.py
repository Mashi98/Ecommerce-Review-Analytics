from src.features.preprocessing import clean_text, preprocess, tokenize


def test_clean_text_lowercases_and_strips_punct_digits():
    assert clean_text("Great Shirt! Size 12 fits well.") == "great shirt size fits well"


def test_clean_text_strips_urls_and_html():
    assert clean_text("Check <b>this</b> out http://example.com/x") == "check this out"


def test_tokenize_removes_stopwords():
    tokens = tokenize("This is a very comfortable dress")
    assert "is" not in tokens
    assert "a" not in tokens
    assert "comfortable" in tokens
    assert "dress" in tokens


def test_tokenize_lemmatizes_verbs_and_nouns():
    tokens = tokenize("The dresses were running small and she was sizing up")
    assert "dress" in tokens
    assert "run" in tokens
    assert "size" in tokens


def test_preprocess_returns_joined_string():
    result = preprocess("I absolutely loved this flattering shirt!")
    assert isinstance(result, str)
    assert "love" in result
    assert "shirt" in result


def test_preprocess_handles_missing_text():
    assert preprocess(None) == ""
    assert preprocess("") == ""
    assert preprocess("   ") == ""
