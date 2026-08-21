from asr_eval.normalize import Normalizer, NormalizerConfig


def test_lowercases_and_strips_punctuation():
    n = Normalizer()
    assert n.normalize("Hello, World!") == "hello world"


def test_keeps_case_when_disabled():
    n = Normalizer(NormalizerConfig(lowercase=False))
    assert n.normalize("Hello World") == "Hello World"


def test_expands_contractions_so_house_style_is_not_an_error():
    n = Normalizer()
    assert n.normalize("don't stop") == "do not stop"
    assert n.normalize("can't stop") == "cannot stop"


def test_collapses_whitespace():
    n = Normalizer()
    assert n.normalize("  a   b \n c ") == "a b c"


def test_fillers_removed_only_when_enabled():
    text = "um so uh yes"
    assert Normalizer().normalize(text) == "um so uh yes"
    dropped = Normalizer(NormalizerConfig(drop_fillers=True)).normalize(text)
    assert dropped == "so yes"


def test_tokenize_and_characters():
    n = Normalizer()
    assert n.tokenize("Hi there!") == ["hi", "there"]
    assert n.characters("Hi there!") == list("hithere")


def test_none_and_empty_are_safe():
    n = Normalizer()
    assert n.normalize("") == ""
    assert n.normalize(None) == ""
    assert n.tokenize("") == []


def test_config_roundtrips_through_dict():
    cfg = NormalizerConfig(drop_fillers=True, lowercase=False)
    assert NormalizerConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_ignores_unknown_keys():
    cfg = NormalizerConfig.from_dict({"lowercase": False, "not_a_real_field": 1})
    assert cfg.lowercase is False


def test_unicode_is_normalized():
    n = Normalizer()
    # Composed vs decomposed accents must not read as a substitution.
    assert n.normalize("café") == n.normalize("café")
