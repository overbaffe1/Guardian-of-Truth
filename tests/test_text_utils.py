

def test_estimate_tokens_accounts_for_cyrillic():
    """A flat len/3 underestimates Russian, which is ~2 chars per token."""
    from guardian.text_utils import estimate_tokens

    ru = "Пользователь хочет вернуть товар и получить деньги на карту"
    en = "The user wants to return the item and get money on the card"
    assert len(ru) == len(en)
    assert estimate_tokens(ru) > estimate_tokens(en)
    # Still bounded and never wildly inflated.
    assert 1 <= estimate_tokens("x") <= 2
    assert estimate_tokens("") == 0
