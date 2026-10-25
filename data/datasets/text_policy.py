"""String-level MSVR310 text policies, shared by training and inference.

Generic removes description content but deliberately retains its effective
length. It is not a claim that all sample-dependent information is removed.
The returned string still passes through ImageDataset's original tokenizer.
"""


def validate_text_policy(mode, generic_length='match_real'):
    if mode not in ('real', 'generic'):
        raise ValueError(f'Unsupported text mode: {mode!r}; expected real or generic')
    if generic_length != 'match_real':
        raise ValueError(f'Unsupported generic length policy: {generic_length!r}')


def apply_text_policy(description, prefix, mode, tokenizer=None,
                      text_length=77, generic_word='vehicle',
                      generic_length='match_real'):
    """Return a caption, preserving the original string exactly in real mode.

Generic replaces retained description BPE tokens with a repeated ordinary word.
77 matches ImageDataset's existing context length (including SOT and EOT).
No random sampling, special-token substitution, or label information is used.
"""
    validate_text_policy(mode, generic_length)
    caption = prefix + description
    if mode == 'real':
        return caption
    if tokenizer is None:
        raise ValueError('generic text requires the dataset SimpleTokenizer')
    if text_length < 2:
        raise ValueError('text_length must leave space for SOT and EOT')

    word_tokens = tokenizer.encode(generic_word)
    special_tokens = {tokenizer.encoder[key] for key in
                      ('<|startoftext|>', '<|endoftext|>', '<|mask|>')}
    if len(word_tokens) != 1 or word_tokens[0] in special_tokens:
        raise ValueError('GENERIC_WORD must encode to one ordinary token')

    prefix_tokens = tokenizer.encode(prefix)
    real_tokens = tokenizer.encode(caption)
    capacity = text_length - 2
    if len(prefix_tokens) > capacity:
        raise ValueError('Text prefix exceeds the available context length')
    if real_tokens[:len(prefix_tokens)] != prefix_tokens:
        raise ValueError('Tokenization changed the prefix/description boundary')
    if any(token in special_tokens for token in real_tokens):
        raise ValueError('Caption contains a reserved special token')

    retained_length = min(len(real_tokens), capacity)
    count = retained_length - len(prefix_tokens)
    generic_caption = prefix + ' '.join([generic_word] * count)
    generic_tokens = tokenizer.encode(generic_caption)
    expected = prefix_tokens + word_tokens * count
    # This also checks EOT/padding placement under the unchanged tokenize().
    if generic_tokens != expected or len(generic_tokens) != retained_length:
        raise ValueError('Generic text failed prefix or effective-length matching')
    return generic_caption
