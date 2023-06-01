
def tokens_to_sent(vocab, tokens):
  keys = list(vocab.keys())
  values = list(vocab.values())
  vocab_size = len(keys)
  positions = [values.index(e) for e in tokens[tokens != vocab_size + 1]]
  words = [keys[p] for p in positions]
  return ' '.join(words)