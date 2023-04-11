
def tokens_to_sent(vocab, tokens):
  keys = list(vocab.keys())
  values = list(vocab.values())
  positions = [values.index(e) for e in tokens[tokens != 1086]]
  words = [keys[p] for p in positions]
  return ' '.join(words)