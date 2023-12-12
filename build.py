import tensorflow as tf
from transformers import BertModel
import bert

model = BertModel.from_pretrained("bert-base-uncased")

def find_nth(haystack: str, needle: str, n: int) -> int:
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# Grabbbing Encoder Weights
keys = model.state_dict().keys()
start = False
index = 0
layer_index = 0
dict_names = ["layer_"+ str(i) for i in range(12)]
list_of_weights = [{} for _ in range(12)]
for key in keys:
    if key == 'encoder.layer.0.attention.self.query.weight': start = True
    if not start: continue
    a_pos = 1 + find_nth(key, '.', 3)
    if a_pos == 0: continue
    layer_index_start = int(key[1 + find_nth(key, '.', 2)])
    layer_index_start = 1 + find_nth(key, '.', 2)
    layer_index_end = find_nth(key, '.', 3)
    layer_index = int(key[layer_index_start:layer_index_end])
    list_of_weights[layer_index][key[a_pos:]] = model.state_dict()[key]

num_blocks = 6
d_model = 768
seq_len = 512
d_hidden = 3072
num_heads = 12
dropout = 0.1 # irrelevant for this model

print("<build.py>: saving 1st 6 encoders...")
encoder_1 = bert.Encoder_1(num_blocks, d_model, seq_len, d_hidden, num_heads, dropout, list_of_weights[0:6])
tf.saved_model.save(encoder_1, 'encoder_1')
print("<build.py>: saving 2nd 6 encoders...")
encoder_2 = bert.Encoder_2(num_blocks, d_model, seq_len, d_hidden, num_heads, dropout, list_of_weights[6:])
tf.saved_model.save(encoder_2, 'encoder_2')
print("<build.py>: finished saving encoders...")