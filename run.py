# Requirements:
# boost (i used 1.83) -> sudo apt-get install boost
# utf8proc-master -> https://github.com/JuliaStrings/utf8proc/tree/master
# python 3.8
# tensorflow (i used 2.13)
# transformers (from hugging-face)
# snpe (and everything else needed to run it)

# ***CHANGE THIS TO WHAT YOU LIKE***
user_input = "Let this work please, I have other stuff to do."

explicit = True

# Tokenizing
# description: Using a C++ script to tokenize sequence. 
# Most of C++ taken from: https://gist.github.com/luistung/ace4888cf5fd1bad07844021cb2c7ecf
import subprocess
tokenize_cmd = ['./tokenize.out', user_input]
tokenize = subprocess.run(tokenize_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if tokenize.returncode != 0:
    print("Tokenization Failed:", tokenize.stdout.decode(), "exiting...")
    exit()
token_seq = tokenize.stdout.decode()
print("Tokens:", token_seq)
token_seq = token_seq.split()
token_seq = [int(token) for token in token_seq] # tokens
# token_type_ids will be generated in the next section

# Embeddings
# description: Using tensorflow rather than snpe to compute embedding due to ram overhead in converting to graphmode and to dlc.
# additional comments: I think for a hypothetical final version, we would just implement a c++ implementation 
# as this layer is comprised mostly of a relatively small amount of look-up operations on a gigantic table that only needs 
# to happen once, so I am thinking it would be better to just store the look-up table in storage instead.
import tensorflow as tf
from transformers import BertModel
from bert import BertEmbedding
import struct
pre_trained_model = BertModel.from_pretrained("bert-base-uncased")
# model parameters
d_model = 768
seq_len = 512
vocab_size = 30522
max_seq_len = 512
embedding_params = {
    'embeddings.word_embeddings.weight':        pre_trained_model.state_dict()['embeddings.word_embeddings.weight'],
    'embeddings.position_embeddings.weight':    pre_trained_model.state_dict()['embeddings.position_embeddings.weight'],
    'embeddings.token_type_embeddings.weight':  pre_trained_model.state_dict()['embeddings.token_type_embeddings.weight'],
    'embeddings.LayerNorm.weight':              pre_trained_model.state_dict()['embeddings.LayerNorm.weight'],
    'embeddings.LayerNorm.bias':                pre_trained_model.state_dict()['embeddings.LayerNorm.bias']
}
# forcing model to run in eager-mode rather than graph-mode as it runs faster 
# (i think b/c graph-mode tries to load entire word_embeddings.weight tensor into memory all at once)
tf.config.run_functions_eagerly(True)
# model instantiation
embedding_layer = BertEmbedding(d_model, seq_len, vocab_size, max_seq_len, embedding_params)
# input pre-processing (creating token_type_ids and mask)
token_num = len(token_seq)
token_seq = tf.convert_to_tensor(token_seq, dtype='int32') # [token_num]
token_seq = tf.concat([token_seq, tf.zeros([seq_len-token_num], dtype='int32')], axis=0) # [512]
segments = tf.zeros([seq_len], dtype='int32') # [512]
# computation
embedding_out = embedding_layer(token_seq, segments) # [512, 768]
if explicit: 
    print('embedding output:', embedding_out)
# write result to file as binary data
print('Writing Embedding Layer Output (will take a minute or two)...')
file = open('embed_out.dat', 'wb')
for i in range(seq_len):
    if (i % 100 == 0): print("row # of 512:", i)
    row_data = struct.pack('f'*d_model, *embedding_out[i])
    file.write(row_data)
file.close()
print('Finished...')

# Encoder (1st Half)
# description: Running first 6 of 12 Transformer Encoders using SNPE 
# additional comments: Only running halve as trying to convert entire 12-blocks Bert model requires too much ram
# generating mask:
mask = tf.concat([tf.ones([token_num]), tf.zeros([seq_len-token_num])], axis=0) # [512]
# writing mask to file as binary data
print('Writing Mask...')
file = open('mask.dat', 'wb')
mask_data = struct.pack('f'*seq_len, *mask)
file.write(mask_data)
file.close()
print('Finished...')
# computation
print("Starting First 6 Encoders...")
run_encoder_1_cmd = ['snpe-net-run', '--container', 'encoder_1.dlc', '--input_list', 'enc_1_inputs.txt']
run_encoder_1 = subprocess.run(run_encoder_1_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if run_encoder_1.returncode != 0:
    print("Error in Running Encoder_1!")
    exit()

# Encoder (2nd Half)
# description: Running other halve of the 12 Transformer Encodeders using SNPE
print("Starting Last 6 Encoders...")
run_encoder_2_cmd = ['snpe-net-run', '--container', 'encoder_2.dlc', '--input_list', 'enc_2_inputs.txt']
run_encoder_2 = subprocess.run(run_encoder_2_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if run_encoder_2.returncode != 0:
    print("Error in Running Encoder_2!")
    exit()

# Finished
print("Finished!, Output located output/Result_0/Encoder_2:0.raw")