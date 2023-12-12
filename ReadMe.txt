To compile everything and generate dlcs for model, use build.sh
To run model, use run.py

File Descriptions:
HuggingFaceOutput.py - Runnning Purely HuggingFace's bert Model to confirm proper output

bert.py - Graph-mode TF models that are DLC-compatiable
run.py - Launches tokenzier, computes embedding in eager-mode TensorFlow (can't to this on the phone), writes embedding output to storage, launches 1st dlc (6 encoders), launches 2nd dlc (6 encoders)
tokenizer.cpp - Tokenizes a text, majority of code taken from https://gist.github.com/luistung/ace4888cf5fd1bad07844021cb2c7ecf
vocab.txt - Used by tokenizer to map words to numbers
