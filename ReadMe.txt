Please ensure ALL requirements are met before attempting to compile

To compile everything and generate dlcs for model, use build.sh
To run model, use run.py

Notable Requirements:
Over 16GB of Memory
Ubuntu 20.02
Python 3.8
TensorFlow 3.11
Qualcomm Neural Processing SDK (latest version)
python transformers package
utf8proc source code (https://github.com/JuliaStrings/utf8proc/tree/master)
  - Please rename folder to "utf8proc-master" instead of "utf8proc" and place in working directory
boost library (sudo apt-get install libboost-all-dev)
g++ 9.4.1

File Descriptions:
HuggingFaceOutput.py - Runnning Purely HuggingFace's bert Model to confirm proper output
bert.py - Graph-mode TF models that are DLC-compatiable
enc_1_inputs.txt - used by run.py to read binary file inputs into dlc model (same for enc_2_inputs.txt)
run.py - Launches tokenzier, computes embedding in eager-mode TensorFlow (can't to this on the phone), writes embedding output to storage, launches 1st dlc (6 encoders), launches 2nd dlc (6 encoders)
tokenizer.cpp - Tokenizes a text, majority of code taken from https://gist.github.com/luistung/ace4888cf5fd1bad07844021cb2c7ecf
vocab.txt - Used by tokenizer to map words to numbers
