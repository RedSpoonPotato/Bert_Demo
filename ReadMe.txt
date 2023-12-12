The code works, but I don't recommend running run.py without proper setup, as it will just write a few files than fail.
The code is not meant to build the executables or the dlcs, but to show off key aspects of what we have.

bert.py - Graph-mode TF models that are DLC-compatiable
tokenizer.cpp - Tokenizes a text, majority of code taken from https://gist.github.com/luistung/ace4888cf5fd1bad07844021cb2c7ecf
run.py - Launches tokenzier, computes embedding in eager-mode TensorFlow (can't to this on the phone), writes embedding output to storage, launches 1st dlc (6 encoders), launches 2nd dlc (6 encoders)