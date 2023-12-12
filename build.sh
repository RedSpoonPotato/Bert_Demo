# recomendation: If using WSL increase the Max Memory Usage
# recomendation: If using WSL increase the Max Memory Usage

# install boost: sudo apt-get install libboost-all-dev

# Change the directory to exactly match locations 
# add 
source /opt/qcom/aistack/qnn/2.16.4.231110/bin/envsetup.sh
source /opt/qcom/aistack/snpe/2.12.0.230626/bin/envsetup.sh

# generate DLC
echo 'building TensorFlow verison of both encoder halves'
python build.py
echo 'Converting 1st 6 encoders to from TF to DLC, if this takes over 2 minutes, you need more memory'
snpe-tensorflow-to-dlc --input_network encoder_1 \
                       --input_dim "input" "1,1,512,768" --input_dim "mask" "1,1,1,512" \
                       --show_unconsumed_nodes \
                       --out_node "Encoder_1"  --output_path encoder_1.dlc
echo 'Converting 2nd 6 encoders to from TF to DLC, if this takes over 2 minutes, you need more memory'
snpe-tensorflow-to-dlc --input_network encoder_2 \
                       --input_dim "input" "1,1,512,768" --input_dim "mask" "1,1,1,512" \
                       --show_unconsumed_nodes \
                       --out_node "Encoder_2"  --output_path encoder_2.dlc

g++ -Wall  tokenizer.cpp -o tokenize.out -L utf8proc-master/ -l:libutf8proc.a -l:libutf8proc.so.3