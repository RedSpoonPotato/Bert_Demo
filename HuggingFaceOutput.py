from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Let this work please, I have other stuff to do."
padded_tokens = tokenizer(text, padding="max_length", max_length=512, truncation=True, return_tensors='pt')
model = BertModel.from_pretrained("bert-base-uncased")

# print(padded_tokens)

output = model(**padded_tokens)
print("HuggingFace Bert Model output:", output['last_hidden_state'])