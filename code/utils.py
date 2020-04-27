import torch

from torchtext import data

def load_data(path, batch_size, tokenizer, name, device):

	init_token_idx = tokenizer.cls_token_id
	eos_token_idx = tokenizer.sep_token_id
	pad_token_idx = tokenizer.pad_token_id
	unk_token_idx = tokenizer.unk_token_id

	max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]

	def tokenize_and_cut(sentence):
		
		# tokens = tokenizer.tokenize(sentence)
		tokens = sentence.strip().split()
		tokens = tokens[:(max_input_length-2)]

		return tokens


	TEXT = data.Field(
		batch_first=True, use_vocab=False, 
		tokenize=tokenize_and_cut, 
		preprocessing=tokenizer.convert_tokens_to_ids, 
		init_token=init_token_idx, 
		eos_token=eos_token_idx, 
		pad_token=pad_token_idx, 
		unk_token=unk_token_idx
	)
	LABEL = data.LabelField(dtype=torch.long, use_vocab=False)

	fields = {"text": ("text", TEXT), "label": ("label", LABEL)}

	d = data.TabularDataset.splits(
		path=path, 
		train=name, format="json", 
		fields=fields
	)[0]
	iterator, _ = data.BucketIterator.splits((d, d), batch_size=batch_size, device=device)

	return iterator

def calc_acc(preds, label):

	return (preds == label).float().sum() / len(preds)
