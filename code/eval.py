import argparse
import pprint
import torch
import utils
import json

import numpy as np

from transformers import BertTokenizer, BertModel
from model import BertClassifier

def load_args():

	parser = argparse.ArgumentParser(
		prog="BERT_CLS",
		description="Bert classifier"
	)

	parser.add_argument("--data_path", type=str, default='', help="data path, contains and `test.json`")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size of data")
	parser.add_argument("--device_id", type=int, default=-1, help="gpu index")
	parser.add_argument("--save_id", type=str, default="01", help="model save id")

	args = parser.parse_args()

	return args


def eval():

	printer = pprint.PrettyPrinter(indent=4)
	args = load_args()
	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	with open("../ckpt/{}/mconf.json".format(args.save_id), 'r') as f:
		mconf = json.load(f)

	if args.device_id >= 0 and torch.cuda.is_available():
		device = torch.device("cuda:{}".format(args.device_id))
	else:
		device = torch.device("cpu")

	print("[DEVICE INFO] using {}".format(device))
	
	test_iter = utils.load_data(args.data_path, args.batch_size, tokenizer, "test.json", device)

	bert = BertModel.from_pretrained("bert-base-uncased")
	model = BertClassifier(
		bert, h_dim=mconf["h_dim"], o_dim=mconf["n_cls"], n_layers=mconf["n_layers"],
		dropout=mconf["dropout"], bidirectional=mconf["bidirectional"]
	)


	model.load_state_dict(torch.load("../ckpt/{}/model.pt".format(args.save_id), map_location=device))
	print("loaded model from ../ckpt/{}/model.pt".format(args.save_id))
	model.eval()
	model.to(device)

	val_loss, val_acc = 0., 0.
	loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
	loss_fn = loss_fn.to(device)
	c = 0

	for batch in test_iter:
		logits = model(batch.text)
		preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
		loss = loss_fn(logits, batch.label)
		acc = utils.calc_acc(preds, batch.label)

		val_loss += loss.item()
		val_acc += acc.item()

	val_loss /= len(test_iter)
	val_acc /= len(test_iter)

	print("-------")
	print(f"[eval] val_loss = {val_loss:.3f}, val_acc = {100*val_acc:.2f}%")


if __name__ == "__main__":

	eval()
