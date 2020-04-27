import argparse
import pprint
import torch
import utils
import os
import json

from transformers import BertTokenizer, BertModel
from model import BertClassifier

def load_args():

	parser = argparse.ArgumentParser(
		prog="BERT_CLS",
		description="Bert classifier"
	)

	parser.add_argument("--data_path", type=str, default='', help="data path, contains `train.json` and `test.json`")
	parser.add_argument("--epochs", type=int, default=20, help="training epochs")
	parser.add_argument("--epochs_per_val", type=int, default=5, help="epochs per evaluation")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size of data")
	parser.add_argument("--device_id", type=int, default=-1, help="gpu index")
	parser.add_argument("--save_id", type=str, default="01", help="model save id")

	parser.add_argument("--h_dim", type=int, default=128, help="hidden dimension of GRU")
	parser.add_argument("--n_cls", type=int, default=2, help="number of classes")
	parser.add_argument("--n_layers", type=int, default=2, help="number of layers")
	parser.add_argument("--bidirectional", action="store_true", help="whether to use bidirectional GRU")
	parser.add_argument("--dropout", type=float, default=0.25, help="dropout rate")

	args = parser.parse_args()

	return args


def train():

	printer = pprint.PrettyPrinter(indent=4)
	args = load_args()
	print(">>>>>>> Options <<<<<<<")
	printer.pprint(vars(args))

	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	if args.device_id >= 0 and torch.cuda.is_available():
		device = torch.device("cuda:{}".format(args.device_id))
	else:
		device = torch.device("cpu")

	print("[DEVICE INFO] using {}".format(device))

	# load data
	train_iter = utils.load_data(args.data_path, args.batch_size, tokenizer, "train.json", device)
	test_iter = utils.load_data(args.data_path, args.batch_size, tokenizer, "test.json", device)

	bert = BertModel.from_pretrained("bert-base-uncased")
	print("loaded bert with {} trainable parameters".format(
		sum(p.numel() for p in bert.parameters() if p.requires_grad)
	))

	model = BertClassifier(
		bert, h_dim=args.h_dim, o_dim=args.n_cls, n_layers=args.n_layers,
		dropout=args.dropout, bidirectional=args.bidirectional
	)
	for name, param in model.named_parameters():
		if name.startswith("bert"):
			param.requires_grad = False
	print("loaded model with {} trainable parameters".format(
		sum(p.numel() for p in model.parameters() if p.requires_grad)
	))

	optimizer = torch.optim.Adam(model.parameters())
	loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


	model = model.to(device)
	loss_fn = loss_fn.to(device)

	best_acc = 0.

	for epoch in range(args.epochs):
		epoch_loss = 0.
		epoch_acc = 0.
		model.train()
		for batch in train_iter:
			optimizer.zero_grad()
			logits = model(batch.text)
			preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
			loss = loss_fn(logits, batch.label)
			acc = utils.calc_acc(preds, batch.label)
			
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()
			epoch_acc += acc.item()

		epoch_loss /= len(train_iter)
		epoch_acc /= len(train_iter)
		print(f"[epoch {epoch+1:03}] train_loss = {epoch_loss:.3f}, train_acc = {100*epoch_acc:.2f}%")
		
		if (epoch + 1) % args.epochs_per_val == 0:
			val_epoch_loss = 0.
			val_epoch_acc = 0.
			model.eval()
			for batch in test_iter:
				logits = model(batch.text).squeeze(1)
				preds = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
				loss = loss_fn(logits, batch.label)
				acc = utils.calc_acc(preds, batch.label)

				val_epoch_loss += loss.item()
				val_epoch_acc += acc.item()

			val_epoch_loss /= len(test_iter)
			val_epoch_acc /= len(test_iter)
			print("-------")
			print(f"[val] val_loss = {val_epoch_loss:.3f}, val_acc = {100*val_epoch_acc:.2f}%")
			print("-------")
			if val_epoch_acc > best_acc:
				best_acc = val_epoch_acc
				if not os.path.exists("../ckpt/{}".format(args.save_id)):
					os.makedirs("../ckpt/{}".format(args.save_id))
				torch.save(model.state_dict(), "../ckpt/{}/model.pt".format(args.save_id))

	mconf = {
		"h_dim": args.h_dim,
		"n_cls": args.n_cls,
		"n_layers": args.n_layers,
		"bidirectional": args.bidirectional,
		"dropout": args.dropout
	}
	with open("../ckpt/{}/mconf.json".format(args.save_id), 'w') as f:
		json.dump(mconf, f)
		print("saved model config to ../ckpt/{}/mconf.json".format(args.save_id))



if __name__ == "__main__":

	train()


