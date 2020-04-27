import torch

class BertClassifier(torch.nn.Module):

	def __init__(self, bert, h_dim, o_dim, n_layers, dropout=0.1, bidirectional=False):

		super(BertClassifier, self).__init__()

		self.bert_model = bert
		emb_dim = bert.config.to_dict()["hidden_size"]

		self.rnn_layer = torch.nn.GRU(
			emb_dim, h_dim,
			num_layers=n_layers, bidirectional=bidirectional,
			batch_first=True, dropout=(0 if n_layers < 2 else dropout)
		)
		self.out_layer = torch.nn.Linear(h_dim*2 if bidirectional else h_dim, o_dim)
		self.dropout_layer = torch.nn.Dropout(dropout)


	def forward(self, text):

		# text.shape = [batch_size, sent_len]

		with torch.no_grad():
			embeddings = self.bert_model(text)[0]

		_, h = self.rnn_layer(embeddings)

		if self.rnn_layer.bidirectional:
			h = self.dropout_layer(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))
		else:
			h = self.dropout_layer(h[-1, :, :])

		output = torch.nn.functional.softmax(self.out_layer(h), dim=1)

		return output
