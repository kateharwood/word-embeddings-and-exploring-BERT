import os
import argparse
import numpy as np
from numpy.lib.function_base import average
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Time the runtime
from datetime import datetime
startTime = datetime.now()


LABELS = ['F', 'T']



def get_wic_subset(data_dir):
	wic = []
	split = data_dir.strip().split('/')[-1]
	with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
		open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
		for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
			word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
			sentence1_word_index, sentence2_word_index = word_indices.split('-')
			label = LABELS.index(label_line.strip())
			wic.append({
				'word': word,
				'sentence1_word_index': int(sentence1_word_index),
				'sentence2_word_index': int(sentence2_word_index),
				'sentence1_words': sentence1.split(' '),
				'sentence2_words': sentence2.split(' '),
				'label': label
			})
	return wic


if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Train a classifier to recognize words in context (WiC).'
	)
	parser.add_argument(
		'--train-dir',
		dest='train_dir',
		required=True,
		help='The absolute path to the directory containing the WiC train files.'
	)
	parser.add_argument(
		'--eval-dir',
		dest='eval_dir',
		required=True,
		help='The absolute path to the directory containing the WiC eval files.'
	)
	# Write your predictions (F or T, separated by newlines) for each evaluation
	# example to out_file in the same order as you find them in eval_dir.  For example:
	# F
	# F
	# T
	# where each row is the prediction for the corresponding line in eval_dir.
	parser.add_argument(
		'--out-file',
		dest='out_file',
		required=True,
		help='The absolute path to the file where evaluation predictions will be written.'
	)
	args = parser.parse_args()


# Initialize training and validation data
y_train = [d['label'] for d in get_wic_subset(args.train_dir)]
y_val = [d['label'] for d in get_wic_subset(args.eval_dir)]

X_train_sentences_1 = [' '.join(d['sentence1_words']) for d in get_wic_subset(args.train_dir)]
X_train_sentences_2 = [' '.join(d['sentence2_words']) for d in get_wic_subset(args.train_dir)]
X_val_sentences_1 = [' '.join(d['sentence1_words']) for d in get_wic_subset(args.eval_dir)]
X_val_sentences_2 = [' '.join(d['sentence2_words']) for d in get_wic_subset(args.eval_dir)]

# Initialize Bert tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertModel.from_pretrained('bert-base-cased')

# Take model out of train mode that uses dropout
bert_model = bert_model.eval()

# Run training sentences through bert
def run_bert(sentences):
	bert_output = []
	for i in range(len(sentences)):
		tokenized_sentence = tokenizer(sentences[i], return_tensors='pt')
		with torch.no_grad():
			output = bert_model(**tokenized_sentence)
		mean_output = torch.mean(output.last_hidden_state, dim=1)
		bert_output.append(mean_output)
	return bert_output

bert_train_output_sentences_1 = run_bert(X_train_sentences_1)
bert_train_output_sentences_2 = run_bert(X_train_sentences_2)
bert_val_output_sentences_1 = run_bert(X_val_sentences_1)
bert_val_output_sentences_2 = run_bert(X_val_sentences_2)

# Get rid of extra dimension
bert_train_output_sentences_1 = [output.squeeze() for output in bert_train_output_sentences_1]
bert_train_output_sentences_2 = [output.squeeze() for output in bert_train_output_sentences_2]
bert_val_output_sentences_1 = [output.squeeze() for output in bert_val_output_sentences_1]
bert_val_output_sentences_2 = [output.squeeze() for output in bert_val_output_sentences_2]


# Take cosine similarity between the bert output for sentences1 and sentences2 to create the input for the linear classifier
X_train = []
for i in range(len(bert_train_output_sentences_1)):
	cos_sim = torch.nn.functional.cosine_similarity(bert_train_output_sentences_1[i], bert_train_output_sentences_2[i], dim=0)
	X_train.append(cos_sim.item())

X_val = []
for i in range(len(bert_val_output_sentences_1)):
	cos_sim = torch.nn.functional.cosine_similarity(bert_val_output_sentences_1[i], bert_val_output_sentences_2[i], dim=0)
	X_val.append(cos_sim.item())

# Reshape data for input to logistic regression
X_train = np.array(X_train).reshape(-1, 1)
X_val = np.array(X_val).reshape(-1, 1)

# Train classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Test classifier
predicted = classifier.predict(X_val)
print(classification_report(y_val, predicted))
print(confusion_matrix(y_val, predicted))

# Write to output file
output_file = open(str(args.out_file), "w")
for prediction in predicted:
	if prediction:
		output_pred = "T"
	else:
		output_pred = "F"
	output_file.write(output_pred + "\n")


# Output the runtime
print(datetime.now() - startTime)