import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from matplotlib import pyplot

conll_data = torch.load("data/conll.pt")
# semeval_data = torch.load("data/semeval.pt")
num_hidden_states = len(conll_data['train'][0]['hidden_states'])

# Build training data one word at a time
# Finds all the tokens in bert that split words and average them
def build_feature_data(data_type="train", hidden_layer_num=0):
    features = []
    for i in range(len(conll_data[data_type])):
        word_token_indices = conll_data[data_type][i]['word_token_indices']
        hidden_state = conll_data[data_type][i]['hidden_states'][hidden_layer_num]

        # Add the embeddings in the hidden state that correspond to the word_token_indices,
        # averaging over the embeddings that correspond to more than one word.
        # This does not include BERT's SOS and EOS tokens.
        for indices in word_token_indices:
            if len(indices) > 1:
                average_bert_reps = torch.mean(hidden_state[indices], dim=0)
                features.append(average_bert_reps)
            else:
                features.append(hidden_state[indices])
    print("Layer ")
    print(hidden_layer_num)
    return features

def build_label_data(data_type, task):
    labels = []
    for i in range(len(conll_data[data_type])):
        labels.append(conll_data[data_type][i][task])
    labels = [label for label_list in labels for label in label_list]
    return labels

# Loop through layers and classify
# Aggregate F1 scores from each layer
def get_f1_scores(task):
    f1_scores = []
    y_train = build_label_data('train', task)
    y_val = build_label_data('validation', task)
    for i in range(num_hidden_states):
        X_train = build_feature_data('train', i)
        X_val = build_feature_data('validation', i)
        X_train = [example.numpy().squeeze() for example in X_train]
        X_val = [example.numpy().squeeze() for example in X_val]

        classifier = LogisticRegression(solver='liblinear')
        classifier.fit(X_train, y_train)
        predicted = classifier.predict(X_val)
        f1 = f1_score(y_val, predicted, average='macro')
        f1_scores.append(f1)
    return f1_scores


pos_f1_scores = get_f1_scores('pos_labels')
ner_f1_scores = get_f1_scores('ner_labels')

print(pos_f1_scores)
print(ner_f1_scores)

layers = list(range(num_hidden_states))
pyplot.plot(layers, pos_f1_scores, label="POS")
pyplot.plot(layers, ner_f1_scores, label="NER")
pyplot.xlabel("Layer Num")
pyplot.ylabel("Macro F1 Score")
pyplot.title('Macro F1 Scores by BERT Layer')
pyplot.legend()

pyplot.show()