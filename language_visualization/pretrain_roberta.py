import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, AutoTokenizer, AutoModelWithLMHead, AutoModel
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# pretrained_model = 'bert-base-uncased'
pretrained_model = 'roberta-base'
# figure_num_points = [1000, 2000, 3000, 4000, 5000]
figure_num_points = [3000]
num_points = figure_num_points[-1]
data_name = 'training_instructions'
data_file = './data/{}.txt'.format(data_name)
device = 'cuda:0'

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModel.from_pretrained(pretrained_model)
model.eval()
model = model.to(device)


data_file_fd = open(data_file)
representations = []
sentences = []
idx = 0
for line in data_file_fd:
	tokenized_text = tokenizer.tokenize(line)
	indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
	tokens_tensor = torch.tensor([indexed_tokens]).to(device)
	with torch.no_grad():
		representation = model(tokens_tensor)[1]
		representations.append(representation.to('cpu'))

	if idx < 100:
		sentences.append(line.strip('\n'))

	idx += 1
	if idx >= num_points:
		break
	if idx % 500 == 0:
		print(idx)
data_file_fd.close()
representations = np.concatenate(representations)
representations = TSNE(n_components=2).fit_transform(representations)
print(representations.shape)

kmeans = KMeans(n_clusters=3, random_state=0).fit(representations)
print(kmeans.labels_.shape)

clusters = {
	0: [],
	1: [],
	2: [],
}
for num_points_on_figure in figure_num_points:
	plt.scatter(representations[:num_points_on_figure,0], representations[:num_points_on_figure,1])
	for i in range(0, 100):
		# plt.text(representations[i,0], representations[i,1], '{}'.format(i+1))
		plt.text(representations[i,0], representations[i,1], '{}'.format(kmeans.labels_[i]))
		clusters[kmeans.labels_[i]].append(sentences[i])
	plt.savefig('./{}_{}_{}_nums.png'.format(pretrained_model, data_name, num_points_on_figure))
	plt.clf()

output_filename = './data/clustered.txt'
output_filename_fd = open(output_filename, 'w')
for i in range(3):
	output_filename_fd.write('cluster {}:\n'.format(i))
	for line in clusters[i]:
		output_filename_fd.write(line + '\n')
	output_filename_fd.write('\n')
output_filename_fd.close()
