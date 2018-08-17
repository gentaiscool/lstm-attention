"""
Program for
ATTENTION-BASED LSTM FOR PSYCHOLOGICAL STRESS DETECTION FROM SPOKEN LANGUAGE USING DISTANT SUPERVISION
IEEE ICASSP 2018 (Genta Indra Winata, Onno Pepijn Kampman, Pascale Fung)

To run the model
# e.g. CUDA_VISIBLE_DEVICES=0 python main.py --input_data_path="../../dataset" --model_type=LSTM --is_attention=True --is_finetune=True --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=35 --validation_split=0.2
with custom embedding
# e.g. CUDA_VISIBLE_DEVICES=0 python main.py --input_data_path="../../dataset" --model_type=LSTM --is_attention=True --is_finetune=False --hidden_units=64 --num_layers=1 --is_bidirectional=True --max_sequence_length=35 --word_embedding_path="../embedding/emotion_embedding_size_50/emotion_embedding.pkl" --vocab_path="../embedding/emotion_embedding_size_50/vocab.pkl" --skip_preprocess=False --validation_split=0.2

Tips:
1. Skip the preprocess, if you are using the same embedding vectors
--skip-preprocess=True
"""

############################################## 
# LOAD LIBRARIES
##############################################
# PYTHON LIBS
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import datetime
from copy import deepcopy
from scipy import spatial
from tqdm import tqdm

# TRAINING MODULES
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
from IPython.display import SVG
import numpy
import warnings
from sklearn import metrics
from copy import deepcopy
import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session

import lstm_models as lstm_models
import svm_models as svm_models
import util as util
import evaluation as eval

def train_svm():
	model = svm_models.svm()

	interview_train_data, interview_train_labels, interview_test_data, interview_test_labels = util.load_interview_data(input_data_path)
	interview_train_data, interview_train_labels = util.generate_embedding(interview_train_data, interview_train_labels, "basic_word2vec300", 300)
	interview_valid_data, interview_valid_labels = util.generate_embedding(interview_valid_data, interview_valid_labels, "basic_word2vec300", 300)
	interview_test_data, interview_test_labels = util.generate_embedding(interview_test_data, interview_test_labels, "basic_word2vec300", 300)

	train_arr = []
	train_arr_label = []
	valid_arr = []
	valid_arr_label = []
	test_arr = []
	test_arr_label = []

	for i in range(len(interview_train_data)):
		total_vec = interview_train_data[i][0]
		for j in range(1, len(interview_train_data[i])):
			total_vec = total_vec + interview_train_data[i][j]
		avg_vec = total_vec / len(interview_train_data[i])
		avg_vec = avg_vec.tolist()
		train_arr.append(avg_vec)
		if int(interview_train_labels[i][1]) == 1:
			train_arr_label.append(1)
		else:
			train_arr_label.append(0)
			
	for i in range(len(interview_valid_data)):
		total_vec = interview_valid_data[i][0]
		for j in range(1, len(interview_valid_data[i])):
			total_vec = total_vec + interview_valid_data[i][j]
		avg_vec = total_vec / len(interview_valid_data[i])
		avg_vec = avg_vec.tolist()
		train_valid.append(avg_vec)
		if int(interview_valid_labels[i][1]) == 1:
			train_valid_label.append(1)
		else:
			train_valid_label.append(0)
		
	for i in range(len(interview_test_data)):
		total_vec = interview_test_data[i][0]
		for j in range(1, len(interview_test_data[i])):
			total_vec = total_vec + interview_test_data[i][j]
		avg_vec = total_vec / len(interview_test_data[i])
		avg_vec = avg_vec.tolist()
		
		test_arr.append(avg_vec)
		if int(interview_test_labels[i][1]) == 1:
			test_arr_label.append(1)
		else:
			test_arr_label.append(0)

	print("train data size:", len(train_arr))
	print("train labels size:", len(train_arr_label))
	print("test data size:", len(test_arr))
	print("test labels size:", len(test_arr_label))

	svm_models.train(model, interview_train_data, interview_train_labels, interview_test_data, interview_test_labels)

def train_lstm(word_to_idx, idx_to_word, vocab, interview_train_data, interview_train_labels, interview_test_data, interview_test_labels, twitter_stress_data, twitter_relax_data, model_type, hidden_units, num_layers, max_sequnce_length, is_attention, is_finetune, is_bidirectional, word_embedding, is_custom_embedding, validation_split):
	batch_size = 128

	num_initialization_epochs = 2
	num_epochs = 10

	if not is_custom_embedding:
		print("> Train LSTM with trainable word embedding matrix")
		model = lstm_models.lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional)
	else:
		print("> Train LSTM with pretrained word embedding matrix")
		model = lstm_models.lstm_word_embedding(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional, word_embedding)

	print("interview_train_data:", interview_train_data.shape)
	print("interview_test_data:", interview_test_data.shape)

	if is_finetune:
		print("twitter_stress_data:", twitter_stress_data.shape)
		print("twitter_relax_data:", twitter_relax_data.shape)
		
		print('>>> Starting initialization...')
		for i in range(num_initialization_epochs):
			print("> epoch:", (i + 1), "/", num_initialization_epochs)
			num_samples = 49000
			
			twitter_train_data = np.zeros((num_samples*2, max_sequence_length),dtype=np.float32)
			twitter_train_labels = np.zeros((num_samples*2, 2),dtype=np.float32)

			arr = np.random.choice(twitter_stress_data.shape[0], num_samples, replace=False)
			arr2 = np.random.choice(twitter_relax_data.shape[0], num_samples, replace=False)

			for j in range(num_samples):
				twitter_train_data[j] = twitter_stress_data[arr[j]]
				twitter_train_labels[j] = 0
				twitter_train_labels[j][1] = 1
				twitter_train_data[num_samples + j] = twitter_relax_data[arr2[j]]
				twitter_train_labels[num_samples + j] = 0
				twitter_train_labels[num_samples + j][0] = 1

			# shuffle the idx
			shuffle_idx = np.arange(len(twitter_train_data))
			np.random.shuffle(shuffle_idx)
			twitter_train_data = twitter_train_data[shuffle_idx]
			twitter_train_labels = twitter_train_labels[shuffle_idx]
			
			model.fit(twitter_train_data, twitter_train_labels, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
		
	#TESTING
	print('\n>>> Testing...')
	test(model, interview_test_data, interview_test_labels, False)

	print('\n>>> Starting training/fine-tuning with interview...')
	history = model.fit(interview_train_data, interview_train_labels, validation_split=validation_split, batch_size=batch_size, epochs=num_epochs, shuffle=True, verbose=1)
		
	# TESTING
	print('\n>>> Testing...')
	test(model, interview_test_data, interview_test_labels, False)

	print('Training finished ', str(datetime.datetime.now()))
	return model


def test(test_model, test_data, test_labels, show_mistake=False):
	test_predictions = test_model.predict(test_data, verbose=0)
	
	# PRINT WRONG PREDICTIONS
	if show_mistake:
		for i in range(len(test_predictions)):
			stress_probability = test_predictions[i][1]
			score = abs(test_labels[i][1] - stress_probability)
			if score > 0:
				seq = ""
				for j in range(len(test_data[i])):
					seq += idx_to_word[test_data[i][j]].strip() + " "
				print(seq, ",", score, ",", test_labels[i][1], stress_probability)
	
	# TEST PERFORMANCE
	res_accu = eval.accuracy(test_predictions, test_labels)
	res_f1 = eval.fscore(test_predictions, test_labels)
	res_recall = eval.recall(test_predictions, test_labels)
	res_precision = eval.precision(test_predictions, test_labels)
	print('Test Accuracy: %.3f' % res_accu)
	print('Test F1-score: %.3f' % res_f1)
	print('Test Recall: %.3f' % res_recall)
	print('Test Precision: %.3f' % res_precision)
	
	return res_accu, res_f1, res_recall, res_precision

if __name__ == '__main__':

	# LIMIT TENSORFLOW MEMORY USAGE
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.7
	set_session(tf.Session(config=config))

	############################################## 
	# PARSE COMMAND OPTIONS
	##############################################

	from optparse import OptionParser
	parser = OptionParser()
	parser.add_option("--model_type", dest="model_type", default="LSTM", help="default=LSTM, LSTM/SVM", metavar="FILE")
	parser.add_option("--is_attention", dest="is_attention", default=True, help="default=True, use attention mechanism", metavar="FILE")
	parser.add_option("--is_finetune", dest="is_finetune", default=True, help="default=True, use fine tuning", metavar="FILE")
	parser.add_option("--hidden_units", dest="hidden_units", default=64, help="default=64, number of hidden units", metavar="FILE")
	parser.add_option("--num_layers", dest="num_layers", default=1, help="default=1, number of layers (only for LSTM/BLSTM models)", metavar="FILE")
	parser.add_option("--is_bidirectional", dest="is_bidirectional", default=True, help="default=True, use BLSTM", metavar="FILE")
	parser.add_option("--input_data_path", dest="input_data_path", default="../../dataset", help="input datapath", metavar="FILE")
	parser.add_option("--max_sequence_length", dest="max_sequence_length", default=35, help="max sequence length", metavar="FILE")
	parser.add_option("--word_embedding_path", dest="word_embedding_path", default=None, help="word embedding in numpy array format path", metavar="FILE")
	parser.add_option("--vocab_path", dest="vocab_path", default=None, help="vocab in numpy array path", metavar="FILE")
	parser.add_option("--skip_preprocess", dest="skip_preprocess", default=True, help="load preprocess files, skip preprocessing part", metavar="FILE")
	parser.add_option("--validation_split", dest="validation_split", default=0.2, help="validation split", metavar="FILE")

	(options, args) = parser.parse_args()
	print("options", options)
	print("args", args)

	model_type = options.model_type
	is_attention = options.is_attention == "True"
	is_finetune = options.is_finetune == "True"
	hidden_units = int(options.hidden_units)
	num_layers = int(options.num_layers)
	is_bidirectional = options.is_bidirectional == "True"
	input_data_path = options.input_data_path
	max_sequence_length = int(options.max_sequence_length)
	skip_preprocess = options.skip_preprocess == "True"
	validation_split = int(options.validation_split)

	word_embedding = None
	word_embedding_path = None
	vocab_path = None
	is_custom_embedding = False

	if not options.word_embedding_path == None:
		word_embedding_path = options.word_embedding_path

		if not options.vocab_path == None:
			vocab_path = options.vocab_path
			is_custom_embedding = True

	print("\n###################################")
	print("              SUMMARY          ")
	print("###################################")
	print("model_type\t\t:", model_type)
	print("is_attention\t\t:", is_attention)
	print("is_finetune\t\t:", is_finetune)
	print("hidden_units\t\t:", hidden_units)
	print("num_layers\t\t:", num_layers)
	print("is_bidirectional\t:", is_bidirectional)
	print("input_data_path\t\t:", input_data_path)
	print("is_custom_embedding\t:", is_custom_embedding)
	print("word_embedding_path\t:", word_embedding_path)
	print("vocab_path\t\t:", vocab_path)
	print("skip_preprocess\t\t:", skip_preprocess)
	print("validation_split\t:", validation_split)
	print("")
	print("###################################\n")

	############################################## 
	# READ DATA
	##############################################

	# LOAD DATA IF POSSIBLE
	try:
		if not skip_preprocess:
			raise ValueError('>>> force to preprocess, force to go to except block')

		print(">>> load data")
		interview_train_data = np.load(input_data_path + "/preprocessed/interview_train_data.npy")
		interview_train_labels = np.load(input_data_path + "/preprocessed/interview_train_labels.npy")
		interview_test_data = np.load(input_data_path + "/preprocessed/interview_test_data.npy")
		interview_test_labels = np.load(input_data_path + "/preprocessed/interview_test_labels.npy")
		twitter_stress_data = np.load(input_data_path + "/preprocessed/twitter_stress_data.npy")
		twitter_relax_data = np.load(input_data_path + "/preprocessed/twitter_relax_data.npy")

		word_to_idx = np.load(input_data_path + "/preprocessed/word_to_idx.npy")
		idx_to_word = np.load(input_data_path + "/preprocessed/idx_to_word.npy")
		vocab = np.load(input_data_path + "/preprocessed/vocab.npy")
	
		if is_custom_embedding:
			word_embedding = np.load(input_data_path + "/preprocessed/word_embedding.npy")
	except:
		if skip_preprocess:
			print(">>> can't load preprocessed data!")

		interview_train_data, interview_train_labels, interview_test_data, interview_test_labels = util.load_interview_data(input_data_path)
		twitter_stress_data, twitter_stress_labels, twitter_relax_data, twitter_relax_labels = util.load_preprocessed_twitter_data(input_data_path)

		############################################## 
		# PREPROCESS DATA
		##############################################
		all_interview_data = []
		all_twitter_data = []
		all_data = []

		print(">>> preprocess data")
		for i in range(len(interview_train_data)):
			interview_train_data[i] = util.preprocess_lowercase_negation(interview_train_data[i])
			all_interview_data.append(interview_train_data[i])
			all_data.append(interview_train_data[i])
			
		for i in range(len(interview_test_data)):
			interview_test_data[i] = util.preprocess_lowercase_negation(interview_test_data[i])
			all_interview_data.append(interview_test_data[i])
			all_data.append(interview_test_data[i])

		for i in range(len(twitter_stress_data)):
			twitter_stress_data[i] = util.preprocess_lowercase_negation(twitter_stress_data[i])
			all_twitter_data.append(twitter_stress_data[i])
			all_data.append(twitter_stress_data[i])
			
		for i in range(len(twitter_relax_data)):
			twitter_relax_data[i] = util.preprocess_lowercase_negation(twitter_relax_data[i])
			all_twitter_data.append(twitter_relax_data[i])
			all_data.append(twitter_relax_data[i])

		print("interview dataset")
		util.check_data(all_interview_data)
		print("twitter dataset")
		util.check_data(all_twitter_data)

		print(">>> generate vocabulary")
		if is_custom_embedding:
			print(">>> load vocab and embedding pickle data")
			vocab = list(pickle.load(open(vocab_path,"rb"), encoding='latin1').values())
			word_embedding = pickle.load(open(word_embedding_path,"rb"), encoding='latin1')

			word_to_idx, idx_to_word, vocab, word_embedding = util.generate_vocab_with_custom_embedding(vocab, word_embedding)
		else:
			print(">>> generate vocabulary from all data")

			word_to_idx, idx_to_word, vocab = util.generate_vocab(all_data)

		# generate word idx from vocabulary
		print(">>> generate word idx")
		interview_train_data, interview_train_labels = util.generate_word_index(interview_train_data, interview_train_labels, word_to_idx, idx_to_word, vocab, max_sequence_length)
		interview_test_data, interview_test_labels = util.generate_word_index(interview_test_data, interview_test_labels, word_to_idx, idx_to_word, vocab, max_sequence_length)

		# oversampling
		print(">>> oversampling")
		print(np.sum(interview_train_labels[:,0]), np.sum(interview_train_labels[:,1]))
		print(np.sum(interview_test_labels[:,0]), np.sum(interview_test_labels[:,1]))

		interview_stress_data = np.concatenate([interview_train_data[:653], interview_test_data[:160]])
		interview_stress_labels = np.concatenate([interview_train_labels[:653], interview_test_labels[:160]])
		interview_relax_data = np.concatenate([interview_train_data[653:], interview_test_data[160:]])
		interview_relax_labels = np.concatenate([interview_train_labels[653:], interview_test_labels[160:]])

		print("interview stress", interview_stress_data.shape, interview_stress_labels.shape)
		print("interview relax", interview_relax_data.shape, interview_relax_labels.shape)

		print(np.sum(interview_stress_labels[:,0]), np.sum(interview_stress_labels[:,1]))
		print(np.sum(interview_relax_labels[:,0]), np.sum(interview_relax_labels[:,1]))

		interview_train_data = np.concatenate([interview_train_data, interview_train_data[:653,:]])
		interview_train_labels = np.concatenate([interview_train_labels, interview_train_labels[:653,:]])

		print("interview train", interview_train_data.shape, interview_train_labels.shape)
		print("interview test", interview_test_data.shape, interview_test_labels.shape)

		print(np.sum(interview_train_labels[:,0]), np.sum(interview_train_labels[:,1]))
		print(np.sum(interview_test_labels[:,0]), np.sum(interview_test_labels[:,1]))

		twitter_stress_data, twitter_stress_labels = util.generate_word_index(twitter_stress_data, twitter_stress_labels, word_to_idx, idx_to_word, vocab, max_sequence_length)
		twitter_relax_data, twitter_relax_labels = util.generate_word_index(twitter_relax_data, twitter_relax_labels, word_to_idx, idx_to_word, vocab, max_sequence_length)

		## split training, validation, test
		twitter_train_stress_data = twitter_stress_data[5000:54515]
		twitter_valid_stress_data = twitter_stress_data[:5000]
		twitter_test_stress_data = twitter_stress_data[54515:59515]

		twitter_train_stress_labels = twitter_stress_labels[5000:54515]
		twitter_valid_stress_labels = twitter_stress_labels[:5000]
		twitter_test_stress_labels = twitter_stress_labels[54515:59515]

		print("twitter stress",twitter_stress_data.shape, twitter_stress_labels.shape)
		print("twitter relax",twitter_relax_data.shape, twitter_relax_labels.shape)

		twitter_train_relax_data = twitter_relax_data[5000:298713]
		twitter_valid_relax_data = twitter_relax_data[:5000]
		twitter_test_relax_data = twitter_relax_data[298713:303713]

		twitter_train_relax_labels = twitter_relax_labels[5000:298713]
		twitter_valid_relax_labels = twitter_relax_labels[:5000]
		twitter_test_relax_labels = twitter_relax_labels[298713:303713]

		twitter_valid_data = np.concatenate([twitter_valid_stress_data, twitter_valid_relax_data])
		twitter_valid_labels = np.concatenate([twitter_valid_stress_labels, twitter_valid_relax_labels])

		twitter_test_data = np.concatenate([twitter_test_stress_data, twitter_test_relax_data])
		twitter_test_labels = np.concatenate([twitter_test_stress_labels, twitter_test_relax_labels])

		print("twitter_train_stress_data", twitter_train_stress_data.shape, twitter_train_stress_labels.shape)
		print("twitter_train_relax_data", twitter_train_relax_data.shape, twitter_train_relax_labels.shape)
		print("twitter_valid data", twitter_valid_data.shape, twitter_valid_labels.shape)
		print("twitter_test data", twitter_test_data.shape, twitter_test_labels.shape)

		print(">>> save data")
		np.save(input_data_path + "/preprocessed/interview_train_data", interview_train_data)
		np.save(input_data_path + "/preprocessed/interview_train_labels", interview_train_labels)
		np.save(input_data_path + "/preprocessed/interview_test_data", interview_test_data)
		np.save(input_data_path + "/preprocessed/interview_test_labels", interview_test_labels)
		np.save(input_data_path + "/preprocessed/twitter_stress_data", twitter_stress_data)
		np.save(input_data_path + "/preprocessed/twitter_relax_data", twitter_relax_data)

		np.save(input_data_path + "/preprocessed/word_to_idx", word_to_idx)
		np.save(input_data_path + "/preprocessed/idx_to_word", idx_to_word)
		np.save(input_data_path + "/preprocessed/vocab", vocab)

		if is_custom_embedding:
			np.save(input_data_path + "/preprocessed/word_embedding", word_embedding)

	############################################## 
	# TRAIN AND TEST MODEL
	##############################################
	train_lstm(word_to_idx, idx_to_word, vocab, interview_train_data, interview_train_labels, interview_test_data, interview_test_labels, twitter_stress_data, twitter_relax_data, model_type, hidden_units, num_layers, max_sequence_length, is_attention, is_finetune, is_bidirectional, word_embedding, is_custom_embedding, validation_split)