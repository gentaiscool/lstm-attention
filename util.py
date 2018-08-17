import numpy as np

############################################## 
# DATA READER AND LOADER
##############################################

def load_twitter_data(inputdatapath):
	twitter_stress_text = np.load(inputdatapath + "/twitter/twitter_stress_text.npy")
	twitter_relax_text = np.load(inputdatapath + "/twitter/twitter_relax_text.npy")

	# shuffle data
	shuffle_idx = np.arange(len(twitter_stress_text))
	np.random.shuffle(shuffle_idx)
	twitter_stress_text = twitter_stress_text[shuffle_idx]

	shuffle_idx = np.arange(len(twitter_relax_text))
	np.random.shuffle(shuffle_idx)
	twitter_relax_text = twitter_relax_text[shuffle_idx]

	return twitter_stress_text, twitter_relax_text

def load_preprocessed_twitter_data(inputdatapath):
	twitter_stress_data_all = np.load(inputdatapath + "/twitter/twitter_stress_text_random.npy")
	twitter_relax_data_all = np.load(inputdatapath + "/twitter/twitter_relax_text_random.npy")
	twitter_stress_label_all = []
	twitter_relax_label_all = []

	for i in range(twitter_stress_data_all.shape[0]):
		twitter_stress_label_all.append("1.0")

	for i in range(twitter_relax_data_all.shape[0]):
		twitter_relax_label_all.append("0.0")
		
	return twitter_stress_data_all, twitter_stress_label_all, twitter_relax_data_all, twitter_relax_label_all
	
def load_interview_data_with_valid(inputdatapath):
	interview_train_data, interview_train_labels = read_data(inputdatapath + "/interview/phase_all_transcript_label_per_sentence/train_text_genta.csv")
	interview_valid_data, interview_valid_labels = read_data(inputdatapath + "/interview/phase_all_transcript_label_per_sentence/valid_text_genta.csv")
	interview_test_data, interview_test_labels = read_data(inputdatapath + "/interview/phase_all_transcript_label_per_sentence/test_text_genta.csv")
	
	return interview_train_data, interview_train_labels, interview_valid_data, interview_valid_labels, interview_test_data, interview_test_labels

def load_interview_data(inputdatapath):
	interview_train_data, interview_train_labels = read_data(inputdatapath + "/interview/phase_all_transcript_label_per_sentence/train_text_genta.csv")
	interview_test_data, interview_test_labels = read_data(inputdatapath + "/interview/phase_all_transcript_label_per_sentence/test_text_genta.csv")
	
	return interview_train_data, interview_train_labels, interview_test_data, interview_test_labels

############################################## 
# GENERATE VOCABULARY, WORD TO IDX MAP
##############################################

def generate_vocab(data):
	# input = array of words
	# output = word_to_idx: dict, idx_to_word: dict, vocab: array

	word_to_idx = {}
	idx_to_word = {}
	vocab = []
	
	word_to_idx[" "] = 0
	idx_to_word[0] = " "
	vocab.append(" ")
	
	# generate vocabulary
	num_word = 1
	for i in range(len(data)):
		words = data[i].split(" ")
		for j in range(len(words)):
			if not words[j] in word_to_idx:
				idx_to_word[num_word] = words[j]
				word_to_idx[words[j]] = num_word
				vocab.append(words[j])
				num_word += 1
			
	return word_to_idx, idx_to_word, vocab

def generate_vocab_with_custom_embedding(vocab, word_embedding):
	# input = vocab: list, word_embdding
	# output = word_to_idx: dict, idx_to_word: dict, vocab: array

	word_to_idx = {}
	idx_to_word = {}
	vocab = [" "] + vocab

	new_embedding = np.zeros((word_embedding.shape[0] + 1, word_embedding.shape[1]))
	for i in range(word_embedding.shape[0]):
		new_embedding[i+1] = word_embedding[i]

	word_to_idx[" "] = 0
	idx_to_word[0] = " "
	
	# generate vocabulary
	for i in range(len(vocab)):
		word_to_idx[vocab[i]] = len(word_to_idx)
		idx_to_word[len(idx_to_word)] = vocab[i]
			
	return word_to_idx, idx_to_word, vocab, new_embedding

def generate_word_index(data, labels, word_to_idx, idx_to_word, vocab, max_sequence_length):
	# input = data: matrix of words, labels: array of words, word_to_idx: dict, idx_to_word: dict, vocab: array of words
	# output = train_data: np array, train_labels: np array

	train_data = np.zeros((len(data), max_sequence_length), dtype=np.int)
	train_labels = np.zeros((len(labels)))
	num_classes = 2
	
	# generate vocabulary
	for i in range(len(data)):
		words = data[i].split(" ")
		for j in range(min(len(words), max_sequence_length)):
			cur_word = words[j]
			if not cur_word in word_to_idx:
				cur_word = " "

			idx = word_to_idx[cur_word]
			train_data[i][j] = idx
		
		label = float(labels[i].replace("\n",""))
		train_labels[i] = label
		
	train_labels = (np.arange(num_classes) == train_labels[:,None]).astype(np.float32)

	return train_data, train_labels

def preprocess_lowercase_negation(seq):
	seq = seq.lower()
	modal_tobe = ["do", "does", "did", "will", "would", "could", "should", "shall", "may", "might", "must", "is", "are", "was", "were", "has", "have", "had"]
	
	arr = seq.split(" ")
	for j in range(len(arr)):
		word = arr[j]
		for i in range(len(modal_tobe)):
			word = word.replace(modal_tobe[i] + "nt", modal_tobe[i] + " not")
			word = word.replace(modal_tobe[i] + "n't", modal_tobe[i] + " not")
			arr[j] = word
			
		word = word.replace("won't", "will not")
		word = word.replace("can't", "cannot")
			
	seq = ""
	for j in range(len(arr)):
		seq += arr[j]
		if j < len(arr) - 1:
			seq += " "
	return seq

def check_data(data):
	num_words = 0
	vocab = {}

	for i in range(len(data)):
		arr = data[i].split(" ")
		for j in range(len(arr)):
			if not arr[j] in vocab:
				vocab[arr[j]] = True

		num_words += len(arr)

	print("Number of words:", num_words)
	print("Number of unique words:", len(vocab))


############################################## 
# GENERATE EMBEDDINGS
##############################################

# print(">>> load embedding")

# # LOAD EMBEDDING
# emotion_rnn300_vocab = pickle.load(open("./pretrained/emotion_embedding_size_300/rnn/vocab.pkl","rb"), encoding='latin1')
# emotion_rnn300_embedding = pickle.load(open("./pretrained/emotion_embedding_size_300/rnn/emotion_embedding.pkl","rb"), encoding='latin1')

# emotion_cnn300_vocab = pickle.load(open("./pretrained/emotion_embedding_size_300/cnn/vocab.pkl","rb"), encoding='latin1')
# emotion_cnn300_embedding = pickle.load(open("./pretrained/emotion_embedding_size_300/cnn/emotion_embedding.pkl","rb"), encoding='latin1')

# emotion_50_vocab = pickle.load(open("./pretrained/emotion_embedding_size_50/vocab.pkl","rb"), encoding='latin1')
# emotion_50_embedding = pickle.load(open("./pretrained/emotion_embedding_size_50/emotion_embedding.pkl","rb"), encoding='latin1')

# word2vec = KeyedVectors.load_word2vec_format('./pretrained/word2vec_embedding_size_300/GoogleNews-vectors-negative300.bin', binary=True)

# basic_word2vec300_vocab = pickle.load(open("./pretrained/basic_word2vec/vocab.pkl","rb"), encoding='latin1')
# basic_word2vec300_embedding = pickle.load(open("./pretrained/basic_word2vec/word2vec300_gensim.pkl","rb"), encoding='latin1')

# custom_vocab = None
# custom_embedding = None

# # FUNCTIONS TO CALL THE EMBEDDING
# emotion_cnn300_reverse_dict = {}
# for i in emotion_cnn300_vocab.keys():
# 	emotion_cnn300_reverse_dict[emotion_cnn300_vocab[i]] = i
	
# emotion_rnn300_reverse_dict = {}
# for i in emotion_rnn300_vocab.keys():
# 	emotion_rnn300_reverse_dict[emotion_rnn300_vocab[i]] = i
	
# emotion_50_reverse_dict = {}
# for i in emotion_50_vocab.keys():
# 	emotion_50_reverse_dict[emotion_50_vocab[i]] = i

# basic_word2vec300_reverse_dict = {}
# for i in basic_word2vec300_vocab.keys():
# 	basic_word2vec300_reverse_dict[basic_word2vec300_vocab[i]] = i

# FOR CUSTOM VOCAB AND EMBEDDING
def set_custom_vocab(vocab_path, embedding_path):
	custom_vocab = pickle.load(open(vocab_path))
	custom_embedding = pickle.load(open(embedding_path))
	
	custom_reverse_dict = {}
	for i in custom_vocab.keys():
		custom_reverse_dict[custom_vocab[i]] = i
		
def set_custom_vocab_np(vocab_path, embedding_path):
	custom_vocab = np.load(open(vocab_path), encoding="latin1")
	custom_embedding = np.load(open(embedding_path), encoding="latin1")
	
	custom_reverse_dict = {}
	for i in custom_vocab.keys():
		custom_reverse_dict[custom_vocab[i]] = i
		
def get_custom_embedding(word):
	if word in custom_reverse_dict.keys():
		return custom_embedding[custom_reverse_dict[word], :]
	else:
		return None
	
def get_emotion_cnn300_embedding(word):
	if word in emotion_cnn300_reverse_dict.keys():
		return emotion_cnn300_embedding[emotion_cnn300_reverse_dict[word], :]
	else:
		return None
	
def get_emotion_rnn300_embedding(word):
	if word in emotion_rnn300_reverse_dict.keys():
		return emotion_rnn300_embedding[emotion_rnn300_reverse_dict[word], :]
	else:
		return None
	
def get_emotion_50_embedding(word):
	if word in emotion_50_reverse_dict.keys():
		return emotion_50_embedding[emotion_50_reverse_dict[word], :]
	else:
		return None
	
def get_basic_word2vec300_embedding(word):
	if word in basic_word2vec300_reverse_dict.keys():
		return basic_word2vec300_embedding[basic_word2vec300_reverse_dict[word], :]
	else:
		return None

def get_distance(word1, word2):
	embed1 = get_embedding(word1)
	embed2 = get_embedding(word2)
	if embed1 is not None and embed2 is not None:
		return spatial.distance.cosine(embed1, embed2)
	return None

def get_word2vec_embedding(word):
	if word in word2vec:
		return word2vec[word]
	else:
		return None

def generate_embedding(data, labels, embedding="rnn300", embedding_size=300):
	num_classes = 2

	if embedding == "google_word2vec":
		data, labels = generate_word2vec_embedding_data(data, labels, embedding_size)
	elif embedding == "emo_rnn300":
		data, labels = generate_emotion_rnn300_embedding_data(data, labels, embedding_size)
	elif embedding == "emo_cnn300":
		data, labels = generate_emotion_cnn300_embedding_data(data, labels, embedding_size)
	elif embedding == "emo_50":
		data, labels = generate_emotion_50_embedding_data(data, labels, embedding_size)
	elif embedding == "basic_word2vec300":
		data, labels = generate_basic_word2vec300_embedding_data(data, labels, embedding_size)
	elif embedding == "custom":
		data, labels = generate_custom_embedding_data(data, labels, embedding_size)
		
	labels = (np.arange(num_classes) == labels[:,None]).astype(np.float32)
	
	return data, labels

def read_data(path):
	texts = []
	labels = []
	
	with open(path, "r") as file:
		line_count = 0
		for line in file:
#             print(line)
			sample = []
			if line_count > 0:
				temp = np.zeros((50))
				row = line.split(",")

				text = row[1].replace("\"","").replace("\n", "")
				label = row[2]

				texts.append(text)
				labels.append(label.replace("\n",""))

				#print("text", text, "label", label)
			line_count+=1
			
		return texts, labels
	
# generate CUSTOM embedding data
def generate_custom_embedding_data(texts, labels, embedding_size):
	
	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	print(len(texts))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(len(words)):
			word = words[j].strip()
			embed = get_custom_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
		
		label = labels[i]
		
		train_data[i] = word_embedding
		train_labels[i] = label
		
	
	return train_data, train_labels
	
def generate_emotion_cnn300_embedding_data(texts, labels, embedding_size):
	
	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	print(len(texts))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(len(words)):
			word = words[j].strip()
			embed = get_emotion_cnn300_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
		
		label = labels[i]
		
		train_data[i] = word_embedding
		train_labels[i] = label
		
	
	return train_data, train_labels

def generate_emotion_rnn300_embedding_data(texts, labels, embedding_size):
	
	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(len(words)):
			word = words[j].strip()
			embed = get_emotion_rnn300_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
		
		label = float(labels[i].replace("\n",""))
		
		train_data[i] = word_embedding
		train_labels[i] = label
		
	return train_data, train_labels

def generate_emotion_50_embedding_data(texts, labels, embedding_size):
	
	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(len(words)):
			word = words[j].strip()
			embed = get_emotion_50_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
		
		label = float(labels[i].replace("\n",""))
		
		train_data[i] = word_embedding
		train_labels[i] = label
	
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
		
	return train_data, train_labels

def generate_basic_word2vec300_embedding_data(texts, labels, embedding_size):
	
	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(min(len(words), max_sequence_length)):
			word = words[j].strip()
			embed = get_basic_word2vec300_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
		
		label = float(labels[i].replace("\n",""))
		
		train_data[i] = word_embedding
		train_labels[i] = label
	
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
		
	return train_data, train_labels

def generate_word2vec_embedding_data(texts, labels, embedding_size):

	train_data = np.zeros((len(texts), max_sequence_length, embedding_size))
	train_labels = np.zeros((len(labels)))
	
	print(len(texts))
	
	for i in range(len(texts)):
		word_embedding = np.zeros((max_sequence_length, embedding_size))
		
		sentence = texts[i]
		words = sentence.split(" ")
		
		for j in range(len(words)):
			word = words[j].strip()
			embed = get_word2vec_embedding(word)
			
			if not embed is None:
				word_embedding[j] = embed
			else:
				word_embedding[j] = np.zeros((embedding_size))
			
		label = float(labels[i].replace("\n",""))
		
		train_data[i] = word_embedding
		train_labels[i] = label
	
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
		
	return train_data, train_labels