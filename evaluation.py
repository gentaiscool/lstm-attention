import numpy as np
from copy import deepcopy

def accuracy(predictions, labels):
	return (np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / np.float32(predictions.shape[0]))

def fscore(predictions, labels):
	a = deepcopy(predictions[:,0])
	a[a<0.5] = 0
	a[a>=0.5] = 1
	b = deepcopy(predictions[:,1])
	b[b<0.5] = 0
	b[b>=0.5] = 1
	positives = np.sum(b)
	negatives = np.sum(a)
	true_positives = np.sum(b*labels[:,1])
	false_positives = positives - true_positives
	true_negatives = np.sum(a*labels[:,0])  
	false_negatives = negatives - true_negatives
	precision = true_positives/positives
	recall = true_positives/(true_positives+false_negatives)
	f_score = (2*precision*recall)/(precision+recall)
	return f_score

def recall(predictions, labels):
	a = deepcopy(predictions[:,0])
	a[a<0.5] = 0
	a[a>=0.5] = 1
	b = deepcopy(predictions[:,1])
	b[b<0.5] = 0
	b[b>=0.5] = 1
	positives = np.sum(b)
	negatives = np.sum(a)
	true_positives = np.sum(b*labels[:,1])
	false_positives = positives - true_positives
	true_negatives = np.sum(a*labels[:,0])  
	false_negatives = negatives - true_negatives
	recall = true_positives/(true_positives+false_negatives)
	return recall

def precision(predictions, labels):
	a = deepcopy(predictions[:,0])
	a[a<0.5] = 0
	a[a>=0.5] = 1
	b = deepcopy(predictions[:,1])
	b[b<0.5] = 0
	b[b>=0.5] = 1
	positives = np.sum(b)
	negatives = np.sum(a)
	true_positives = np.sum(b*labels[:,1])
	false_positives = positives - true_positives
	true_negatives = np.sum(a*labels[:,0])  
	false_negatives = negatives - true_negatives
	precision = true_positives/positives
	return precision