from sklearn import svm

############################################## 
# SVM MODELS
##############################################

def svm():
	return svm.NuSVC()

def train(model, train_data, train_labels, test_data, test_labels):
	print(len(train_arr))
	print(len(train_arr_label))
	print(len(test_arr))
	print(len(test_arr_label))

	scoring = ['accuracy', 'precision', 'recall', 'f1_score']
	model.fit(train_arr, train_arr_label)
	y_predict = model.predict(test_arr)
	confusion = metrics.confusion_matrix(test_arr_label, y_predict)

	print("#######################################")
	print("confusion matrix")
	print("#######################################")
	print(confusion)

	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]

	classification_error = (FP + FN) / float(TP + TN + FP + FN)

	print("accuracy:", metrics.accuracy_score(test_arr_label, y_predict))
	print("recall:", metrics.recall_score(test_arr_label, y_predict))
	print("precision:", metrics.precision_score(test_arr_label, y_predict))
	print("f1:", metrics.f1_score(test_arr_label, y_predict))

	return model