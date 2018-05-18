import numpy as np
import data_helpers
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sklearn.feature_extraction.text


print("Loading data...")
x_text, y = data_helpers.load_data_and_labels("rt-polarity.pos", "rt-polarity.neg")

# calculate the BOW representation
count_vector = sklearn.feature_extraction.text.CountVectorizer()
word_counts = count_vector.fit_transform(x_text)

# TFIDF
tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True).fit(word_counts)
x = tf_transformer.transform(word_counts)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
test_sample_index = -1 * int(.2 * float(len(y)))
x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]

del x, y, x_shuffled, y_shuffled

#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

clf = SVC()
svm_model = clf.fit(x_train, y_train)

y_test_pred = svm_model.predict(x_test)
print("Accuracy on test data:", accuracy_score(y_test, y_test_pred))
