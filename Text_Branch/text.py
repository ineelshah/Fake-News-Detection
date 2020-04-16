# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.14

#importing libraries
from keras.utils.vis_utils import plot_model
from keras.models import Model
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.layers import Flatten, BatchNormalization, Concatenate
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Conv2D
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers.merge import concatenate



#Loading the data in a dataframe
df = pd.read_csv('drive/My Drive/The_Research/all_data_refined.csv')
df = df.drop(['emotion'], axis = 1)
docs = df['text']

#test
for i in range(len(docs)):
  if docs[i] == "": print(i)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)

# pad documents to a max length
wordlen = max(df['word_count'])
max_length = wordlen # Change this if needed
print("Max_Length: ", max_length)
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


# load the whole embedding into memory
embeddings_index = dict()
f = open('drive/My Drive/glove_data/glove.6B/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()


# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector


#load the labels
type = df['type']
labels = []
for types in type:
  if types == 'real':
    labels.append(1)
  elif types == 'fake':
    labels.append(0)


arr = df[df.columns[8: 23]]

listt = list()
for i in range(len(df)):
  listt.append(i)


X_train_1 = list()
X_train_2 = list()
X_test_1 = list()
X_test_2 = list()


X_train, X_test, y_train, y_test = train_test_split(listt, labels, test_size=0.33)

for i in X_train:
  X_train_1.append(padded_docs[i])
  X_train_2.append(arr.iloc[i])

for i in X_test:
  X_test_1.append(df['text'][i])
  X_test_2.append(arr.iloc[i])


# define Implicit
inputs1 = Input(shape=(max_length,))
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs1)
a = (Dropout(0.5))(e)
b = (Conv1D(filters=10, kernel_size=(4)))(a)
c=MaxPooling1D(pool_size=2)(b)
d=Flatten()(c)
f=Dense(128)(d)
g=(BatchNormalization())(f)
z = Activation('relu')
h=(Dropout(0.8))(g)

# define Explicit
inputs2 = Input(shape=(15,))
q=(Dense(128))(inputs2)
r=(BatchNormalization())(q)
u=Activation('relu')(r)



merged = concatenate([h, u])
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# summarize the model
print("____________________")
print(model.summary())
print("____________________")

plot_model(model, show_shapes=True, to_file='multichannelbeta.png')

# fit the model
print("Fitting")
model.fit([X_train_1, X_train_2], array(y_train), epochs=5, verbose=1, batch_size=16)
print("Fitted")

# evaluate the model
print("____________________")
loss, accuracy = model.evaluate([X_test_1, X_test_2] , array(y_test), verbose=1, batch_size=16)
print('Accuracy: %f' % (accuracy*100))

print("____________________")
output = model.predict(X_test)
print(output)

image_input = Input((512, 512, 1))
vector_input = Input((12,))

image_model = Conv2D(32,(8,8), strides=(4,4))(image_input)
image_model = Activation('relu')(image_model)
image_model = Conv2D(64,(4,4), strides=(2,2))(image_model)
image_model = Activation('relu')(image_model)
image_model = Conv2D(64,(3,3), strides=(1,1))(image_model)
image_model = Activation('relu')(image_model)
image_model = Flatten()(image_model)
image_model = Dense(512)(image_model)
image_model = Activation('relu')(image_model)

value_model = Dense(16)(vector_input)
value_model = Activation('relu')(value_model)
value_model = Dense(16)(value_model)
value_model = Activation('relu')(value_model)
value_model = Dense(16)(value_model)
value_model = Activation('relu')(value_model)

merged = concatenate([image_model, value_model])

output = Dense(1, activation='sigmoid')(merged)

model = Model(inputs=[image_input, vector_input], outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam')

df = pd.read_csv('drive/My Drive/The_Research/all_data_refined.csv')
df = df.drop(['emotion'], axis = 1)
a = df[df.columns[8: 23]]
a = a[0:1]
a

df = pd.read_csv('drive/My Drive/The_Research/all_data_refined.csv')
arr = df.drop(['emotion'], axis = 1)
arr = df[df.columns[8: 23]]



listt = list()
for i in range(len(df)):
  listt.append(i)
X_train, X_test, y_train, y_test = train_test_split(listt, labels, test_size=0.33)
print(X_train)

X_train_1 = list()
X_train_2 = list()
X_test_1 = list()

for i in X_test:
    X_train_1.append(df['text'][i])
    X_train_2.append(arr.iloc[i])
print(len(X_train_1))

