
def text_implict_preprocessing(texts):
    # initialize tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(texts)

    padded_texts = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

    return tokenizer, padded_texts, vocab_size

def load_embedding_matrix(tokenizer, VOCAB_SIZE):
    embeddings_index = dict()
    f = open('drive/My Drive/glove_data/glove.6B/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((VOCAB_SIZE, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def image_implicit_preprocessing(uuid, type):
    image_dataset = []
    for i in range(len(uuid)):
        sub_dir = ''
        if(type[i] == 1):
            sub_dir = 'real'
        else :
            sub_dir = 'fake'
        path = os.path.join(IMAGE_DIR, sub_dir)
        temp_img = image.load_img(os.path.join(path, str(uuid[i])+'.jpg'),target_size=(224,224))
        temp_img = image.img_to_array(temp_img)
        image_dataset.append(temp_img)

    image_dataset = np.array(image_dataset)
    image_dataset = preprocess_input(image_dataset)

    return image_dataset