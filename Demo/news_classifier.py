import pickle
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000

index_to_category = {
    0:'business',
    1:'entertainment',
    2:'politics',
    3:'sport',
    4:'tech'
}

def load_model():
    # load json and create model
    json_file = open('news-classifier-model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("news-classifier-model.h5")
    print("Loaded model from disk")

    # compile new model
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return loaded_model

def preprocessInput(headline, content=''):
    tokenizer = []

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        print("Loaded tokenizer from disk")

    text = [headline + content]
    sequence = tokenizer.texts_to_sequences(text)
    text = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    return text

def predict_news_category(headline, content=''):
    #load pretrained model
    model = load_model()

    # Preprocessing on text input
    preprocessed_text = preprocessInput(headline, content)

    # Predicting category
    category_probability = model.predict(preprocessed_text)
    categoryIndex = category_probability.argmax(axis=-1)

    #return news category
    if(max(category_probability[0]) > 0.35):
        return index_to_category[categoryIndex[0]]
    else:
        return 'others'

    