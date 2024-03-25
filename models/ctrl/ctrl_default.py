from keras.utils import pad_sequences
from keras.models import load_model
from keras.models import Sequential
import time

model = load_model('CTRL.h5')  # loading model


def get_execution_time(iterations, model, input_data):
    for i in range(iterations):
        start_time = time.time()
        output = model.predict(input_data)
        end_time = time.time()

        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")


input_data = pad_sequences([[0] * 1130], maxlen=1130)  # creating input data

embedding_model = Sequential(model.layers[0])  # creating a new model with embedding layer only

print("Execution time for embedding:")
get_execution_time(20, embedding_model, input_data)  # embedding benchmarking results

print("~" * 40)

lstm_model = Sequential()  # creating a new model with lstm+dance
for i, layer in enumerate(model.layers):
    if i != 0:
        lstm_model.add(layer)


lstm_model.build((None, 1130, 128))

embedding_output = embedding_model.predict(input_data)  # get new input data

print("Execution time for lstm+dance:")
get_execution_time(20, lstm_model, (embedding_output))  # lstm+dance benchmarking results
