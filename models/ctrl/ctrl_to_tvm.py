import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tvm
import time
from tvm import relay
from tvm.contrib import graph_executor
from keras.utils import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


model = load_model('CTRL.h5')  # loading model


def get_execution_time(iterations, opt_lvl, mod, dev, target, params):  # converte to tvm

    with tvm.transform.PassContext(opt_level=opt_lvl):  # create GraphExecutor
    	tvm_model = relay.build_module.create_executor("graph", mod, dev, target, params)

    for i in range(iterations):
        start_time = time.time()
        output = tvm_model.evaluate()
        end_time = time.time()

        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")


embedding_model = Sequential(model.layers[0])  # creating a new model with embedding layer only

input_shape = (1, 1130, 128)  # creating input data
shape_dict = {"embedding_input": input_shape}

mod, params = relay.frontend.from_keras(embedding_model, shape_dict, layout="NHWC")  # get relay from keras model

target = tvm.target.Target("llvm  -mcpu=core-avx2")  # set target & device
dev = tvm.cpu(0)

print("Execution time for embedding on tvm [with optimization]:")  # embedding benchmarking results on tvm (with optimization)
get_execution_time(20, 3, mod, dev, target, params)

print("~" * 40)

target = tvm.target.Target("llvm")  # without optimization

print("Execution time for embedding on tvm [without optimization]:")  # embedding benchmarking results on tvm (without optimization)
get_execution_time(20, 0, mod, dev, target, params)

print("~" * 40)

lstm_model = Sequential()
lstm_model.add(LSTM(units=256, input_shape=(10, 128)))
lstm_model.add(Dense(units=2, activation='softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

input_shape =[1, 10, 128]
shape_dict = {"lstm_input": input_shape}


mod, params = relay.frontend.from_keras(lstm_model, shape_dict, layout="NHWC")  # get relay from keras model

target = tvm.target.Target("llvm -mcpu=core-avx2")

print("Execution time for lstm+dance on tvm [with optimization]:")  # lstm+dance benchmarking results on tvm (with optimization)
get_execution_time(20, 3, mod, dev, target, params)

print("~" * 40)

target = tvm.target.Target("llvm")  # without optimization

print("Execution time for lstm+dance on tvm [without optimization]:")  # lstm+dance benchmarking results on tvm (without optimization)
get_execution_time(20, 0, mod, dev, target, params)

print("~" * 40)
