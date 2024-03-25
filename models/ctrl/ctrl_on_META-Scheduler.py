import tvm
from tensorflow import keras
from tvm import autotvm, relay
from keras.utils import to_categorical
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms
import multiprocessing
import numpy as np
import os
from tvm.contrib import graph_executor
from keras.utils import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

model = load_model('CTRL.h5')  # loading model

embedding_model = Sequential(model.layers[0])  # creating a new model with embedding layer only

input_shape = (1, 1130, 128)  # creating input data
shape_dict = {"embedding_input": input_shape}
mod, params = relay.frontend.from_keras(embedding_model, shape_dict, layout="NHWC")

model_name = "embedding_model"
input_name = "embedding_input"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = multiprocessing.cpu_count()

os.environ["TVM_NUM_THREADS"] = str(int(num_threads))

strategy_name = "evolutionary"
work_dir = "{}_meta-scheduler_{}".format(model_name, strategy_name)
target = tvm.target.Target("llvm -mcpu=core-avx2 -num-cores {}".format(int(num_threads)))


def evaluate_performance(lib, data_shape, dtype="float32"):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))


# extract workloads from relay program
def extract_tasks(mod, target, params, strategy):

    extracted_tasks = ms.relay_integration.extract_tasks(
        mod, target, params
    )
    assert (len(extracted_tasks) > 0)

    tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
        extracted_tasks, work_dir, strategy=strategy
    )

    for idx, task in enumerate(tasks):
        print("Task: %d, desc: %s" % (idx, task.task_name))

    return tasks, task_weights


tasks, task_weights = extract_tasks(mod, target, params, strategy_name)


def run_tuning(tasks, task_weights, work_dir, n_trials):
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    print("Begin tuning...")
    evaluator_config = ms.runner.config.EvaluatorConfig(number=1, repeat=10, enable_cpu_cache_flush=True)
    database = ms.tune.tune_tasks(
        tasks=tasks,
        task_weights=task_weights,
        work_dir=work_dir,
        max_trials_global=n_trials,
        num_trials_per_iter=64,
        max_trials_per_task=256,
        builder=ms.builder.LocalBuilder(),
        runner=ms.runner.LocalRunner(evaluator_config=evaluator_config),
    )


run_tuning(tasks, task_weights, work_dir, 513)


def evaluate(module, data_shape, work_dir, target="llvm"):
    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    print("Compile...")
    database = ms.database.JSONDatabase(f"{work_dir}/database_workload.json",
                                        f"{work_dir}/database_tuning_record.json",
                                        allow_missing=False)
    with tvm.transform.PassContext(opt_level=3):
        lib = ms.relay_integration.compile_relay(database, module, target, params)
        evaluate_performance(lib, data_shape)


for i in range(7):
    print(f"{'|'*25} Test №{i} {'|'*25}")
    evaluate(mod, input_shape, work_dir, target)

print("~" * 40)

lstm_model = Sequential()
lstm_model.add(LSTM(units=256, input_shape=(10, 128)))
lstm_model.add(Dense(units=2, activation='softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

input_shape =[1, 10, 128]
shape_dict = {"lstm_input": input_shape}
mod, params = relay.frontend.from_keras(lstm_model, shape_dict, layout="NHWC")

model_name = "lstm_model"
input_name = "lstm_input"

for i in range(7):
    print(f"{'|'*25} Test №{i} {'|'*25}")
    evaluate(mod, input_shape, work_dir, target)
