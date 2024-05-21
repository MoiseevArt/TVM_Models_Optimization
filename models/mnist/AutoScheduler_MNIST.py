import tvm
from tensorflow import keras
from tvm import auto_scheduler, relay
from tvm.contrib.debugger import debug_executor
from tvm.contrib import graph_executor
from keras.utils import to_categorical
from keras.datasets import mnist
import multiprocessing
import numpy as np
import os

(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 28, 28, 1)
y_test = to_categorical(y_test)

model = keras.models.load_model("mnist_keras_model")

input_shape = [1, 28, 28, 1]  # [batch, height, width, channels]
shape_dict = {"input_input": input_shape}
mod, params = relay.frontend.from_keras(model, shape_dict, layout="NHWC")
print(mod)

model_name = "my_mnist_model"
input_name = "input_input"
target = tvm.target.Target("llvm -mcpu=core-avx2")

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = multiprocessing.cpu_count()
print("Num threads: ", int(num_threads))
os.environ["TVM_NUM_THREADS"] = str(int(num_threads))


def collect_per_layer_stat(lib, device, json_graph=None):
    if json_graph is None:
        json_graph = lib.get_graph_json()
    debug_module = debug_executor.GraphModuleDebug(lib["debug_create"]("default", device), [device], json_graph, None)
    debug_module.run(number=10, repeat=3)


def evaluate_performance(lib, data_shape, dtype="float32"):
    # upload parameters to device
    dev = tvm.cpu()
    data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    module = graph_executor.GraphModule(lib["default"](dev))
    module.set_input(input_name, data_tvm)

    # evaluate
    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=100, repeat=3))

    collect_per_layer_stat(lib, dev)


log_file = "%s.auto-scheduler.log" % model_name


# extract workloads from relay program
def extract_tasks(mod, target, params):
    print("Mod:")
    print(mod)
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod, params, target)
    assert (len(tasks) > 0)

    for idx, task in enumerate(tasks):
        print("Task: %d, desc: %s" % (idx, task.desc))
    return tasks, task_weights


tasks, task_weights = extract_tasks(mod, target, params)


def run_tuning(tasks, task_weights, log_file, n_trials):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=n_trials,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)


run_tuning(tasks, task_weights, log_file, 512)


def evaluate(module, data_shape, log_file, target="llvm"):
    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=3):
        print("Compile...")
        lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with auto_scheduler.ApplyHistoryBest(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)


for i in range(10):
    print(f"{'|'*25} Test №{i} {'|'*25}")
    evaluate(mod, input_shape, log_file, target)
