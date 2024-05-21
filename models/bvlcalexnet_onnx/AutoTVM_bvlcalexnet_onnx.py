import tvm
from tvm import autotvm, relay
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tensorflow import keras
import onnx
import multiprocessing
import numpy as np
import os

# Загрузка модели из ONNX файла
onnx_model = onnx.load("bvlcalexnet-7.onnx")

# Преобразование ONNX модели в Relay формат
mod, params = relay.frontend.from_onnx(onnx_model)

# Определение входной формы данных
input_shape = [1, 224, 224, 3]  # [batch, height, width, channels]
shape_dict = {"input": input_shape}  # "input" - это имя входного тензора в вашей ONNX модели

# Извлечение имени входа из ONNX графа
input_name = onnx_model.graph.input[0].name if onnx_model.graph.input else None
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


log_file = "%s.autotvm.log" % "my_bvlcalexnet_model"


# extract workloads from relay program
def extract_tasks(mod, target, params):
    print("Mod:")
    print(mod)
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod, target=target, params=params
    )
    assert (len(tasks) > 0)
    for idx, task in enumerate(tasks):
        print("Task: %d,  workload: %s" % (idx, task.workload))
    return tasks


tasks = extract_tasks(mod, "llvm -mcpu=core-avx2", params)


def run_tuning(
        tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
):
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank-binary")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = len(task.config_space)
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )


measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
)

# run tuning tasks
run_tuning(tasks, measure_option, tuner="xgb", log_filename=log_file)


def evaluate(module, data_shape, log_file, target="llvm"):
    # compile kernels in default mode
    print("Evaluation of the network compiled in 'default' mode without auto tune:")
    with tvm.transform.PassContext(opt_level=1):
        print("Compile...")
        lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)

    # compile kernels in kernel tuned only mode
    print("\nEvaluation of the network been tuned on kernel level:")
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=1):
            lib = relay.build(module, target=target, params=params)
        evaluate_performance(lib, data_shape)


for i in range(5):
    print(f"{'|' * 25} Test №{i} {'|' * 25}")
    evaluate(mod, input_shape, log_file, target)
