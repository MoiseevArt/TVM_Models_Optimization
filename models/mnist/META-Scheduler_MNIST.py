import tvm
from tensorflow import keras
from tvm import autotvm, relay
from keras.utils import to_categorical
from tvm.contrib import graph_executor
from tvm import meta_schedule as ms
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

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = multiprocessing.cpu_count()
print("Num threads: ", int(num_threads))
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
    print("Mod:")
    print(mod)
    print("Extract tasks...")
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


for i in range(10):
    print(f"{'|'*25} Test №{i} {'|'*25}")
    evaluate(mod, input_shape, work_dir, target)
