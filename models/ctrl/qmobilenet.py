import tvm
import time
import torch
import numpy as np
from tvm import relay
from PIL import Image
from tvm.contrib.download import download_testdata
from torchvision.models.quantization import mobilenet as qmobilenet


def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def get_synset():
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())


def run_tvm_model(mod, params, input_name, inp, target="llvm"):
    with tvm.transform.PassContext(opt_level=1):
        lib = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))

    runtime.set_input(input_name, inp)
    runtime.run()
    return runtime.get_output(0).numpy(), runtime


synset = get_synset()

inp = get_imagenet_input()


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)


qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()


pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)
script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()


input_name = "input"
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

target = "llvm"
tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target=target)

pt_top3_labels = np.argsort(pt_result[0])[::-1][:3]
tvm_top3_labels = np.argsort(tvm_result[0])[::-1][:3]

print("PyTorch top3 labels:", [synset[label] for label in pt_top3_labels])
print("TVM top3 labels:", [synset[label] for label in tvm_top3_labels])

print("%d in 1000 raw floating outputs identical." % np.sum(tvm_result[0] == pt_result[0]))

##########################################################################
# Measure performance
# -------------------------
# Here we give an example of how to measure performance of TVM compiled models.
n_repeat = 100  # should be bigger to make the measurement more accurate
dev = tvm.cpu(0)
print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))


model = qmobilenet.mobilenet_v2(pretrained=True, quantize=True)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


img_path = '1.jpg'
input_image = Image.open(img_path).convert('RGB')
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)


start_time = time.time()
with torch.no_grad():
    output = model(input_batch)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Время выполнения предсказания: {elapsed_time:.4f} секунд")
