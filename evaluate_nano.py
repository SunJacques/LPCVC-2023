import torch
import argparse
from sample_solution.evaluation.accuracy import AccuracyTracker
from tqdm import tqdm
from torchvision import transforms as T
import numpy as np
from PIL import Image
from torch2trt import torch2trt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
print("[Finish import packages]")

IMG_SIZE = 256
mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]
colors = ['green', 'red', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'pink', 'lime', 'brown', 'gray', 'olive', 'teal', 'navy']
cmap = ListedColormap(colors[:14])

accuracyTrackerVal = AccuracyTracker(n_classes=14)

def load_image(idx, datapath):
    transform = T.Compose([T.ToTensor(),T.Resize(size=IMG_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)])
    print("[Loading the image...] at " + datapath)
    img = Image.open(datapath + '/val/IMG/val_' + str(46).zfill(4) + '.png').convert('RGB')
    print("[Finish load image]")
    x = np.array(img)
    x = transform(x)
    t = T.Compose([T.Normalize(mean, std)])
    x = t(x)
    return torch.tensor(np.expand_dims(x, axis=0)).cuda(), img

def save_image(output, img, save_images_path):
    output = output.cpu().data.max(1)[1].numpy()
    output.astype(np.uint8)
    output = np.squeeze(output, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].imshow(output, vmin=0, vmax=13, cmap=cmap, alpha=0.4)
    axes[0].set_xlabel('Prediction')

    axes[1].imshow(img)

    plt.savefig('result.png')

def main():
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--modelpath', default="PSPNET_MN_NEWSHED_1063_dice_0.5777767708560559.pth")
    parser.add_argument('--datapath', default='dataset')
    parser.add_argument('--dev', default="cuda:0")
    parser.add_argument('--save_images_path', default="")
    args = parser.parse_args()
    args.device = torch.device(args.dev)
    torch.cuda.empty_cache()
    #------------------------------------------------------------   
    print("[Loading model...] at " + args.modelpath)
    model = torch.load(args.modelpath)
    model.to(args.device)
    model.eval()
    print("[Finish load model]")
    #------------------------------------------------------------
    print("[Quantization of the model...]")
    model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model_int8.to(args.device)
    model_int8.eval()
    print("[Finish Quantization]")
    #------------------------------------------------------------
    x, img = load_image(random.randint(1, 100), args.datapath)
    #------------------------------------------------------------
    # convert to TensorRT feeding sample data as input
    print("[Convert the model with TensorRT...]")
    model= torch2trt(model, [x])
    print("[Finish TensorRT]")
    #------------------------------------------------------------
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print("[Start measuring performance...] with 30 inferences")
    running_inference_time = 0
    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(4):
            _ = model(x)
        # MEASURE PERFORMANCE
        for i in range(30):
            starter.record()
            output = model(x)
            ender.record()
            torch.cuda.synchronize()
            running_inference_time += starter.elapsed_time(ender)
    print("[Finish measuring performance]")
    print("___________________________________________________")
    print("______________________RESULTS______________________")
    print("___________________________________________________")

    print("Mean Inference Time = " + str(round(running_inference_time/30, 3)) + " ms")

    save_image(output, img, args.save_images_path)
    print("Save results at " + args.save_images_path)


if __name__ == '__main__':
    main()
