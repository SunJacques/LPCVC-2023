import torch
import argparse
from vanilla_lpcvc import LPCVCDataset

from sample_solution.evaluation.accuracy import AccuracyTracker
from tqdm import tqdm
from torchvision import transforms as T
import numpy as np
from PIL import Image
from fast_scnn import FastSCNN
import segmentation_models_pytorch as smp
from torch2trt import torch2trt


print("Finish import packages")

IMG_SIZE = 256
mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]

ENCODER = 'mobilenet_v2'
ENCODER_WEIGHTS = 'imagenet'
N_CLASSES = 14
ACTIVATION = 'sigmoid'

accuracyTrackerVal = AccuracyTracker(n_classes=14)

def eval(model, args, val_loader, memory_allocated_prev):

    running_time = 0
    running_memory = 0
    iteration=0

    saved_images = np.zeros((3, IMG_SIZE, IMG_SIZE, 3))
    print("start timer")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print("end timer")
    with torch.no_grad():
        print("load loop")
        loop = tqdm(val_loader)
        print("finish loop")
        for batch_idx, (inputs, labels) in enumerate(loop):
            print("iter")
            iteration+=1
            
            print("swicth input/output to the device")
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            
            print("make inference")
            starter.record()
            outputs=model(inputs)
            ender.record()
            print("finish inference")

            memory_allocated = torch.cuda.memory_allocated(args.device)
            running_memory += memory_allocated - memory_allocated_prev

            torch.cuda.synchronize()

            running_time += starter.elapsed_time(ender)

            print("detach from gpu")
            outputs = outputs.cpu().data.max(1)[1].numpy()
            labels = labels.cpu().data.max(1)[1].numpy()

            outputs.astype(np.uint8)
            labels.astype(np.uint8)

            print("update accuracy")
            accuracyTrackerVal.update(labels, outputs)


    val_time = running_time/iteration
    print("Memory used for one inference : {:.2f} MB".format(running_memory / iteration / (1024 * 1024)))

    return(val_time, saved_images)



def save_images(saved_images, path):
    input_image, target_image, pred_image = saved_images[0], saved_images[1], saved_images[2]

    input_image = np.array((input_image*0.1707+0.45)*255, dtype=np.uint8)
    target_image = np.array(target_image*255, dtype=np.uint8)
    pred_image = np.array(pred_image*255, dtype=np.uint8)

    Image.fromarray(input_image).save(path+"input.png")
    Image.fromarray(target_image).save(path+"target.png")
    Image.fromarray(pred_image).save(path+"pred.png")


def main():
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--modelpath', default="")
    parser.add_argument('--datapath', default='LPCVCDataset')
    parser.add_argument('--dev', default="cuda:0")
    parser.add_argument('--save_images_path', default="")
    args = parser.parse_args()
    args.device = torch.device(args.dev)

    print("Finish parsing")

    torch.cuda.empty_cache()

    print(args.device)

    #model = torch.load('/home/jetson/Desktop/LPCVC2023/tes2')
    print("Loading model...")
    #model = FastSCNN(nclass=14).to(args.device)
    model = smp.PSPNet(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        classes=N_CLASSES, 
    ).cuda()
    print("Loading weights...")
    model.load_state_dict(torch.load('/home/jetson/Desktop/LPCVC2023/final_model.pt'))    #model= torch.load(args.modelpath, map_location=args.device)
    model.eval()
    print("Finish load model")
    model_int8 = torch.ao.quantization.quantize_dynamic(
                model,  # the original model
                {torch.nn.Linear},  # a set of layers to dynamically quantize
                dtype=torch.qint8)  # the target dtype for quantized weights
    model_int8.cuda()
    model_int8.eval()
    print("Finish Quantization")

    transform = T.Compose([T.ToTensor(),T.Resize(size=IMG_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)])
    print("Loading dataset...")
    img = Image.open('/home/jetson/Desktop/LPCVC2023/dataset/val/IMG/val_' + str(0).zfill(4) + '.png').convert('RGB')
    print("finish load image")
    img = np.array(img)
    img = transform(img)
    t = T.Compose([T.Normalize(mean, std)])
    img = t(img)
    print("finish aug")
    x = torch.tensor(np.expand_dims(img, axis=0)).cuda()
    print("finish cuda")

    #x = torch.ones((1, 3, 256, 256)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x])
    print("Finish TensorRT")
    #model.eval()


    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print("finish load sample")
    #image = torch.tensor(np.expand_dims(img, axis=0)).to(args.device)
    print("transform to tensor")
    with torch.no_grad():
        for i in range(30):
            starter.record()
            output = model_trt(x)
            ender.record()
            torch.cuda.synchronize()
            print(starter.elapsed_time(ender))
    print("predict to tensor")
    print(output.shape)
    #output_vis = output.cpu().data.max(1)[1].numpy()

    # print("finish loas dataset", len(val_dataset))
    # print("evaluate the model")
    # val_time, saved_images = eval(model, args, val_loader, memory_allocated_prev)
    # print(torch.cuda.memory_summary())
    # print("Total memory used : {:.2f} MB".format(torch.cuda.memory_allocated(args.device) / (1024 * 1024)))
    # print("inf_time: {:.3f} ms, mean_dice: {:.3f}, score: {:.3f}".format(val_time, accuracyTrackerVal.get_mean_dice(), 1000*accuracyTrackerVal.get_mean_dice()/val_time))
    # save_images(saved_images, args.save_images_path)

if __name__ == '__main__':
    main()
