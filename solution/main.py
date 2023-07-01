import torch
from argparse import ArgumentParser, Namespace
from torchvision import transforms as T
from typing import List
import numpy as np
from PIL import Image
from torch2trt import torch2trt
import torch.nn.functional as F
import pkg_resources
import os

IMG_SIZE = 256
SIZE: List[int] = [512, 512]
mean = [0.4607, 0.4558, 0.4192]
std = [0.1855, 0.1707, 0.1769]

def getArgs() -> Namespace:
    # NOTE: These variables can be changed
    programName: str = "LPCVC 2023 Sample Solution"
    authors: List[str] = ["Noé Vernier","Jacques Sun", "Ulysse Ristocelli, Hong Fan"]

    prog: str = programName
    usage: str = f"This is the {programName}"
    description: str = f"This {programName} does create a single segmentation map of arieal scenes of disaster environments captured by unmanned arieal vehicles (UAVs)"
    epilog: str = f"This {programName} was created by {''.join(authors)}"

    # NOTE: Do not change these flags
    parser: ArgumentParser = ArgumentParser(prog, usage, description, epilog)
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Filepath to an image to create a segmentation map of",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Filepath to the corresponding output segmentation map",
    )

    return parser.parse_args()

def loadImageToTensor(imagePath):
    transform = T.Compose([T.ToTensor(),T.Resize(size=IMG_SIZE, interpolation=T.functional.InterpolationMode.NEAREST)])
    img = Image.open(imagePath).convert('RGB')
    x = np.array(img)
    x = transform(x)
    t = T.Compose([T.Normalize(mean, std)])
    x = t(x)
    return torch.tensor(np.expand_dims(x, axis=0)).cuda()


def main():
    args = getArgs()
    image_files: List[str] = os.listdir(args.input)
    
    torch.cuda.empty_cache()
    
    modelPath: str = "model.pkl"
    with pkg_resources.resource_stream(__name__, modelPath) as model_file:
        #------------------------------------------------------------   
        model = torch.load(model_file)
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        #------------------------------------------------------------
        model_int8 = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model_int8.to(device)
        model_int8.eval()
        #------------------------------------------------------------
        input_image_path: str = os.path.join(args.input,image_files[0])
        x = loadImageToTensor(imagePath=input_image_path)
        #------------------------------------------------------------
        # convert to TensorRT feeding sample data as input
        model = torch2trt(model, [x])
        #------------------------------------------------------------
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        #Warm Up
        for image_file in image_files[:15]:
            input_image_path: str = os.path.join(args.input, image_file)
            imageTensor: torch.Tensor = loadImageToTensor(imagePath=input_image_path)
            imageTensor = imageTensor.to(device)
            outTensor: torch.Tensor = model(imageTensor)

        time = 0
        with torch.no_grad():
            # MEASURE PERFORMANCE
            for image_file in image_files:
                input_image_path: str = os.path.join(args.input, image_file)
                output_image_path: str = os.path.join(args.output, image_file)
                imageTensor: torch.Tensor = loadImageToTensor(imagePath=input_image_path)
                imageTensor = imageTensor.to(device)
                starter.record()
                output = model(imageTensor)
                ender.record()
                torch.cuda.synchronize()
                
                time += starter.elapsed_time(ender)
                
                outTensor: torch.Tensor = F.interpolate(
                    outTensor, SIZE, mode="bilinear", align_corners=True
                )

                outArray: np.ndarray = outTensor.cpu().data.max(1)[1].numpy()
                outArray: np.ndarray = outArray.astype(np.uint8)

                outImage: np.ndarray = np.squeeze(outArray, axis=0)
                outImage = Image.fromarray(outImage, mode='L')
                outImage.save(output_image_path)
                
        del model
        del imageTensor
        del outTensor
        torch.cuda.empty_cache()
        model_file.close()
        
        print("[Finish measuring performance]")
        print("___________________________________________________")
        print("______________________RESULTS______________________")
        print("___________________________________________________")

        print("Total time = " + str(time) + " ms")
        print("Mean Inference Time = " + str(round(time/len(image_files), 3)) + " ms")