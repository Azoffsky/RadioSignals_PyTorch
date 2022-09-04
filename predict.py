# USAGE
# python predict.py --image images/chirp_55.jpg --model output\model.pth

import torchvision.transforms as transforms
import torch
import argparse
import cv2
import os

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True)
ap.add_argument('-m', '--model', required=True)
args = vars(ap.parse_args())

# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# list containing all the class labels
labels = ['chirp', 'm-seq', 'simple']

# initialize the model and load the trained weights
model = torch.load(args["model"]).to(device)
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# read and preprocess the image
image = cv2.imread(args['image'])
# get the ground truth class
gt_class = args['image'].split('/')[-1][:6]
# get the image copy
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]
cv2.putText(orig_image,
            f"GT: {gt_class}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2, cv2.LINE_AA
            )
cv2.putText(orig_image,
            f"Pred: {pred_class}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2, cv2.LINE_AA
            )
print(f"GT: {gt_class}, pred: {pred_class}")
cv2.imwrite(f"outputs/{args['image'].split('/')[-1].split('.')[0]} as {gt_class}_.png", orig_image)
cv2.imshow('Result', orig_image)
cv2.waitKey(0)
