import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
import random


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load data
data = datasets.ImageFolder(root='./images', transform=transform)
data_loader = DataLoader(data, batch_size=32, shuffle=True)

# Import pre-trained model
model = torchvision.models.shufflenet_v2_x1_0(weights=None)

num_classes = len(data.classes)
model.fc = nn.Linear(1024, num_classes)

state_dict = torch.load('./shufflenet_model.pth')
model.load_state_dict(state_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

feature_map = []  


def forward_hook(module, inp, outp):
    feature_map.append(outp)


# Register forward hook for the last layer
for module in model.children():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(forward_hook)
        break  

grad = []  

def backward_hook(module, grad_in, grad_out):
    grad.append(grad_out[0])


# Register backward hook for the last layer
for module in model.children():
    if isinstance(module, nn.Conv2d):
        module.register_backward_hook(backward_hook)
        break  

def _normalize(cams: Tensor) -> Tensor:
    """CAM normalization"""
    cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
    cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))

    return cams

def overlay_mask(img: Image.Image, mask: Tensor, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError('img argument needs to be a PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.detach().numpy()
    overlay = (255 * cmap(overlay ** 2)[:, :, 1:]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


with torch.no_grad():
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        idx = random.randint(0, len(images) - 1)
        sample_image = images[idx]
        sample_label = labels[idx]

        outputs = model(images)

        orign_img = transforms.ToPILImage()(sample_image.cpu())  # Convert tensor to PIL Image

        out = model(images)  
        cls_idx = torch.argmax(out).item()  
        score = out[:, cls_idx].sum()  

        with torch.enable_grad():
            model.zero_grad()
            score.backward(retain_graph=True)  

        weights = grad[0].squeeze(0).mean(dim=(1, 2)) 

        grad_cam = (weights.view(*weights.shape, 1, 1) * feature_map[0].squeeze(0)).sum(0)
        grad_cam = _normalize(F.relu(grad_cam, inplace=True)).cpu()
        mask = grad_cam
        result = overlay_mask(orign_img, mask)
        result.show()







