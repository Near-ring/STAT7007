import torchvision.models.mobilenetv2

from utils_common import *
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM, CAM, LayerCAM, ScoreCAM
from torchvision.io.image import read_image
from torchvision import transforms
# from skimage.transform import resize
from torchvision.transforms.functional import normalize, resize, to_pil_image


data, train_loader, test_loader = load_images('./images', test_ratio=0.2, batch_size=32, shuffle=True)

num_classes = len(data.classes)
# model = torchvision.models.MNASNet(alpha=1.0, num_classes=num_classes)
model = torchvision.models.vgg16(weights=None, num_classes=num_classes)
# model = torchvision.models.MobileNetV2(num_classes=num_classes)
# model = torchvision.models.resnet34(weights=None, num_classes=num_classes)
# model = torchvision.models.shufflenet_v2_x1_0(weights=None)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = torchvision.models.squeezenet1_0(weights=None)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# cam = GradCAM(model)

model_name = 'vgg16'

num_epochs = 17
train_losses, train_accs = train_model(model, train_loader, optimizer, num_epochs)
torch.save(model.state_dict(), f'{model_name}.pth')
test_report(model, f'{model_name}.pth', test_loader, data)
eval_model(model, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(f'{model_name}.pth'))
model.to(device)
test_report(model, f'{model_name}.pth', test_loader, data)
eval_model(model, test_loader)


# img = read_image('./images/punching_hole/img_02_425506400_00018.jpg')
# input_tensor = img.repeat(3, 1, 1)
# img = input_tensor
# input_tensor = resize(input_tensor, (224, 224)) / 255
# # input_tensor = normalize(input_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# input_tensor = input_tensor.unsqueeze(0)
# print(input_tensor.shape)
# #input_tensor = input_tensor.to(device)
#
# out = model(input_tensor)
# print(out.shape)
# activation_map = cam(out.squeeze(0).argmax().item(), out)
# print(activation_map[0].shape)
# print(activation_map[0].squeeze(0).shape)
#
# amap_0 = activation_map[0].squeeze(0)
# heatmap_resized = torch.nn.functional.interpolate(amap_0.unsqueeze(0).unsqueeze(0),
#                                                     size=input_tensor.shape[2:],
#                                                     mode='bilinear',
#                                                     align_corners=False).squeeze(0).squeeze(0)
# # heatmap_resized = resize(amap_0, (input_tensor.shape[2], input_tensor.shape[3]), anti_aliasing=True)
#
# plt.imshow(heatmap_resized.numpy())
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# plt.imshow(input_tensor.squeeze(0).permute(1, 2, 0).numpy())
# plt.axis('off')
# plt.tight_layout()
# plt.show()
#
# aaa = out.squeeze(0).argmax().item()