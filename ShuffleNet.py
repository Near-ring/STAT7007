
from utils_common import *

data, train_loader, test_loader = load_images('./images', test_ratio=0.2, batch_size=32, shuffle=True)

model = torchvision.models.shufflenet_v2_x1_0(weights=None)
num_classes = len(data.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 8
train_losses, train_accs = train_model(model, train_loader, optimizer, num_epochs)

plot_train_curve(train_losses, train_accs)

torch.save(model.state_dict(), 'shufflenet_model.pth')

test_report(model, "shufflenet_model.pth", test_loader, data)

eval_model(model, test_loader)

# plt.figure()
# plt.plot(range(1, len(test_losses) + 1), test_losses, label='Test Loss')
# plt.xlabel('Batch')
# plt.ylabel('Loss')
# plt.title('Test Loss')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(range(1, len(test_accs) + 1), test_accs, label='Test Accuracy')
# plt.xlabel('Batch')
# plt.ylabel('Accuracy')
# plt.title('Test Accuracy')
# plt.legend()
# plt.show()

# correct_count, total_count = 0, 0
#
# for images, labels in test_loader:
#     images, labels = images.to(device), labels.to(device)
#
#     idx = random.randint(0, len(images) - 1)
#     sample_image = images[idx].unsqueeze(0)
#     sample_label = labels[idx]
#
#     with torch.no_grad():
#         outputs = model(sample_image)
#         _, predicted = torch.max(outputs, 1)
#
#     predicted_label = predicted.item()
#     actual_label = sample_label.item()
#     correct = 'Correct' if predicted_label == actual_label else 'Incorrect'
#
#     test_data_classes = data.classes
#
#     plt.figure()
#     sample_image = sample_image.cpu().squeeze().numpy()
#     sample_image = np.transpose(sample_image, (1, 2, 0))
#     sample_image = 0.5 * sample_image + 0.5
#     plt.title(
#         f"Actual Label: {test_data_classes[actual_label]}, Predicted Label: {test_data_classes[predicted_label]} ({correct})")
#     plt.imshow(sample_image)
#
#     if predicted_label == actual_label:
#         correct_count += 1
#     total_count += 1
#
# accuracy = correct_count / total_count * 100
# print(f"Accuracy of randomised sampling tests: {accuracy:.2f}%")
