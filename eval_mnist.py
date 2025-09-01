import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. Define your CNN architecture (same as training)
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Recreate model
model = ConvolutionalNetwork()

# 3. Load weights into it
model.load_state_dict(torch.load("mnist_cnn.pth"))  # <-- state_dict file
model.eval()

# 4. Test on one MNIST image
transform = transforms.ToTensor()
test_data = datasets.MNIST(root="./cnn_data", train=False, download=True, transform=transform)

img, label = test_data[1947]  # pick an image
with torch.no_grad():
    prediction = model(img.view(1, 1, 28, 28))
    predicted_class = prediction.argmax(dim=1).item()

print(f"True label: {label}, Predicted: {predicted_class}")

# Optional: show the image
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Predicted: {predicted_class}, True: {label}")
plt.show()

