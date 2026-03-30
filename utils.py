from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


#Global variable
trained_model = None



#Load the pretrained Resnet model
class DeepFakeClassifierResNet(nn.Module):

    def __init__(self, num_classes = 1, dropout_rate = 0.3):
        super().__init__()

        self.model = models.resnet50(weights= "DEFAULT")

        for params in self.model.parameters():
            params.requires_grad = False

        for params in self.model.layer4.parameters():
            params.requires_grad = True


        self.model.fc = nn.Sequential(

            nn.Dropout(p = dropout_rate),
            nn.Linear(in_features= self.model.fc.in_features, out_features= num_classes)
        )

    def forward(self, x):

        x = self.model(x)

        return x
    


#Helper Function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = DeepFakeClassifierResNet()
        trained_model.load_state_dict(torch.load("./artifacts/model.pth", map_location= torch.device("cpu")))
        trained_model.eval()


    with torch.inference_mode():
        outputs = trained_model(image_tensor)
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float().squeeze(1)

        return "FAKE" if predicted.item() == 0 else "REAL"

