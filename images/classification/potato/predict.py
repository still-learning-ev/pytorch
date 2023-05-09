import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

potato = ['/kaggle/input/plantvillage-dataset/color/Potato___healthy',
                '/kaggle/input/plantvillage-dataset/color/Potato___Late_blight',
                '/kaggle/input/plantvillage-dataset/color/Potato___Early_blight']
class CNNPotato(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64,stride=1, padding=0, kernel_size=3 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=len(potato))
        )
    def conv_operation(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x
    def classifier(self, x):
        x = self.dense(x)
        return x
    
    def forward(self, x):
        x = self.conv_operation(x)
        x = self.classifier(x)
        
        return x

def predict_image(image):
    # input numpy array
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((256, 256), antialias=True)])
    image = transform(image).unsqueeze(0)

    print(image.shape, image.min(), image.max())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNNPotato().to(device)
    
    model.load_state_dict(torch.load('C:\\Users\\lonex\\OneDrive\\Desktop\\GIT\\pytorch\\potato\\models\\CNN-potato-disease-1.pth', map_location=torch.device(device)))

    predicted_data = model(image.to(device))
    # print(predicted_data)
    predict_label = torch.argmax(torch.softmax(predicted_data, dim=1))
    confidence = round((torch.softmax(predicted_data, dim=1).max().item() * 100), 2)
    
    class_name = [i.split('/')[-1] for i in potato]

    return confidence, predict_label.item(), class_name[predict_label.item()]
    # fig, ax = plt.subplots()
    # ax.plot()
    # plt.imshow(image.squeeze().permute(1,2,0))
    # plt.title(f"Predicted--> [{class_name[predict_label.item()]}]\n")
    # plt.xlabel(f"Confidence = [{confidence}]\n Class = [{predict_label.item()}]")
    # # plt.show()
    # img_bytes = io.BytesIO()
    # plt.savefig(img_bytes, format='png')
    # # img_bytes.seek(0)
    # return img_bytes