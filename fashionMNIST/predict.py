import torch
import torch.nn as nn
import numpy as np 

class FashionMNISTModelV1(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=84, kernel_size=5, stride=1, padding=1),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=84, out_channels=100, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=10),
            #nn.Softmax(dim=1)
        )
       

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


def predict_image(image):
    image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
    image = image / 255

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FashionMNISTModelV1().to(device)
    
    model.load_state_dict(torch.load('C:/Users/lonex/OneDrive/Desktop/GIT/pytorch/fashionMNIST/model/FashionMNIST_CNN.pth', map_location=torch.device(device)))

    predicted_data = model(image.to(device))
    

    predict_label = torch.argmax(torch.softmax(predicted_data, dim=1))

    class_name = {
        0 : 'T-shirt/top',
        1 : 'Trouser',
        2 : 'Pullover',
        3 : 'Dress',
        4 : 'Coat',
        5 : 'Sandal',
        6 : 'Shirt',
        7 : 'Sneaker',
        8 : 'Bag',
        9 : 'Ankle boot',
    }

    return predict_label.item(), class_name[predict_label.item()]


if __name__ == "__main__":
    print(predict_image(np.random.random_sample(size=(28,28))))
