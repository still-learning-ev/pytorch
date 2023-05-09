import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import os

def predict_image(image):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Transform the image
    transform_to_tensor = transforms.Compose([transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize(mean=[0], std=[1]), transforms.Resize((244, 244), antialias=True)])

    # image transform
    image = transform_to_tensor(image)
    model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
    
    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models/model-alexnet-transfer.pt'), map_location=torch.device(device)))

    print("here")
    pred = model(image.unsqueeze(dim=0).to(device))
    pred_label = torch.argmax(torch.softmax(pred, dim=1)).item()
    

    title = 'Dog' if pred_label == 0 else 'Cat'
    
    plt.imshow(image.permute(1,2,0))
    plt.title(f"pred {title}")
    
    return pred_label, title
