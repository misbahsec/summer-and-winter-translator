import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
import os

# 1. Device Setup (Uses GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Model Architecture (Generator)
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorResNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super(GeneratorResNet, self).__init__()
        model = [
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]

        for _ in range(n_residual_blocks):
            model += [ResnetBlock(256)]

        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]

        model += [nn.Conv2d(64, output_channels, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# 3. Function to load a specific model file
def load_specific_model(filename):
    model = GeneratorResNet().to(device)
    
    # Path to your folder containing the .pth files
    folder_name = 'cyclegan-summer2winter-pytorch-default-v1'
    model_path = os.path.join(folder_name, filename)
    
    if os.path.exists(model_path):
        try:
            # Load weights (map_location ensures it works on CPU if CUDA is missing)
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle dictionary-style checkpoints
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            print(f"Successfully loaded: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    else:
        print(f"Warning: {filename} not found in '{folder_name}' folder!")
    
    model.eval()
    return model

# 4. Load both models into memory (to avoid reloading during runtime)
print("Initializing Models...")
model_winter_to_summer = load_specific_model('generator_B2A.pth') # Winter -> Summer
model_summer_to_winter = load_specific_model('generator_A2B.pth') # Summer -> Winter

# 5. Image Processing Pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def denormalize(img_tensor):
    """Convert tensor back to PIL image"""
    img = img_tensor.cpu().detach().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# 6. Prediction Function
def predict_image(input_img, direction):
    if input_img is None:
        return None
    
    # Preprocess the input image
    img_t = input_img.convert('RGB')
    input_tensor = transform(img_t).unsqueeze(0).to(device)
    
    # Select model based on user selection
    if direction == "Winter ➡ Summer":
        active_model = model_winter_to_summer
    else:
        active_model = model_summer_to_winter
    
    # Generate image
    with torch.no_grad():
        output_tensor = active_model(input_tensor)[0]
    
    # Post-process and return the output
    output_img = denormalize(output_tensor)
    return output_img

# 7. Gradio UI Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Radio(["Winter ➡ Summer", "Summer ➡ Winter"], label="Select Conversion Direction", value="Winter ➡ Summer")
    ],
    outputs=gr.Image(type="pil", label="Generated Image"),
    title="Winter-Summer Image Transformer",
    description="Transform images from Winter to Summer or Summer to Winter."
)

if __name__ == "__main__":
    interface.launch()