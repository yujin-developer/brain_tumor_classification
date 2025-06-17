import streamlit as st
import torch
import timm
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io

from PIL import Image
from torchvision import transforms


# Load trained-model pkl
metadata = joblib.load("Xception_metadata.pkl")
input_size = metadata.get("input_size", (224, 224))
class_names = metadata["class_names"]  # glioma, meningioma, notumor, pituitary

# Use Xception model 
model = timm.create_model("xception", pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("Xception_weights.pth", map_location="cpu"))
model.eval()

# image preprocessing (same with eval)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def preprocess(image):
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)

def grad_cam(input_tensor, model, target_layer):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # hook
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    score = output[:, pred_class]
    score.backward()

    grad = gradients[0][0]           # (C, H, W)
    activation = activations[0][0]   # (C, H, W)

    # weights
    weights = grad.mean(dim=(1, 2))
    cam = (weights[:, None, None] * activation).sum(dim=0)
    cam = np.maximum(cam.cpu().numpy(), 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)  # Normalize

    # Hook remove
    handle_fw.remove()
    handle_bw.remove()
    return cam

def show_cam_on_image(img, mask, alpha=0.5):
    # img: PIL.Image
    img = np.array(img.resize(input_size)) / 255.0  # (input_size, input_size, 3)
    
    # if mask is small(8x8), img size upsample
    from PIL import Image as PILImage
    mask_img = PILImage.fromarray(np.uint8(255 * mask))
    mask_img = mask_img.resize((img.shape[1], img.shape[0]), resample=PILImage.BILINEAR)
    mask = np.array(mask_img) / 255.0  # (input_size, input_size)

    heatmap = plt.get_cmap("jet")(mask)[:, :, :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.uint8(255 * overlay)
    return overlay

# Streamlit UI
st.title("Brain Tumor Classifier (XceptionNet)")

uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=250)

    if st.button("Diagnose"):
        # Image preprocessing
        input_tensor = preprocess(image)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)[0]
            conf, pred = torch.max(prob, 0)

        # Result
        st.markdown(f"**Predicted Result:** {class_names[pred]} ({conf.item()*100:.2f}%)")

        # Grad-CAM
        # timm==0.6.13
        target_layer = model.conv4.pointwise
        cam = grad_cam(input_tensor, model, target_layer)
        cam_img = show_cam_on_image(image, cam)

        # Grad-CAM heatmap
        st.markdown("**XAI Heatmap (Grad-CAM):**")
        st.image(cam_img, caption="Grad-CAM Heatmap", width=250)
