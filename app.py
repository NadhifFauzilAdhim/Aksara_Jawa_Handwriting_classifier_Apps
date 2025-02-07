import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# 1. Daftar Kelas
class_names = [
    'ba', 'ca', 'da', 'dha', 'ga', 'ha', 'ja', 'ka', 'la', 'ma', 'na',
    'nga', 'nya', 'pa', 'ra', 'sa', 'ta', 'tha', 'wa', 'ya'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Fungsi Memuat Model
@st.cache_resource
def load_model():
    model_path = './model/hancaraka_Model.pth'
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        model = torch.load(model_path, map_location=device)
    
    model.to(device)
    model.eval()
    return model

model = load_model()

# 3. Preprocessing Gambar
def square_padding(image, size=128):
    old_size = image.size  # (width, height)
    
    if min(old_size) == 0:  # Cek jika gambar kosong
        raise ValueError("Gambar yang diberikan memiliki ukuran tidak valid (0x0).")

    ratio = float(size) / max(old_size)
    new_size = tuple([max(1, int(x * ratio)) for x in old_size])  # Cegah ukuran 0

    image = image.resize(new_size, Image.Resampling.LANCZOS)

    new_image = Image.new("RGB", (size, size), (255, 255, 255))
    new_image.paste(image, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return new_image


def preprocess_image(image):
    image = square_padding(image)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

# 4. Fungsi Segmentasi Karakter
def segment_characters(image):
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    filtered_contours = sorted(filtered_contours, key=lambda c: cv2.boundingRect(c)[0])
    
    characters = []
    boxes = []
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        roi = image_cv[y:y+h, x:x+w]
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        characters.append(roi_pil)
        boxes.append((x, y, w, h))
    
    return characters, boxes, image_cv

# 5. Fungsi Batch Prediction
def batch_predict(characters):
    inputs = torch.cat([preprocess_image(char) for char in characters], dim=0)
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = F.softmax(outputs, dim=1)
        top5_prob, top5_catid = torch.topk(probabilities, 5, dim=1)

    results = []
    for i in range(len(characters)):
        top5 = [(class_names[top5_catid[i, j]], top5_prob[i, j].item() * 100) for j in range(5)]
        results.append(top5)
    
    return results

# 6. Antarmuka Streamlit
st.title('🔍 Klasifikasi Aksara Jawa dengan ResNet18')
st.write('Deteksi dan klasifikasi multiple aksara Jawa dalam satu gambar')

uploaded_file = st.file_uploader("📤 Unggah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼 Gambar yang Diunggah', use_column_width=True)
    
    characters, boxes, orig_image = segment_characters(image)
    
    if len(characters) == 0:
        st.error("Tidak ada karakter yang terdeteksi!")
    else:
        top5_list = batch_predict(characters)
        
        # Visualisasi hasil
        st.subheader("🔍 Hasil Deteksi")
        fig, ax = plt.subplots()
        output_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        ax.imshow(output_image)
        
        for (x, y, w, h), pred in zip(boxes, [t[0][0] for t in top5_list]):
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y-10, pred, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.axis('off')
        st.pyplot(fig)
        
        # Detail prediksi
        st.subheader("📊 Detail Prediksi per Karakter")
        for i, (char, preds) in enumerate(zip(characters, top5_list)):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(char, caption=f'Karakter {i+1}', width=100)
            with col2:
                st.write(f"**Prediksi Teratas:** {preds[0][0]} ({preds[0][1]:.2f}%)")
                fig2, ax2 = plt.subplots(figsize=(3, 2))
                names = [p[0] for p in preds]
                probs = [p[1] for p in preds]
                ax2.barh(names[::-1], probs[::-1], color='lightblue')
                ax2.set_xlabel('Probabilitas (%)')
                ax2.set_xlim(0, 100)
                st.pyplot(fig2)

st.markdown("""
### **📌 Catatan Penggunaan:**
1. Gambar harus memiliki latar belakang kontras dengan aksara
2. Setiap aksara harus terpisah (tidak saling menempel)
3. Hasil deteksi bergantung pada kualitas gambar
""")