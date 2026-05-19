import io
import pickle
import cv2
import numpy as np
import streamlit as st
import torch
import timm
from PIL import Image
from torchvision import transforms
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Real vs AI Image Detector",
    page_icon="🔍",
    layout="centered"
)
 
# ─────────────────────────────────────────────────────────────
# CPU unpickler — handles models saved on GPU machines
# ─────────────────────────────────────────────────────────────
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(
                io.BytesIO(b), map_location='cpu', weights_only=False
            )
        return super().find_class(module, name)
 
 
# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_face_model():
    """
    Notebook 1 — 140K Real & Fake Faces
    Saved as: efficientnet_faces.pkl
    Specialist: human faces & portraits
    """
    with open('./models/efficientnet_faces.pkl', 'rb') as f:
        info = CPUUnpickler(f).load()
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    model.load_state_dict(info['state_dict'])
    model.eval()
    return model, info
 
 
@st.cache_resource
def load_general_model():
    """
    Notebook 2 — Defactify (HuggingFace)
    Saved as: efficientnet_general.pkl
    Specialist: general scenes, objects, nature
    """
    with open('./models/efficientnet_general.pkl', 'rb') as f:
        info = CPUUnpickler(f).load()
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
    model.load_state_dict(info['state_dict'])
    model.eval()
    return model, info
 
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def detect_face(pil_image, coverage_threshold=0.10):
    """
    Uses OpenCV Haar Cascade to detect faces.
    Only returns True if a face covers at least 10% of the image
    — filters out false positives on flowers, animals, objects.
    Returns (is_portrait: bool, face_boxes: list)
    """
    img_w, img_h = pil_image.size
    img_area     = img_w * img_h
 
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
 
    cascade_frontal = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cascade_profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )
 
    faces_frontal = cascade_frontal.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(80, 80)
    )
    faces_profile = cascade_profile.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(80, 80)
    )
 
    if len(faces_frontal) > 0 and len(faces_profile) > 0:
        faces = np.concatenate([faces_frontal, faces_profile])
    elif len(faces_frontal) > 0:
        faces = faces_frontal
    elif len(faces_profile) > 0:
        faces = faces_profile
    else:
        return False, []
 
    largest_face_area = max(w * h for (x, y, w, h) in faces)
    coverage          = largest_face_area / img_area
 
    if coverage < coverage_threshold:
        return False, faces
 
    return True, faces
def predict(pil_image, model):
    """
    Runs inference. Both models use:
      class_names = ['FAKE', 'REAL']  →  0=FAKE, 1=REAL
    """
    transform  = get_transform()
    tensor     = transform(pil_image).unsqueeze(0)
 
    model.eval()
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        pred  = out.argmax(dim=1).item()
 
    label  = 'REAL' if pred == 1 else 'FAKE'
    confidence = probs[pred].item() * 100
    real_prob  = probs[1].item() * 100
    fake_prob  = probs[0].item() * 100
 
    return label, confidence, real_prob, fake_prob
 

st.title("🔍 Real vs AI Image Detector")
st.markdown("""
This app uses **two EfficientNet-B0 models** working together:
- 🧑 A **face specialist** trained on 140,000 real and StyleGAN-generated faces
- 🌍 A **scene specialist** trained on 96,000 MS COCO and AI-generated scenes (DALL-E 3, Midjourney v6, SD3)
 
The app automatically detects which type of image you upload and routes it to the right model.
""")
 
st.divider()
 
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Info")
 
    st.markdown("**🧑 Face Specialist**")
    st.caption("Notebook 1 — 140K Real & Fake Faces")
    st.metric("Dataset",     "140,000 face images")
    st.metric("Fake source", "StyleGAN")
    st.metric("Accuracy",    "98.38%")
    st.metric("AUC-ROC",     "0.9991")
 
    st.divider()
 
    st.markdown("**🌍 Scene Specialist**")
    st.caption("Notebook 2 — Defactify (HuggingFace)")
    st.metric("Dataset",     "96,000 scene images")
    st.metric("Fake source", "SD3 · DALL-E 3 · MJ v6")
 
    st.divider()
 
    st.markdown("**🔀 Routing Logic**")
    st.markdown("""
    1. Upload image
    2. OpenCV scans for faces
    3. **Face found** → 🧑 Face specialist
    4. **No face** → 🌍 Scene specialist
    5. Result + confidence shown
    """)
 
    st.divider()
    st.markdown("**Architecture:** EfficientNet-B0")
    st.markdown("**Parameters:** 4M per model")
    st.markdown("**Input size:** 224×224 px")
 
# ── Upload ────────────────────────────────────────────────────
st.subheader("📤 Upload an image")
uploaded = st.file_uploader(
    "Drag and drop or click to upload",
    type=['jpg', 'jpeg', 'png', 'webp']
)
 
if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')
 
    # ── Face detection & model routing ───────────────────────
    has_face, face_boxes = detect_face(image)
 
    if has_face:
        model_label = "🧑 Face Specialist"
        model_desc  = "Trained on 140K real & StyleGAN faces"
        specialist  = 'face'
        model, info = load_face_model()
    else:
        model_label = "🌍 Scene Specialist"
        model_desc  = "Trained on Defactify (SD3, DALL-E 3, Midjourney v6)"
        specialist  = 'general'
        model, info = load_general_model()
 
    # ── Layout ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
 
    with col1:
        # Draw face bounding boxes if detected
        if has_face and len(face_boxes) > 0:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for (x, y, w, h) in face_boxes:
                cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 200, 100), 3)
            img_annotated = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            st.image(img_annotated, caption='Uploaded image (face detected)', use_container_width=True)
        else:
            st.image(image, caption='Uploaded image', use_container_width=True)
 
    with col2:
        st.info(f"**Model selected:** {model_label}\n\n{model_desc}")
 
        with st.spinner('Analyzing...'):
            try:
                label, confidence, real_prob, fake_prob = predict(image, model)
            except FileNotFoundError as e:
                st.error(f"Model file not found: {e}\n\nMake sure both .pkl files are in the same folder as app.py")
                st.stop()
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.stop()
 
        if label == 'REAL':
            st.success("✅ **REAL IMAGE**")
        else:
            st.error("🚨 **FAKE / AI-GENERATED**")
 
        st.metric("Confidence", f"{confidence:.1f}%")
        st.divider()
 
        st.markdown("**Class probabilities:**")
        st.markdown(f"🟢 REAL: **{real_prob:.1f}%**")
        st.progress(real_prob / 100)
        st.markdown(f"🔴 FAKE: **{fake_prob:.1f}%**")
        st.progress(fake_prob / 100)
 
    # ── Routing decision panel ────────────────────────────────
    st.divider()
    st.subheader("🔀 Routing Decision")
 
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown("**Step 1 — Face Detection**")
        if has_face:
            st.success(f"✅ {len(face_boxes)} face(s) detected")
        else:
            st.warning("❌ No face detected")
    with r2:
        st.markdown("**Step 2 — Model Selected**")
        st.info(model_label)
    with r3:
        st.markdown("**Step 3 — Prediction**")
        if label == 'REAL':
            st.success(f"REAL ({confidence:.1f}%)")
        else:
            st.error(f"FAKE ({confidence:.1f}%)")
 
    # ── Confidence interpretation ─────────────────────────────
    st.divider()
    st.subheader("📖 Interpretation")
 
    if confidence >= 95:
        st.info(f"The model is **very confident** ({confidence:.1f}%) in this prediction.")
    elif confidence >= 80:
        st.warning(f"The model is **moderately confident** ({confidence:.1f}%). Treat with some caution.")
    else:
        st.warning(f"The model is **uncertain** ({confidence:.1f}%). This image has features of both classes.")
 
    if label == 'FAKE':
        if specialist == 'face':
            st.markdown("""
            **What makes this face likely AI-generated:**
            - Unnaturally smooth or perfectly symmetric features
            - Subtle artifacts in eyes, teeth, or hair boundaries
            - Overly perfect skin texture (StyleGAN signature)
            - Unnatural background blending around the face
            """)
        else:
            st.markdown("""
            **What makes this image likely AI-generated:**
            - Unnatural texture patterns in backgrounds or objects
            - Unusual color distributions (too vivid or too flat)
            - Subtle artifacts in fine details and edges
            - Inconsistent lighting or shadows across the scene
            """)
    else:
        st.markdown("""
        **What makes this image likely real:**
        - Natural noise and grain patterns
        - Realistic lighting and shadows
        - Organic imperfections in textures
        - Consistent color distribution
        """)
 
    st.divider()
    if specialist == 'face':
        st.caption("⚠️ Face specialist trained on StyleGAN faces. May be less accurate on very recent AI face generators (DALL-E 3, Midjourney faces).")
    else:
        st.caption("⚠️ Scene specialist trained on Defactify (SD3, DALL-E 3, Midjourney v6). Best on everyday scenes and objects.")
 
else:
    # ── Placeholder ───────────────────────────────────────────
    st.info("👆 Upload an image above to get started")
 
    st.subheader("💡 What to try:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Real images (should say REAL):**
        - Photos from your phone camera
        - Selfies or portraits
        - News photographs
        - Nature or travel photos
        """)
    with col2:
        st.markdown("""
        **AI images (should say FAKE):**
        - Midjourney generated images
        - DALL-E 3 outputs
        - Stable Diffusion images
        - StyleGAN faces
        """)
 
    st.divider()
    st.subheader("🔀 How the routing works:")
    st.code("""
Upload Image
      ↓
OpenCV Face Detection
   ↙              ↘
Face found       No face found
   ↓                  ↓
🧑 Face Model    🌍 Scene Model
(140K faces)     (Defactify)
   ↘              ↙
    REAL or FAKE
    """)