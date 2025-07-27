import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
import shutil
import time
import pickle
from scipy.spatial.distance import cosine, euclidean

st.set_page_config(page_title="Multi-Face Recognition")

#import streamlit as st
from PIL import Image

# Load your image
image = Image.open("MukhID-removebg-preview.png")  # replace with your file


# Create empty columns and place image in the center column
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(image, use_container_width=True)



st.markdown("""
    <h4 style='text-align: center; font-family: Arial;'>
        
            
            
            
            Hello Everyone! Welcome to MukhID, A Human Face Recognition App 
    </h4>
""", unsafe_allow_html=True)


st.markdown("""
    <h1 style='text-align: center; font-family: Helvetica;'>
    Human Face Recognition App
    </h1>
""", unsafe_allow_html=True)

def set_background_image(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
        encoded_img = f"data:image/jpg;base64,{base64.b64encode(img_bytes).decode()}"

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("{encoded_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background
import base64
set_background_image("imag.jpg")

# ---- Input method selection ----
input_method = st.selectbox("Choose input method", ["Upload", "Webcam"])

# ---- Face Mode ----
face_mode = st.radio("Choose Mode", ["Single Face", "Multi-Face"])

#if face_mode == "Single Face":
# ---- Model and distance metric ----
model_name = st.selectbox("Choose model", ["VGG-Face", "Facenet", "ArcFace"])
distance_metric = st.selectbox("Choose distance metric", ["cosine", "euclidean", "euclidean_l2"])

# ---- Define the correct .pkl file to use ----
model_key = model_name.lower().replace("-", "").replace(" ", "")
embedding_file = f"./embeddings/representations_{model_key}.pkl"

# ---- Thresholds ----
thresholds = {
    "Facenet": {"cosine": 0.43, "euclidean": 8.5, "euclidean_l2": 0.85},
    "VGG-Face": {"cosine": 0.43, "euclidean": 12, "euclidean_l2": 1.1},
    "ArcFace": {"cosine": 0.43, "euclidean": 9.5, "euclidean_l2": 0.85}
}

threshold = thresholds.get(model_name, {}).get(distance_metric, 0.4)

# ---- Distance function ----
def calculate_distance(e1, e2, metric):
    if metric == "cosine":
        return cosine(e1, e2)
    elif metric == "euclidean":
        return euclidean(e1, e2)
    elif metric == "euclidean_l2":
        e1 = np.array(e1) / np.linalg.norm(e1)
        e2 = np.array(e2) / np.linalg.norm(e2)
        return euclidean(e1, e2)
    return None

# ---- Offer to add unmatched face ----
def offer_add_to_database(image_path):
    if st.checkbox("Do you want to add this face to the database?"):
        person_name = st.text_input("Enter name for this person (used as filename):")
        if st.button("Confirm and Add"):
            save_name = person_name.strip().lower().replace(" ", "_") + f"_{int(time.time())}.jpg"
            save_path = os.path.join("faces", save_name)
            shutil.copy(image_path, save_path)
            st.success(f"✅ Image saved as `{save_name}` in `faces/` folder.")

            # Recompute embeddings
            with st.spinner("Updating database..."):
                new_reps = []
                for file in os.listdir("faces"):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join("faces", file)
                        try:
                            emb = DeepFace.represent(
                                img_path=img_path,
                                model_name=model_name,
                                detector_backend="opencv",
                                enforce_detection=True
                            )
                            for obj in emb:
                                obj["identity"] = img_path
                                new_reps.append(obj)
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                pd.DataFrame(new_reps).to_pickle(embedding_file)
                st.success("✅ Database updated!")

# ---- Handle input method ----
query_img_path = None

if input_method == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            query_img_path = tmp_file.name
        st.image(query_img_path, caption="Uploaded Image", use_container_width=True)

elif input_method == "Webcam":
    captured_image = st.camera_input("📸 Capture a photo")
    if captured_image:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(captured_image.getvalue())
            query_img_path = tmp_file.name
        st.image(query_img_path, caption="Captured Image", use_container_width=True)

# ---- Recognition logic ----
if query_img_path:
    with st.spinner("Running recognition..."):
        try:
            df = pd.read_pickle(embedding_file)
        except Exception as e:
            st.error(f"❌ Failed to load embeddings: {e}")
            st.stop()

        try:
            detector = "retinaface" if face_mode == "Multi-Face" else "opencv"
            embeddings = DeepFace.represent(
                img_path=query_img_path,
                model_name=model_name,
                detector_backend=detector,
                enforce_detection=True
            )
        except Exception as e:
            st.error(f"❌ Failed to extract embedding(s): {e}")
            st.stop()

        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        for idx, emb in enumerate(embeddings):
            query_embedding = emb["embedding"]
            distances = []

            for _, row in df.iterrows():
                db_embedding = row["embedding"]
                dist = calculate_distance(query_embedding, db_embedding, distance_metric)
                distances.append((row["identity"], dist))

            if distances:
                distances.sort(key=lambda x: x[1])
                identity, distance = distances[0]

                st.subheader(f"🔎 Result for Face #{idx+1}")
                st.markdown(f"**Best Match:** `{os.path.basename(identity)}`")
                st.markdown(f"**Distance:** `{distance:.4f}` (Threshold: `{threshold}`)")

                if distance <= threshold:
                    st.success("✅ Match Accepted 🔓")
                    img = cv2.imread(identity)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    filename = os.path.basename(identity)
                    name_only = os.path.splitext(filename)[0]
                    if "_" in name_only and name_only.split("_")[-1].isdigit():
                        name_only = " ".join(name_only.split("_")[:-1])
                    person_name = name_only.replace("_", " ").title()

                    st.markdown(f"### 🧑 Matched Person: **{person_name}**")
                    st.image(img, width=300)
                else:
                    st.error("❌ Match Rejected 🔒")
                    offer_add_to_database(query_img_path)
            else:
                st.warning("No matches found 🔒")
                offer_add_to_database(query_img_path)

    os.remove(query_img_path)
