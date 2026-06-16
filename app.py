import streamlit as st
import numpy as np, pickle, gzip, cv2, os
from PIL import Image
import torch, timm
from torchvision import transforms
from skimage.feature import hog
from sklearn.preprocessing import normalize

st.set_page_config(page_title="Hybrid AI — Disease Detection",
                   page_icon="🌿", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#0d3320 0%,#14532d 100%); }
h1,h2,h3 { color:#eafaf1 !important; }
.card { background:#ffffff; border-radius:18px; padding:26px;
        box-shadow:0 8px 24px rgba(0,0,0,0.18); }
.result-pos { background:#eafaf1; border-radius:12px; padding:16px; margin:8px 0; }
.result-warn{ background:#fef5e7; border-radius:12px; padding:16px; margin:8px 0; }
.big { font-size:30px; font-weight:800; color:#14532d; }
.muted{ color:#5c5c5c; font-size:13px; }
</style>
""", unsafe_allow_html=True)

def load_gz(path):
    if not os.path.exists(path):
        parts = sorted([f for f in os.listdir('.') if f.startswith(os.path.basename(path)+'.part')])
        if parts:
            with open(path,'wb') as out:
                for p in parts:
                    with open(p,'rb') as f:
                        out.write(f.read())
    with gzip.open(path,'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_all():
    swin = timm.create_model('swin_base_patch4_window7_224',
                             pretrained=True, num_classes=0).eval()
    return (swin,
            load_gz('final_model.pkl.gz'),
            load_gz('cat_plant.pkl.gz'),
            load_gz('med_lgb_fused.pkl.gz'),
            load_gz('cat_med_fused.pkl.gz'),
            load_gz('xgb_med.pkl.gz'),
            pickle.load(open('plant_classes.pkl','rb')))

swin, plant_lgb, plant_cat, med_lgb, med_cat, xgb_med, plant_classes = load_all()
med_classes = ['COVID-19','Normal','Pneumonia']

tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

treatments = {
 "Tomato___Late_blight":"Destroy infected plants; apply copper-based protectant + systemic fungicide. Avoid leaf wetness.",
 "Tomato___Early_blight":"Apply mancozeb/chlorothalonil. Remove lower infected leaves, improve airflow.",
 "Tomato___healthy":"Plant is healthy. Maintain regular watering and monitoring.",
 "Potato___Late_blight":"Remove and destroy infected plants immediately. Apply protectant fungicide.",
 "Potato___Early_blight":"Apply chlorothalonil. Rotate crops, avoid overhead irrigation.",
 "Potato___healthy":"Plant is healthy. Maintain good soil drainage.",
}
med_advice = {
 "COVID-19":"Isolate patient. Confirm with RT-PCR. Seek medical care immediately.",
 "Normal":"No infection detected. Lungs appear clear.",
 "Pneumonia":"Consult doctor. Likely bacterial/viral pneumonia — may need antibiotics.",
}

def swin_emb(img):
    with torch.no_grad():
        return swin(tf(img).unsqueeze(0)).numpy()

def hog_feats(img):
    arr = np.array(img.resize((224,224)).convert('RGB'))
    pf=[]
    for ps in [64,128]:
        g = cv2.cvtColor(cv2.resize(arr,(ps,ps)), cv2.COLOR_RGB2GRAY)
        pf.extend(hog(g, pixels_per_cell=(16,16),
                      cells_per_block=(2,2), feature_vector=True))
    return np.array(pf, dtype=np.float32).reshape(1,-1)

st.title("🌿 Hybrid Learning Framework")
st.markdown("<p class='muted' style='color:#bfe3cd'>Swin Transformer + XGBoost + Attention Fusion + LightGBM/CatBoost</p>", unsafe_allow_html=True)

mode = st.sidebar.radio("Select Domain", ["🌿 Plant Disease","🏥 Medical X-Ray"])
st.sidebar.success("Plant: 98.80%")
st.sidebar.info("Medical: 95.03%")

c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload Image")
    up = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Analysis Result")
    if up:
        with st.spinner("Analyzing..."):
            emb = swin_emb(img)
            if "Plant" in mode:
                ens = plant_lgb.predict_proba(emb)*0.5 + plant_cat.predict_proba(emb)*0.5
                idx = int(np.argmax(ens,1)[0]); conf=float(ens[0].max()*100)
                name = plant_classes[idx]
                clean = name.replace("___"," - ").replace("_"," ")
                adv = treatments.get(name,"Consult an agriculture expert.")
                healthy = "healthy" in name.lower()
                labels = plant_classes
            else:
                xp = xgb_med.predict_proba(hog_feats(img))
                fused = np.concatenate([normalize(emb)*0.4, normalize(xp)*0.6], axis=1)
                ens = med_lgb.predict_proba(fused)*0.5 + med_cat.predict_proba(fused)*0.5
                idx = int(np.argmax(ens,1)[0]); conf=float(ens[0].max()*100)
                clean = med_classes[idx]
                adv = med_advice.get(clean,"Consult a doctor.")
                healthy = (clean=="Normal")
                labels = med_classes

            cls = "result-pos" if healthy else "result-warn"
            st.markdown(f"<div class='{cls}'><span class='muted'>Disease Identified</span><br><span class='big'>{clean}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-pos'><span class='muted'>Confidence Level</span><br><span class='big'>{conf:.2f}%</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-warn'><span class='muted'>Expert Advice</span><br><b style='color:#1a1a1a'>{adv}</b></div>", unsafe_allow_html=True)

            st.markdown("<br><b style='color:#14532d;font-size:16px'>Top 3 Predictions</b>", unsafe_allow_html=True)
            for rank, i in enumerate(ens[0].argsort()[-3:][::-1]):
                n = labels[i].replace("___"," - ").replace("_"," ")
                pct = float(ens[0][i]*100)
                bar_col = "#14532d" if rank == 0 else "#5a9367"
                st.markdown(f"""
                <div style='margin:10px 0;'>
                  <div style='display:flex;justify-content:space-between;color:#1a1a1a;font-size:14px;font-weight:600;margin-bottom:4px;'>
                    <span>{n}</span><span>{pct:.1f}%</span>
                  </div>
                  <div style='background:#e0e0e0;border-radius:8px;height:14px;width:100%;'>
                    <div style='background:{bar_col};width:{pct:.1f}%;height:14px;border-radius:8px;'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload an image to see results.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p class='muted' style='text-align:center;color:#bfe3cd'>Jorhat Institute of Science and Technology — ECE Dept</p>", unsafe_allow_html=True)
