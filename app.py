# Resume Analyzer - Streamlit App
# deep learning project using pytorch

import sqlite3, re, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import streamlit as st
import torch, torch.nn as nn

try:
    from wordcloud import WordCloud
    HAS_WC = True
except: HAS_WC = False

try:
    import PyPDF2
    HAS_PDF = True
except: HAS_PDF = False

# paths
DATA_PATH = Path("data/UpdatedResumeDataSet.csv")
DATA_CANDIDATES = [
    DATA_PATH,
    Path("UpdatedResumeDataSet.csv"),
    Path("data/Resume.csv"),
    Path("Resume.csv"),
    Path("archive (7)/Resume/Resume.csv")
]
MODEL_PATH = Path("models/resume_classifier.pth")
ENCODER_PATH = Path("models/label_encoder.pkl")
TFIDF_PATH = Path("models/tfidf_vectorizer.pkl")
METRICS_PATH = Path("models/metrics.json")
DB_PATH = Path("resume_analyzer.db")

# skills we look for in resumes
SKILLS = {
    "programming": ["python","java","javascript","c++","c#","php","swift","kotlin","go",
                     "typescript","r","matlab","sql","html","css","bash"],
    "data_science": ["machine learning","deep learning","tensorflow","pytorch","keras",
                     "pandas","numpy","data analysis","statistics","nlp","computer vision"],
    "web": ["react","angular","vue","node.js","django","flask","spring boot","rest api",
            "bootstrap","mongodb","firebase"],
    "cloud": ["aws","azure","gcp","docker","kubernetes","jenkins","ci/cd","linux","git"],
    "database": ["mysql","postgresql","mongodb","redis","oracle","sqlite","nosql"],
    "soft_skills": ["leadership","communication","teamwork","problem solving",
                    "project management","agile","scrum"],
}

# what each role needs
ROLE_SKILLS = {
    "Data Science": ["python","machine learning","deep learning","pandas","sql","statistics"],
    "Java Developer": ["java","spring boot","sql","rest api","git"],
    "Python Developer": ["python","django","flask","sql","git","docker"],
    "Web Designing": ["html","css","javascript","react","bootstrap"],
    "DevOps Engineer": ["docker","kubernetes","aws","linux","ci/cd","git"],
    "Business Analyst": ["excel","sql","power bi","communication","agile"],
    "Database": ["sql","mysql","postgresql","mongodb","oracle"],
    "Testing": ["selenium","manual testing","jira","agile","sql"],
    "HR": ["communication","leadership","teamwork","project management"],
    "Sales": ["communication","leadership","teamwork"],
}

# neural network (same as training)
class ResumeNet(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, out)
        )
    def forward(self, x): return self.net(x)

# page config
st.set_page_config(page_title="Resume Analyzer", page_icon="📄", layout="wide")

# css
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); font-family: 'Inter'; }
.header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 30px; border-radius: 15px;
    margin-bottom: 20px; text-align: center; }
.header h1 { color: white; font-size: 2.3em; margin: 0; }
.header p { color: rgba(255,255,255,0.85); margin-top: 5px; }
.card { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 18px; margin: 6px 0; }
.mbox { background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.25);
    border-radius: 12px; padding: 20px; text-align: center; }
.mbox h5 { color: rgba(255,255,255,0.5); font-size: 0.75em; margin: 0; text-transform: uppercase; }
.mbox h2 { color: #a8b4ff; font-size: 1.7em; margin: 8px 0 4px; }
.mbox p { color: rgba(255,255,255,0.4); font-size: 0.8em; margin: 0; }
.tag { display: inline-block; background: rgba(102,126,234,0.15); border: 1px solid rgba(102,126,234,0.3);
    color: #a8b4ff; padding: 4px 10px; border-radius: 15px; margin: 3px; font-size: 0.82em; }
.role { display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white;
    padding: 8px 22px; border-radius: 20px; font-size: 1.2em; font-weight: 700; }
.tip { background: rgba(255,193,7,0.06); border-left: 3px solid #ffc107; padding: 10px 14px;
    border-radius: 0 8px 8px 0; margin: 5px 0; color: rgba(255,255,255,0.8); font-size: 0.9em; }
.ok { background: rgba(76,175,80,0.06); border-left: 3px solid #4caf50; padding: 10px 14px;
    border-radius: 0 8px 8px 0; margin: 5px 0; color: rgba(255,255,255,0.8); font-size: 0.9em; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important; }
.stButton>button { background: linear-gradient(135deg, #667eea, #764ba2) !important; color: white !important;
    border: none !important; border-radius: 10px !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, confidence REAL, skills TEXT, feedback TEXT, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.commit(); conn.close()

def save(role, conf, skills, feedback):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (role,confidence,skills,feedback) VALUES (?,?,?,?)",
                 (role, conf, json.dumps(skills), feedback))
    conn.commit(); conn.close()

def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id,role,confidence,skills,feedback,date FROM history ORDER BY id DESC LIMIT 50").fetchall()
    conn.close(); return rows

# helpers
def clean(text):
    text = re.sub(r'http\S+|www\S+|\S+@\S+', ' ', str(text))
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

def find_skills(text):
    t = text.lower()
    found = {}
    for cat, skills in SKILLS.items():
        m = [s for s in skills if re.search(r'\b'+re.escape(s)+r'\b', t)]
        if m: found[cat] = m
    return found

def flat(d): return [s for v in d.values() for s in v]

def missing(role, found):
    have = [s.lower() for s in flat(found)]
    return [s for s in ROLE_SKILLS.get(role, []) if s not in have]

def tips(role, found, miss, conf):
    t = []
    n = len(flat(found))
    if conf >= 0.7: t.append(f"Strong match for {role}!")
    elif conf >= 0.5: t.append(f"Moderate match for {role}. Can improve.")
    else: t.append("Weak match. Add more relevant skills.")
    if n >= 8: t.append(f"Good skills ({n} found).")
    else: t.append(f"Only {n} skills found. Add more.")
    if miss: t.append(f"Learn: {', '.join(miss[:4])}")
    return t

def read_pdf(f):
    if not HAS_PDF: return None
    try:
        r = PyPDF2.PdfReader(f)
        return "".join([p.extract_text() or "" for p in r.pages])
    except: return None

def score(found, miss, conf):
    s = min(len(flat(found))*5, 40) - min(len(miss)*3, 20) + conf*40
    return round(max(0, min(100, s)), 1)

# load model
@st.cache_resource
def load_model():
    try:
        le = joblib.load(ENCODER_PATH)
        tfidf = joblib.load(TFIDF_PATH)
        ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        m = ResumeNet(ckpt['inp'], ckpt['out'])
        m.load_state_dict(ckpt['state']); m.eval()
        return m, le, tfidf, True
    except: return None, None, None, False

@st.cache_data
def load_data():
    for path in DATA_CANDIDATES:
        try:
            if path.exists():
                df = pd.read_csv(path)
                # Auto-detect resume text column and normalize to 'Resume'
                for col_name in ['Resume', 'Resume_str', 'resume', 'resume_str', 'Text', 'text']:
                    if col_name in df.columns:
                        if col_name != 'Resume':
                            df['Resume'] = df[col_name]
                        break
                return df
        except Exception:
            continue
    return None

@st.cache_data
def load_metrics():
    try: return json.load(open(METRICS_PATH))
    except: return None

# main app
init_db()

st.markdown("""<div class="header"><h1>📄 Resume Analyzer AI</h1>
<p>Upload your resume • Get AI predictions • Know your best role</p></div>""", unsafe_allow_html=True)

model, le, tfidf, loaded = load_model()
if not loaded:
    st.error("Model not found! Run `python train_model.py` first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📄 ANALYZE", "📊 VISUALS", "📜 HISTORY", "ℹ️ ABOUT"])

# ---- ANALYZE ----
with tab1:
    st.subheader("Upload Your Resume")
    method = st.radio("Input method:", ["Upload File", "Paste Text"], horizontal=True)
    resume = ""

    if method == "Upload File":
        f = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"])
        if f:
            resume = read_pdf(f) if f.name.endswith(".pdf") else f.read().decode("utf-8", errors="ignore")
            if not resume: st.warning("Couldn't read file. Try pasting text.")
    else:
        resume = st.text_area("Paste resume:", height=180, placeholder="Paste your resume here...")

    if st.button("🔍 Analyze Resume", type="primary", use_container_width=True):
        if len(resume.strip()) < 50:
            st.warning("Please provide a longer resume text.")
        else:
            with st.spinner("Analyzing..."):
                feat = torch.FloatTensor(tfidf.transform([clean(resume)]).toarray())
                with torch.no_grad():
                    probs = torch.softmax(model(feat), 1)
                    conf, idx = probs.max(1)
                    conf, idx = conf.item(), idx.item()
                role = le.inverse_transform([idx])[0]
                found = find_skills(resume)
                miss = missing(role, found)
                fb = tips(role, found, miss, conf)
                sc = score(found, miss, conf)
                save(role, conf, flat(found), " | ".join(fb))

            st.markdown("---")
            st.markdown(f'<div style="text-align:center;margin:15px 0"><span class="role">{role}</span></div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f'<div class="mbox"><h5>Confidence</h5><h2>{conf:.0%}</h2></div>', unsafe_allow_html=True)
            with c2: st.markdown(f'<div class="mbox"><h5>Score</h5><h2>{sc}/100</h2></div>', unsafe_allow_html=True)
            with c3: st.markdown(f'<div class="mbox"><h5>Skills</h5><h2>{len(flat(found))}</h2></div>', unsafe_allow_html=True)

            # top roles
            st.subheader("Top Matches")
            tk = min(5, len(le.classes_))
            tp, ti = torch.topk(probs[0], tk)
            st.dataframe(pd.DataFrame({"Role": [le.inverse_transform([i.item()])[0] for i in ti],
                                        "Match": [f"{p.item()*100:.1f}%" for p in tp]}),
                         use_container_width=True, hide_index=True)

            # skills
            if found:
                st.subheader("Skills Found")
                for cat, sk in found.items():
                    tags = "".join([f'<span class="tag">{s}</span>' for s in sk])
                    st.markdown(f'<div class="card"><strong style="color:rgba(255,255,255,0.5)">{cat.replace("_"," ").title()}</strong><div style="margin-top:6px">{tags}</div></div>', unsafe_allow_html=True)

            if miss:
                st.subheader(f"Missing Skills for {role}")
                for s in miss:
                    st.markdown(f'<div class="tip">🔸 Learn <b>{s}</b></div>', unsafe_allow_html=True)

            st.subheader("Tips")
            for t in fb:
                cls = "ok" if "strong" in t.lower() or "good" in t.lower() else "tip"
                st.markdown(f'<div class="{cls}">{t}</div>', unsafe_allow_html=True)

            # quick chart
            if found:
                fig, ax = plt.subplots(figsize=(5,3))
                fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#1a1a2e')
                cats = [c.replace("_"," ").title() for c in found]
                ax.barh(cats, [len(v) for v in found.values()], color='#667eea')
                ax.set_title("Your Skills", color='white', fontsize=11)
                ax.tick_params(colors='white')
                for s in ['top','right']: ax.spines[s].set_visible(False)
                for s in ['bottom','left']: ax.spines[s].set_color((1, 1, 1, 0.2))
                plt.tight_layout(); st.pyplot(fig); plt.close()

# ---- VISUALS ----
with tab2:
    st.subheader("Dataset & Model Visuals")
    df = load_data()
    metrics = load_metrics()

    if df is not None:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,5))
            fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#1a1a2e')
            vc = df['Category'].value_counts()
            ax.barh(vc.index, vc.values, color=plt.cm.plasma(np.linspace(0.2,0.9,len(vc))))
            ax.set_title("Resumes per Category", color='white', fontsize=12)
            ax.tick_params(colors='white', labelsize=7)
            for s in ['top','right']: ax.spines[s].set_visible(False)
            for s in ['bottom','left']: ax.spines[s].set_color((1, 1, 1, 0.2))
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            if HAS_WC:
                fig, ax = plt.subplots(figsize=(6,5))
                fig.patch.set_facecolor('#0f0c29')
                txt = " ".join(df['Resume'].apply(lambda x: re.sub(r'[^a-zA-Z\s]','',str(x).lower())))
                wc = WordCloud(width=700, height=400, background_color='#0f0c29', colormap='plasma', max_words=80).generate(txt)
                ax.imshow(wc); ax.axis('off')
                ax.set_title("Common Words", color='white', fontsize=12)
                st.pyplot(fig); plt.close()
            else:
                fig, ax = plt.subplots(figsize=(6,5))
                fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#0f0c29')
                top5 = df['Category'].value_counts().head(6)
                ax.pie(top5.values, labels=top5.index, autopct='%1.1f%%',
                       colors=['#667eea','#764ba2','#f093fb','#4facfe','#00f2fe','#43e97b'],
                       textprops={'color':'white','fontsize':8})
                ax.set_title("Top Categories", color='white', fontsize=12)
                st.pyplot(fig); plt.close()

    if metrics:
        st.subheader("Training Results")
        m1, m2 = st.columns(2)
        with m1: st.markdown(f'<div class="mbox"><h5>Accuracy</h5><h2>{metrics["accuracy"]*100:.1f}%</h2></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="mbox"><h5>Classes</h5><h2>{metrics["num_classes"]}</h2></div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if 'train_losses' in metrics:
                fig, ax = plt.subplots(figsize=(6,3.5))
                fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#1a1a2e')
                ax.plot(metrics['train_losses'], color='#f093fb', linewidth=2)
                ax.set_title("Loss Curve", color='white', fontsize=11)
                ax.set_xlabel("Epoch", color='white'); ax.set_ylabel("Loss", color='white')
                ax.tick_params(colors='white'); ax.grid(alpha=0.1, color='white')
                for s in ['top','right']: ax.spines[s].set_visible(False)
                for s in ['bottom','left']: ax.spines[s].set_color((1, 1, 1, 0.2))
                plt.tight_layout(); st.pyplot(fig); plt.close()
        with c2:
            if 'train_accuracies' in metrics:
                fig, ax = plt.subplots(figsize=(6,3.5))
                fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#1a1a2e')
                ax.plot(metrics['train_accuracies'], color='#667eea', linewidth=2, label='Train')
                if metrics.get('test_accuracies'):
                    te = [0] + list(range(4, len(metrics['train_accuracies']), 5))
                    te = te[:len(metrics['test_accuracies'])]
                    ax.plot(te, metrics['test_accuracies'], color='#43e97b', marker='o', markersize=4, label='Test')
                ax.set_title("Accuracy", color='white', fontsize=11)
                ax.set_xlabel("Epoch", color='white')
                ax.tick_params(colors='white')
                ax.legend(facecolor='#1a1a2e', labelcolor='white')
                ax.grid(alpha=0.1, color='white')
                for s in ['top','right']: ax.spines[s].set_visible(False)
                for s in ['bottom','left']: ax.spines[s].set_color((1, 1, 1, 0.2))
                plt.tight_layout(); st.pyplot(fig); plt.close()

        if 'confusion_matrix' in metrics:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10,8))
            fig.patch.set_facecolor('#0f0c29'); ax.set_facecolor('#1a1a2e')
            sns.heatmap(np.array(metrics['confusion_matrix']), annot=True, fmt='d', cmap='plasma',
                        xticklabels=metrics.get('categories',[]), yticklabels=metrics.get('categories',[]), ax=ax)
            ax.set_title("Confusion Matrix", color='white', fontsize=13)
            ax.tick_params(colors='white', labelsize=6)
            plt.xticks(rotation=45, ha='right'); plt.tight_layout()
            st.pyplot(fig); plt.close()

# ---- HISTORY ----
with tab3:
    st.subheader("Analysis History")
    rows = get_history()
    if rows:
        hdf = pd.DataFrame(rows, columns=["ID","Role","Confidence","Skills","Feedback","Date"])
        hdf["Confidence"] = (hdf["Confidence"]*100).round(1).astype(str) + "%"
        st.dataframe(hdf, use_container_width=True, hide_index=True)
    else:
        st.info("No history yet. Analyze a resume first!")

# ---- ABOUT ----
with tab4:
    st.subheader("About")
    st.markdown("""
    <div class="card">
        <h3 style="color:white;margin-top:0">Resume Analyzer AI</h3>
        <p style="color:rgba(255,255,255,0.7)">
        Deep learning project that uses PyTorch to classify resumes into job roles.
        Extracts skills, finds gaps, and gives improvement tips.</p>
        <br>
        <b style="color:#a8b4ff">Tech Stack:</b>
        <span style="color:rgba(255,255,255,0.6)">PyTorch, Streamlit, scikit-learn, SQLite, Matplotlib</span>
        <br><br>
        <b style="color:#a8b4ff">Model:</b>
        <span style="color:rgba(255,255,255,0.6)">3-layer NN (5000→512→256→128→25) with BatchNorm & Dropout</span>
        <br><br>
        <b style="color:#a8b4ff">Dataset:</b>
        <span style="color:rgba(255,255,255,0.6)">Kaggle Resume Dataset (962 resumes, 25 categories)</span>
    </div>
    <div class="card" style="text-align:center;margin-top:10px">
        <p style="color:rgba(255,255,255,0.4);font-size:0.85em">
        DEEP LEARNING PROJECT • BCA • 45 DAYS INTERNSHIP • EXCELLENCE TECHNOLOGY</p>
    </div>
    """, unsafe_allow_html=True)
