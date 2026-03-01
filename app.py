import streamlit as st
import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import heapq
from models.generator import Generator
from utils.dataset import TextSummaryDataset

st.set_page_config(page_title="Automated Text Summarization", page_icon="✨", layout="wide")

# Custom Premium CSS for styling the Web App
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700;900&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Fixed Solid Background */
    .stApp {
        background-color: #1a2a6c;
        color: #ffffff;
    }
    
    /* Main H1 Title with glowing colorful text */
    h1 {
        background: -webkit-linear-gradient(45deg, #FFD700, #FF8C00, #FF0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0px 4px 15px rgba(255, 0, 128, 0.4);
        font-size: 44px !important;
    }
    
    /* Container Glassmorphism - Super sleek blur */
    div.block-container {
        background: rgba(15, 15, 30, 0.65);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-radius: 30px;
        padding: 4rem 3rem;
        box-shadow: 0 15px 40px 0 rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-top: 50px;
        margin-bottom: 50px;
    }
    
    /* Input Text Area Styling */
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        color: #ffffff !important;
        border-radius: 15px;
        font-size: 16px;
        padding: 15px;
        transition: all 0.4s ease;
    }
    .stTextArea textarea:focus {
        border-color: #00f2fe !important;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.6) !important;
        background-color: rgba(255, 255, 255, 0.15) !important;
    }
    
    /* Custom Neon Submit Button */
    .stButton>button {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 700;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
        box-shadow: 0 8px 15px rgba(255, 65, 108, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 15px 25px rgba(255, 65, 108, 0.6);
        color: white;
    }
    
    /* Interactive Radio Buttons layout */
    div.row-widget.stRadio > div{
        background: linear-gradient(90deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.1) 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #00f2fe;
        transition: all 0.3s ease;
    }
    div.row-widget.stRadio > div:hover {
        border-left: 5px solid #FFD700;
        transform: translateX(5px);
    }
    
    /* Success & Info Output Boxes */
    .stAlert {
        border-radius: 20px !important;
        background-color: rgba(0, 255, 135, 0.1) !important;
        border: 1px solid rgba(0, 255, 135, 0.3) !important;
        color: #e0e0e0 !important;
        padding: 20px;
        font-size: 18px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Secondary text / description styling */
    .stMarkdown p {
        color: #e0e0f5;
        font-size: 17px;
        line-height: 1.6;
    }
    
    /* Custom divider line */
    hr {
        border: 0;
        height: 2px;
        background-image: linear-gradient(to right, rgba(0,0,0,0), rgba(255,255,255,0.5), rgba(0,0,0,0));
        margin: 30px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Basic NLTK installs cleanly behind the scenes
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
setup_nltk()

def load_gan_model():
    dataset = TextSummaryDataset('data/train.csv')
    vocab_size = len(dataset.vocab)
    G = Generator(vocab_size)
    G.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
    G.eval()
    return G, dataset

def generate_accurate_summary(text, num_sentences=2):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    
    # If the user only gave 1 sentence, summarize it by extracting the most important conceptual words
    if len(sentences) <= 1:
        words = word_tokenize(text)
        summary_words = [w for w in words if w.lower() not in stop_words and w.isalnum()]
        
        # If it stripped everything out, return text, otherwise return the distillation
        if len(summary_words) < 2:
            return text
        return ' '.join(summary_words)
        
    if len(sentences) == 2:
        num_sentences = 1
    
    stop_words = set(stopwords.words("english"))
    word_frequencies = {}
    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words and word.isalnum():
            if word not in word_frequencies:
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    if not word_frequencies:
        return text
        
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_frequency)
        
    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
                        
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)


st.title("✨ Automated Text Summarization")
st.markdown("This application provides multiple summarization models. The `Custom GAN` is trained on our local dataset (~25 sentences), whereas the `Accurate Extractive Summary` uses a statistical NLP approach to accurately summarize real-world text! Choose the accurate model for the best results on arbitrary text.")

mode = st.radio("Select Summarization Model:", ["⚡ Accurate Extractive Summary (Recommended for real text)", "🎓 Educational Custom GAN (Trained on 25 AI sentences)"])

# Create a 2-column layout
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Enter text to summarize:", height=200, placeholder="A generative adversarial network (GAN) is a class of machine learning frameworks...")
    submit_btn = st.button("Summarize", type="primary")

with col2:
    if submit_btn:
        if user_input.strip() == "":
            st.warning("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summary..."):
                if "Accurate" in mode:
                    # Use Extractive text rank for actual summarization
                    try:
                        summary = generate_accurate_summary(user_input, 2)
                        st.markdown("### Generated Summary")
                        st.success(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        
                else:
                    # Use Custom Educational GAN
                    try:
                        G, dataset = load_gan_model()
                        inv_vocab = {v: k for k, v in dataset.vocab.items()}
                        
                        input_words_clean = user_input.split()
                        input_tokens = [dataset.vocab.get(w, 1) for w in input_words_clean]
                        if len(input_tokens) == 0:
                            st.warning("Could not tokenize input.")
                            st.stop()
                            
                        input_tensor = torch.tensor(input_tokens).unsqueeze(0)
                        
                        with torch.no_grad():
                            out = G(input_tensor)
                        pred_tokens = out.argmax(dim=-1).squeeze(0).tolist()
                        
                        pred_words = []
                        last_word = ""
                        for i, t in enumerate(pred_tokens):
                            word = inv_vocab.get(t, '<UNK>')
                            
                            # Soft Copy fallback only if it's completely unknown and within bounds
                            if word == '<UNK>':
                                # Try to find a matching word in input that hasn't been used, or just skip
                                if i < len(input_words_clean):
                                    word = input_words_clean[i]
                                else:
                                    continue
                                    
                            if word not in ['<PAD>', '<UNK>', '.', ',']:
                                if word.lower() != last_word.lower():
                                    pred_words.append(word)
                                    last_word = word
                        
                        if len(pred_words) > 0:
                            pred_words[0] = pred_words[0].capitalize()
                            
                        # Fix formatting issues like trailing spaces
                        summary = " ".join(pred_words).strip()
                        
                        # If it heavily failed, let user know
                        if len(summary.split()) < 3 and len(input_words_clean) > 5:
                            summary += " (Warning: Input text significantly deviated from training data concepts.)"
                        
                        st.markdown("### Generated Summary")
                        st.info(summary)
                        
                        # Warning if it contains a lot of UNK mappings
                        unk_count = input_tokens.count(1)
                        if unk_count > len(input_tokens) * 0.5:
                            st.warning("⚠️ Warning: Your input contains words the GAN was never trained on. Switch to 'Accurate Extractive Summary' above to get a proper summary!")
                    except Exception as e:
                        st.error(f"GAN Model Error: {e}")
    else:
        st.markdown("<div style='text-align: center; margin-top: 50px; color: #888;'>Your generated summary will appear here.</div>", unsafe_allow_html=True)
