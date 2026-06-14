import streamlit as st
import requests


# --- UI CONFIG ---
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🚫",
    layout="centered"
)

st.title("🚫 Smart Spam Detection System")
st.write("Enter text below to evaluate risk patterns using the trained ML backend.")


# --- IMPORTANT: FIXED API URL ---
API_URL = "http://127.0.0.1:8000/predict"


# --- INPUT UI ---
user_input = st.text_area("Enter your message content here:", height=150)


# --- ACTION ---
if st.button("Analyze Text", type="primary"):

    if not user_input.strip():
        st.warning("Please type a message before executing analysis.")

    else:
        with st.spinner("Analyzing message text patterns..."):

            try:
                response = requests.post(
                    API_URL,
                    json={"text": user_input},
                    timeout=10
                )

                if response.status_code == 200:
                    res = response.json()
                    conf = res["confidence"] * 100

                    st.subheader("Analysis Results:")

                    if res["is_spam"]:
                        st.error(f"🚨 Spam Detected! ({conf:.2f}% confidence)")
                    else:
                        st.success(f"✅ Legitimate Message ({conf:.2f}% confidence)")

                else:
                    st.error(
                        f"Backend Error: {response.status_code} - {response.text}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot connect to FastAPI backend. "
                    "Make sure server is running:\n"
                    "`uvicorn main:app --reload`"
                )

            except requests.exceptions.Timeout:
                st.error("⏳ Request timed out. Backend is too slow or not responding.")