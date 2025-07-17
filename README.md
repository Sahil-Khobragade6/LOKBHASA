# 🌍 Lokbhasha - Multilingual WhatsApp Chatbot

**Lokbhasha** is a multilingual chatbot built using Python and Streamlit. It allows users to interact in regional languages like **Hindi**, **Telugu**, and **Tamil** . It uses the **Google Translate API** for translation and the **Google Gemini API** for generating intelligent responses.

---

## ✅ Features

- Multilingual support (Hindi, Telugu, Tamil, English)
- AI-generated answers using Google Gemini
- Auto translation with Google Translate API
- Simple UI via Streamlit
- WhatsApp chatbot ready (Twilio integration optional)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **APIs**: Google Translate API, Google Gemini API  


---

---

## 🔧 Installation & Execution

### Prerequisites

- Python 3.8+
- Google Translate API key
- Google Gemini API key

### Steps

1. **Clone the repo**
   ```bash
   cd Lokbhasha

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
GOOGLE_API_KEY=your_google_api_key

streamlit run app.py
