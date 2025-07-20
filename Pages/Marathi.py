import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from deep_translator import GoogleTranslator
from functools import lru_cache
import time
import re
from typing import Optional, List
from utils import initialize_session_state, apply_language_styles, save_user_preferences, get_user_preferences

# Constants
MAX_LINKS = 5
RESPONSE_TIMEOUT = 30
MIN_RESPONSE_LENGTH = 50

class MarathiChatBot:
    """Marathi language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._initialize_environment()
    
    def _initialize_environment(self):
        load_dotenv()
        initialize_session_state()
        apply_language_styles('Marathi')
    
    @lru_cache(maxsize=32)
    def get_translator(self, src: str, dest: str):
        try:
            return GoogleTranslator(source=src, target=dest)
        except Exception as e:
            st.warning(f"⚠️ भाषांतर सेवेत त्रुटी: {str(e)}")
            return None

    @st.cache_resource
    def get_model_config(_self):
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 45,
            "max_output_tokens": 1500,
            "stop_sequences": ["---समाप्त---"]
        }

    @st.cache_resource
    def get_model(_self):
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google API की सापडली नाही. कृपया तुमच्या पर्यावरणीय सेटिंग तपासा.")

            genai.configure(api_key=genai_api_key)

            system_instruction = """तुम्ही एक ज्ञानाधारित, मराठी सहाय्यक आहात जो संपूर्ण आणि मदतनीस उत्तरं देतो.

सखोल मार्गदर्शक तत्त्वे:
1. उत्तर मराठीत स्पष्ट, नैसर्गिक भाषेत द्या
2. व्याकरणदृष्ट्या योग्य आणि सुसंस्कृत शब्द वापरा
3. किमान १०० शब्दांची माहितीपूर्ण उत्तरे द्या
4. उत्तर स्पष्ट परिच्छेदांमध्ये सादर करा
5. शक्य असल्यास विश्वसनीय माहिती किंवा उदाहरणे वापरा
6. पुनरावृत्ती टाळा
7. प्रश्नाच्या स्वरूपावर आधारित योग्य भाषाशैली ठेवा
8. योग्य संदर्भ आणि पार्श्वभूमी द्या
9. उत्तर नैसर्गिकपणे समाप्त करा"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ मॉडेल प्रारंभ करण्यात अयशस्वी: {str(e)}")
            return None

    @lru_cache(maxsize=32)
    def _extract_domain(self, url: str) -> Optional[str]:
        try:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            return domain_match.group(1) if domain_match else None
        except:
            return None

    def extract_links(self, text: str) -> Optional[str]:
        try:
            cleaned_text = re.sub(r'\n\n(संबंधित|विश्वसनीय) लिंक्स:.*$', '', text, flags=re.DOTALL)
            cleaned_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', cleaned_text)

            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, cleaned_text)

            if not urls:
                return None

            valid_urls = []
            seen_domains = set()

            for url in urls[:MAX_LINKS * 2]:
                clean_url = url.strip('()[].,!?').rstrip('.')
                if not re.match(r'https?://[^/]*\.[^/]+', clean_url):
                    continue

                domain = self._extract_domain(clean_url)
                if domain and domain not in seen_domains:
                    seen_domains.add(domain)
                    valid_urls.append(clean_url)
                    if len(valid_urls) >= MAX_LINKS:
                        break

            if not valid_urls:
                return None

            formatted_links = "\n\n---\n\n🔗 **संबंधित स्रोत:**\n\n"
            for url in valid_urls:
                domain = self._extract_domain(url)
                if domain:
                    display_name = domain.replace('www.', '')
                    formatted_links += f"• [{display_name}]({url})\n"

            return formatted_links.rstrip()
        except Exception as e:
            st.warning(f"⚠️ लिंक्स मिळवताना त्रुटी: {str(e)}")
            return None

    def _create_optimized_prompt(self, question: str) -> str:
        return f"""खालील प्रश्नाचे सविस्तर उत्तर मराठीत द्या:

{question}

अटी:
- किमान १०० शब्दांचे सविस्तर उत्तर द्या
- स्पष्ट परिच्छेद आणि संदर्भ वापरा
- पार्श्वभूमी आणि आवश्यक माहिती द्या
- अचूकता आणि उपयुक्ततेवर भर द्या

कृपया 3-5 विश्वसनीय वेबसाइटचे दुवे उत्तराच्या शेवटी जोडा."""

    def get_response(self, question: str) -> Optional[str]:
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None

            if "chat_session_marathi" not in st.session_state:
                st.session_state.chat_session_marathi = self._model.start_chat(history=[])

            prompt = self._create_optimized_prompt(question)

            with st.spinner("🤔 विचार करत आहे..."):
                response = st.session_state.chat_session_marathi.send_message(prompt)

                if not response or not response.text:
                    return "⚠️ उत्तर रिकामं आहे. कृपया प्रश्न पुन्हा विचारा."

                response_text = response.text.strip()
                if len(response_text) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ उत्तर खूप लहान आहे. पुन्हा प्रयत्न करत आहे...")
                    return None

                links = self.extract_links(response_text)
                final_response = response_text + (links if links else "")
                return final_response

        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API कोटा संपला आहे. कृपया नंतर पुन्हा प्रयत्न करा."
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ नेटवर्क त्रुटी. कृपया कनेक्शन तपासा."
            else:
                st.error(f"❌ उत्तर मिळवताना त्रुटी: {str(e)}")
                return None

    def display_chat_history(self):
        if "chat_history_marathi" not in st.session_state:
            st.session_state.chat_history_marathi = [
                AIMessage(content="🙏 **नमस्कार!** मी तुमचा मराठी सहाय्यक आहे. कृपया तुमचा प्रश्न विचारा!")
            ]

        for message in st.session_state.chat_history_marathi:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)

    def handle_user_input(self):
        user_query = st.chat_input(
            "💬 आपला प्रश्न येथे मराठीत टाइप करा...", 
            key="marathi_chat_input",
            max_chars=1000
        )

        if user_query and user_query.strip():
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ कृपया सविस्तर प्रश्न विचारा.")
                return

            start_time = time.time()
            st.session_state.chat_history_marathi.append(HumanMessage(content=cleaned_query))

            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)

            with st.chat_message("assistant", avatar="🤖"):
                result = self.get_response(cleaned_query)

                if result:
                    st.markdown(result)
                    st.session_state.chat_history_marathi.append(AIMessage(content=result))
                    response_time = time.time() - start_time
                    if response_time > 0:
                        st.sidebar.success(f"⚡ प्रतिसाद वेळ: {response_time:.2f}s")
                else:
                    error_msg = "😔 उत्तर देण्यात अडचण आली. कृपया नंतर पुन्हा प्रयत्न करा."
                    st.error(error_msg)
                    st.session_state.chat_history_marathi.append(AIMessage(content=error_msg))

    def run_chat_interface(self):
        try:
            self.display_chat_history()
            self.handle_user_input()
        except Exception as e:
            st.error(f"❌ चैट इंटरफेस त्रुटी: {str(e)}")
            st.info("🔄 कृपया पृष्ठ रीफ्रेश करा आणि पुन्हा प्रयत्न करा.")

def main():
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **प्रवेश नाकारला**: मराठी मोड्यूल वापरण्यासाठी कृपया लॉगिन करा.")
        st.info("👈 साइडबारमधून लॉगिन करा.")
        return

    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #006400 0%, #32CD32 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
        font-family: 'Noto Sans Devanagari', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>मराठी सहाय्यक चैटबॉट</h1>
    </div>
    """, unsafe_allow_html=True)

    try:
        chatbot = MarathiChatBot()
        chatbot.run_chat_interface()

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666;'>❤️ ने बनवलेला मराठी भाषेसाठी सहाय्यक</p>", 
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"❌ **अॅप्लिकेशन त्रुटी**: {str(e)}")
        st.info("🔄 कृपया पृष्ठ रीफ्रेश करा.")

if __name__ == "__main__":
    main()
