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

class HindiChatBot:
    """Hindi language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment variables and session state"""
        load_dotenv()
        initialize_session_state()
        apply_language_styles('Hindi')
    
    @lru_cache(maxsize=32)
    def get_translator(self, src: str, dest: str):
        """Get cached translator instance"""
        try:
            return GoogleTranslator(source=src, target=dest)
        except Exception as e:
            st.warning(f"⚠️ Translation service error: {str(e)}")
            return None
    
    @st.cache_resource
    def get_model_config(_self):
        """Get cached model configuration optimized for Hindi"""
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 45,  # Slightly higher for better Hindi vocabulary
            "max_output_tokens": 1500,
            "stop_sequences": ["---समाप्त---"]
        }
    
    @st.cache_resource
    def get_model(_self):
        """Get cached model instance with Hindi-specific configuration"""
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google Generative AI के लिए API key नहीं मिली। कृपया अपनी environment variables जांचें।")
            
            genai.configure(api_key=genai_api_key)
            
            system_instruction = """आप एक विशेषज्ञ हिंदी सहायक हैं जो व्यापक और सहायक उत्तर प्रदान करते हैं।

कठोर दिशानिर्देश:
1. हमेशा प्राकृतिक, शुद्ध हिंदी में उत्तर दें
2. उचित देवनागरी व्याकरण और पेशेवर शब्दावली का उपयोग करें
3. विस्तृत, जानकारीपूर्ण उत्तर दें (न्यूनतम 100 शब्द)
4. अपने उत्तर को स्पष्ट पैराग्राफ में संरचित करें
5. जब संभव हो, विश्वसनीय जानकारी और तथ्य प्रदान करें
6. दोहराव या भराव वाली सामग्री से बचें
7. प्रश्न की जटिलता के आधार पर उचित स्वर का उपयोग करें
8. तथ्यात्मक प्रश्नों के लिए, संदर्भ और पृष्ठभूमि प्रदान करें
9. उत्तर को स्वाभाविक रूप से समाप्त करें
10. सम्मानजनक संबोधन (आप) का उपयोग करें"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
            return None
    
    @lru_cache(maxsize=128)
    def clean_repeated_text(self, text: str) -> str:
        """Remove repeated words and phrases from Hindi text with caching"""
        try:
            words = text.split()
            cleaned_words = []
            prev_word = None
            
            for word in words:
                if word != prev_word:
                    cleaned_words.append(word)
                    prev_word = word
            
            return ' '.join(cleaned_words)
        except Exception:
            return text
    
    @lru_cache(maxsize=32)
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL with caching"""
        try:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            return domain_match.group(1) if domain_match else None
        except:
            return None
    
    def extract_links(self, text: str) -> Optional[str]:
        """Extract and format valid URLs from Hindi response text"""
        try:
            # Clean existing link sections
            cleaned_text = re.sub(r'\n\n(संबंधित|विश्वसनीय) लिंक्स:.*$', '', text, flags=re.DOTALL)
            cleaned_text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', cleaned_text)
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, cleaned_text)
            
            if not urls:
                return None
            
            # Process and deduplicate URLs
            valid_urls = []
            seen_domains = set()
            
            for url in urls[:MAX_LINKS * 2]:  # Process more to get better selection
                # Clean URL
                clean_url = url.strip('()[].,!?').rstrip('.')
                
                # Validate URL format
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
            
            # Format links in Hindi
            formatted_links = "\n\n---\n\n🔗 **संबंधित संसाधन:**\n\n"
            for url in valid_urls:
                domain = self._extract_domain(url)
                if domain:
                    display_name = domain.replace('www.', '')
                    formatted_links += f"• [{display_name}]({url})\n"
            
            return formatted_links.rstrip()
            
        except Exception as e:
            st.warning(f"⚠️ Link extraction error: {str(e)}")
            return None
    
    def _create_optimized_prompt(self, question: str) -> str:
        """Create an optimized Hindi prompt for better responses"""
        return f"""इस प्रश्न का व्यापक उत्तर हिंदी में दें:

{question}

आवश्यकताएं:
- विस्तृत, सुव्यवस्थित उत्तर प्रदान करें (न्यूनतम 100 शब्द)
- स्पष्ट पैराग्राफ और तार्किक प्रवाह का उपयोग करें
- प्रासंगिक संदर्भ और पृष्ठभूमि की जानकारी शामिल करें
- जहां उचित हो, विशिष्ट तथ्य और उदाहरण दें
- पेशेवर लेकिन बातचीत का स्वर बनाए रखें
- सटीकता और सहायकता पर ध्यान दें

कृपया अलग-अलग विश्वसनीय स्रोतों (शैक्षिक, सरकारी, समाचार, अनुसंधान संस्थान) से 3-5 संदर्भ लिंक प्रदान करें।
लिंक को अपने उत्तर के अंत में सरल URLs के रूप में प्रारूपित करें।"""
    
    def translate_text(self, text: str, src: str, dest: str) -> Optional[str]:
        """Translate text with error handling"""
        try:
            translator = self.get_translator(src, dest)
            if translator:
                result = translator.translate(text)
                return result if result else None
            return None
        except Exception as e:
            st.warning(f"⚠️ Translation error: {str(e)}")
            return None
    
    def get_direct_hindi_response(self, question: str) -> Optional[str]:
        """Get response directly in Hindi with optimized handling"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_hindi" not in st.session_state:
                st.session_state.chat_session_hindi = self._model.start_chat(history=[])
            
            # Create optimized prompt
            prompt = self._create_optimized_prompt(question)
            
            # Get response with spinner
            with st.spinner("🤔 सोच रहा हूं..."):
                response = st.session_state.chat_session_hindi.send_message(prompt)
                
                if not response or not response.text:
                    return "⚠️ मुझे एक खाली उत्तर मिला। कृपया अपना प्रश्न दोबारा पूछें।"
                
                # Clean repeated text
                cleaned_response = self.clean_repeated_text(response.text.strip())
                
                # Validate response quality
                if len(cleaned_response) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ उत्तर बहुत छोटा लगता है। पुनः प्रयास कर रहा हूं...")
                    return None
                
                # Extract and append links
                links = self.extract_links(cleaned_response)
                final_response = cleaned_response + (links if links else "")
                
                return final_response
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API quota समाप्त हो गया। कृपया बाद में पुनः प्रयास करें।"
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ नेटवर्क त्रुटि हुई। कृपया अपना कनेक्शन जांचें और पुनः प्रयास करें।"
            else:
                st.error(f"❌ उत्तर उत्पन्न करने में त्रुटि: {str(e)}")
                return None
    
    def get_fallback_response(self, question: str) -> Optional[str]:
        """Fallback method using translation when direct Hindi fails"""
        try:
            # Translate question to English
            translated_query = self.translate_text(question, 'hi', 'en')
            if not translated_query:
                return None
            
            # Get English response (simplified version)
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            english_prompt = f"Answer this question comprehensively: {translated_query}"
            
            if "chat_session_hindi_fallback" not in st.session_state:
                st.session_state.chat_session_hindi_fallback = self._model.start_chat(history=[])
            
            response = st.session_state.chat_session_hindi_fallback.send_message(english_prompt)
            
            if response and response.text:
                # Translate back to Hindi
                hindi_response = self.translate_text(response.text, 'en', 'hi')
                return hindi_response
            
            return None
            
        except Exception as e:
            st.warning(f"⚠️ Fallback method error: {str(e)}")
            return None
    
    def display_chat_history(self):
        """Display chat history with improved Hindi formatting"""
        if "chat_history_hindi" not in st.session_state:
            st.session_state.chat_history_hindi = [
                AIMessage(content="🙏 **नमस्ते!** मैं आपका हिंदी सहायक हूं। कृपया अपने प्रश्न हिंदी में पूछें!")
            ]
        
        for message in st.session_state.chat_history_hindi:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
    
    def handle_user_input(self):
        """Handle user input with validation and processing"""
        user_query = st.chat_input(
            "💬 अपना प्रश्न हिंदी में यहाँ टाइप करें...", 
            key="hindi_chat_input",
            max_chars=1000
        )
        
        if user_query and user_query.strip():
            # Validate input
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ कृपया अधिक विस्तृत प्रश्न पूछें।")
                return
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Add user message to history
            st.session_state.chat_history_hindi.append(HumanMessage(content=cleaned_query))
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                # Try direct Hindi response first
                result = self.get_direct_hindi_response(cleaned_query)
                
                # Fallback to translation if direct response fails
                if not result:
                    st.info("🔄 दूसरी विधि का प्रयास कर रहा हूं...")
                    result = self.get_fallback_response(cleaned_query)
                
                if result:
                    st.markdown(result)
                    st.session_state.chat_history_hindi.append(AIMessage(content=result))
                    
                    # Show performance metrics
                    response_time = time.time() - start_time
                    if response_time > 0:
                        st.sidebar.success(f"⚡ प्रतिक्रिया समय: {response_time:.2f}s")
                else:
                    error_msg = "😔 मुझे उत्तर उत्पन्न करने में समस्या हो रही है। कृपया अपना प्रश्न दोबारा पूछें या बाद में प्रयास करें।"
                    st.error(error_msg)
                    st.session_state.chat_history_hindi.append(AIMessage(content=error_msg))
    
    def run_chat_interface(self):
        """Main chat interface"""
        try:
            self.display_chat_history()
            self.handle_user_input()
            
        except Exception as e:
            st.error(f"❌ चैट इंटरफेस त्रुटि: {str(e)}")
            st.info("🔄 कृपया पेज को रिफ्रेश करें और पुनः प्रयास करें।")

def main():
    """Main function that respects the st.flag dependency"""
    
    # Check authentication flag
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **पहुंच से इनकार**: कृपया हिंदी मॉड्यूल का उपयोग करने के लिए लॉग इन करें।")
        st.info("👈 अपने खाते में लॉग इन करने के लिए साइडबार का उपयोग करें।")
        return
    
    # Page configuration with Hindi styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
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
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
    .hindi-text {
        font-family: 'Noto Sans Devanagari', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> हिंदी सहायक चैटबॉट </h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run chatbot
    try:
        chatbot = HindiChatBot()
        
        
        # Main chat interface
        chatbot.run_chat_interface()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666; font-family: \"Noto Sans Devanagari\", sans-serif;'>हिंदी भाषा समर्थन के लिए ❤️ से बनाया गया</p>", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"❌ **एप्लिकेशन त्रुटि**: {str(e)}")
        st.info("🔄 कृपया पेज को रिफ्रेश करें। यदि समस्या बनी रहे, तो सहायता से संपर्क करें।")

if __name__ == "__main__":
    main()