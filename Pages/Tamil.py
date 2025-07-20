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
MIN_RESPONSE_LENGTH = 80  # Adjusted for Tamil text

class TamilChatBot:
    """Tamil language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._translator_cache = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment variables and session state"""
        load_dotenv()
        initialize_session_state()
        apply_language_styles('Tamil')
    
    @staticmethod
    def clean_repeated_text(text: str) -> str:
        """Remove repeated words and phrases from Tamil text"""
        if not text:
            return text
            
        try:
            # Split text into words
            words = text.split()
            # Remove consecutive repeated words
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i-1]:
                    cleaned_words.append(word)
            return ' '.join(cleaned_words)
        except Exception:
            return text
    
    @lru_cache(maxsize=32)
    def get_translator(self, src: str, dest: str) -> GoogleTranslator:
        """Get cached translator instance"""
        return GoogleTranslator(source=src, target=dest)
    
    @st.cache_resource
    def get_model_config(_self):
        """Get cached model configuration"""
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1500,  # Increased for better responses
            "stop_sequences": ["---முடிவு---"]  # Add stop sequence for better control
        }
    
    @st.cache_resource
    def get_model(_self):
        """Get cached model instance with error handling"""
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google API key கிடைக்கவில்லை. தயவுசெய்து environment variables ஐ சரிபார்க்கவும்.")
            
            genai.configure(api_key=genai_api_key)
            
            system_instruction = """நீங்கள் ஒரு உதவிகரமான மற்றும் நன்கு தமிழ் பேசும் AI உதவியாளர்.

கடுமையான வழிகாட்டுதல்கள்:
1. எப்போதும் இயல்பான, சரளமான தமிழில் மட்டுமே பதிலளிக்கவும்
2. சரியான தமிழ் இலக்கணம் மற்றும் தொழில்முறை சொல்லகராதியை பயன்படுத்தவும்
3. விரிவான, தகவல் நிறைந்த பதில்களை வழங்கவும் (குறைந்தது 80 வார்த்தைகள்)
4. தெளிவான பத்திகள் மற்றும் தர்க்கரீதியான ஓட்டத்துடன் பதிலை அமைக்கவும்
5. முடிந்தவரை துல்லியமான மற்றும் நம்பகமான தகவல்களை வழங்கவும்
6. திரும்பத் திரும்ப வரும் வாக்கியங்கள் அல்லது நிரப்பு உள்ளடக்கத்தைத் தவிர்க்கவும்
7. கேள்வியின் சிக்கலான தன்மைக்கு ஏற்ற தொனியைப் பயன்படுத்தவும்
8. உண்மை சார்ந்த கேள்விகளுக்கு சூழல் மற்றும் பின்புலத்தை வழங்கவும்
9. தேவையற்ற முடிவு வார்த்தைகள் இல்லாமல் இயல்பாக பதிலை முடிக்கவும்
10. முற்றிலும் அவசியம் இல்லாவிட்டால் ஆங்கில வார்த்தைகளைத் தவிர்க்கவும்
11. பயனர்களை மரியாதையுடன் (நீங்கள்) உரையாடுக"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ மாடல் initialization தோல்வியடைந்தது: {str(e)}")
            return None
    
    def translate_text(self, text: str, src: str, dest: str) -> Optional[str]:
        """Optimized translation with caching and error handling"""
        if not text or not text.strip():
            return text
            
        try:
            translator = self.get_translator(src, dest)
            translated = translator.translate(text)
            return self.clean_repeated_text(translated) if translated else None
        except Exception as e:
            st.warning(f"⚠️ மொழிபெயர்ப்பு பிழை: {str(e)}")
            return None
    
    @lru_cache(maxsize=32)
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL with caching"""
        try:
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            return domain_match.group(1) if domain_match else None
        except:
            return None
    
    def extract_links(self, text: str) -> Optional[str]:
        """Extract and format valid URLs from response text"""
        try:
            # Clean existing link sections
            cleaned_text = re.sub(r'\n\n(தொடர்புடைய|நம்பகமான) இணைப்புகள்:.*$', '', text, flags=re.DOTALL)
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
            
            # Format links
            formatted_links = "\n\n---\n\n🔗 **தொடர்புடைய இணைப்புகள்:**\n\n"
            for url in valid_urls:
                domain = self._extract_domain(url)
                if domain:
                    display_name = domain.replace('www.', '')
                    formatted_links += f"• [{display_name}]({url})\n"
            
            return formatted_links.rstrip()
            
        except Exception as e:
            st.warning(f"⚠️ இணைப்பு பிரித்தெடுக்கும் பிழை: {str(e)}")
            return None
    
    def _create_optimized_prompt(self, question: str) -> str:
        """Create an optimized prompt for better responses"""
        return f"""இந்த கேள்விக்கு தமிழில் விரிவாக பதிலளிக்கவும்:

{question}

தேவைகள்:
- விரிவான, நன்கு அமைக்கப்பட்ட பதிலை வழங்கவும் (குறைந்தது 80 வார்த்தைகள்)
- தெளிவான பத்திகள் மற்றும் தர்க்கரீதியான ஓட்டத்தைப் பயன்படுத்தவும்
- தொடர்புடைய சூழல் மற்றும் பின்புல தகவல்களை சேர்க்கவும்
- தேவையான இடங்களில் குறிப்பிட்ட உண்மைகள் மற்றும் உदாரணங்களை மேற்கோள் காட்டவும்
- தொழில்முறை ஆனால் உரையாடல் தொனியை பராமரிக்கவும்
- துல்லியம் மற்றும் உதவிகரமான தன்மையில் கவனம் செலுத்தவும்

பல்வேறு அதிகாரபூர்வ ஆதாரங்களிலிருந்து (கல்வி, அரசு, செய்தி, ஆராய்ச்சி நிறுவனங்கள்) 3-5 நம்பகமான குறிப்பு இணைப்புகளை வழங்கவும்.
உங்கள் பதிலின் இறுதியில் இணைப்புகளை எளிய URL களாக வடிவமைக்கவும்."""
    
    def get_response(self, question: str) -> Optional[str]:
        """Get response from the model with error handling and optimization"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_tamil" not in st.session_state:
                st.session_state.chat_session_tamil = self._model.start_chat(history=[])
            
            # Create optimized prompt
            prompt = self._create_optimized_prompt(question)
            
            # Get response with timeout handling
            with st.spinner("🤔 சிந்தித்துக்கொண்டிருக்கிறேன்..."):
                response = st.session_state.chat_session_tamil.send_message(prompt)
                
                if not response or not response.text:
                    return "⚠️ எனக்கு வெற்று பதில் கிடைத்தது. தயவுசெய்து உங்கள் கேள்வியை வேறுவிதமாகக் கேட்க முயற்சிக்கவும்."
                
                response_text = response.text.strip()
                cleaned_response = self.clean_repeated_text(response_text)
                
                # Validate response quality
                if len(cleaned_response) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ பதில் மிகக் குறுகியதாகத் தோன்றுகிறது. மீண்டும் முயற்சித்துக்கொண்டிருக்கிறேன்...")
                    return None
                
                # Extract and append links
                links = self.extract_links(cleaned_response)
                final_response = cleaned_response + (links if links else "")
                
                return final_response
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API quota முடிந்துவிட்டது. தயவுசெய்து பின்னர் முயற்சிக்கவும்."
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ நெட்வொர்க் பிழை ஏற்பட்டது. தயவுசெய்து உங்கள் இணைப்பைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்."
            else:
                st.error(f"❌ பதில் உருவாக்குவதில் பிழை: {str(e)}")
                return None
    
    def get_direct_tamil_response(self, question: str) -> Optional[str]:
        """Get response directly in Tamil without translation"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_tamil" not in st.session_state:
                st.session_state.chat_session_tamil = self._model.start_chat(history=[])
            
            prompt_template = f"""
கேள்வி: {question}

பதில் தரும்போது கீழ்கண்ட வழிகாட்டுதல்களைப் பின்பற்றவும்:
1. தெளிவான, சரியான தமிழில் பதிலளிக்கவும்
2. தேவையான விவரங்களை விரிவாக விளக்கவும் (குறைந்தது 80 வார்த்தைகள்)
3. தமிழ் இலக்கண விதிகளைக் கடைப்பிடிக்கவும்
4. மீண்டும் மீண்டும் வார்த்தைகளைப் பயன்படுத்த வேண்டாம்
5. பதிலை முழுமையான வாக்கியங்களில் அளிக்கவும்
6. வெவ்வேறு வலைத்தளங்களிலிருந்து 4-5 நம்பகமான இணைப்புகளை வழங்கவும் (ஒரு வலைத்தளத்திலிருந்து ஒன்றுக்கு மேற்பட்ட இணைப்புகளை வழங்க வேண்டாம்)
7. இணைப்புகள் பல்வேறு வகையான மூலங்களிலிருந்து இருக்க வேண்டும் (எ.கா: செய்திகள், கல்வி, அரசு, ஆராய்ச்சி முதலியன)
8. இணைப்புகளின் விளக்கங்களைக் கொடுக்க வேண்டாம், URL மட்டும் கொடுக்கவும்
"""
            
            response = st.session_state.chat_session_tamil.send_message(prompt_template)
            cleaned_response = self.clean_repeated_text(response.text)
            
            # Extract and append links if available
            links = self.extract_links(cleaned_response)
            if links:
                cleaned_response += links
            
            return cleaned_response
            
        except Exception as e:
            st.warning(f"Direct Tamil response error: {str(e)}")
            return None
    
    def display_chat_history(self):
        """Display chat history with improved formatting"""
        if "chat_history_tamil" not in st.session_state:
            st.session_state.chat_history_tamil = [
                AIMessage(content="👋 **வணக்கம்!** நான் உங்கள் தமிழ் உதவியாளர். எதையும் கேட்க தயங்காதீர்கள்!")
            ]
        
        for message in st.session_state.chat_history_tamil:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
    
    def handle_user_input(self):
        """Handle user input with validation and processing"""
        user_query = st.chat_input(
            "💬 உங்கள் கேள்வியை தமிழில் இங்கே டைப் செய்யுங்கள்...", 
            key="tamil_chat_input",
            max_chars=1000
        )
        
        if user_query and user_query.strip():
            # Validate input
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ தயவுசெய்து மேலும் விரிவான கேள்வி கேளுங்கள்.")
                return
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Add user message to history
            st.session_state.chat_history_tamil.append(HumanMessage(content=cleaned_query))
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    # Try direct Tamil response first
                    result = self.get_direct_tamil_response(cleaned_query)
                    
                    if not result:
                        # Fallback to translation if direct response fails
                        translated_query = self.translate_text(cleaned_query, 'ta', 'en')
                        if translated_query:
                            result = self.get_response(translated_query)
                            if result:
                                result = self.translate_text(result, 'en', 'ta')
                    
                    if result:
                        st.markdown(result)
                        st.session_state.chat_history_tamil.append(AIMessage(content=result))
                        
                        # Show performance metrics
                        response_time = time.time() - start_time
                        if response_time > 0:
                            st.sidebar.success(f"⚡ பதில் நேரம்: {response_time:.2f}s")
                    else:
                        error_msg = "😔 எனக்கு பதில் உருவாக்குவதில் சிக்கல் உள்ளது. தயவுசெய்து உங்கள் கேள்வியை வேறுவிதமாகக் கேட்கவும் அல்லது பின்னர் முயற்சிக்கவும்."
                        st.error(error_msg)
                        st.session_state.chat_history_tamil.append(AIMessage(content=error_msg))
                        
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.warning(error_msg)
                    st.session_state.chat_history_tamil.append(AIMessage(content=error_msg))
    
    def run_chat_interface(self):
        """Main chat interface"""
        try:
            self.display_chat_history()
            self.handle_user_input()
            
        except Exception as e:
            st.error(f"❌ அரட்டை இடைமுகப் பிழை: {str(e)}")
            st.info("🔄 தயவுசெய்து பக்கத்தை புதுப்பித்து மீண்டும் முயற்சிக்கவும்.")

def main():
    """Main function that respects the st.flag dependency"""
    
    # Check authentication flag
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **அணுகல் மறுக்கப்பட்டது**: தமிழ் தொகுதியைப் பயன்படுத்த தயவுசெய்து உள்நுழையவும்.")
        st.info("👈 உங்கள் கணக்கில் உள்நுழைய பக்கப்பட்டியைப் பயன்படுத்தவும்.")
        return
    
    # Page configuration
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
    .tamil-text {
        font-family: 'Noto Sans Tamil', 'Latha', 'Vijaya', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>உதவியாளர் தமிழ் சாட்‌பாட்</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run chatbot
    try:
        chatbot = TamilChatBot()
        
        
        # Main chat interface
        chatbot.run_chat_interface()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666;' class='tamil-text'>தமிழ் மொழி ஆதரவுக்காக ❤️ உடன் உருவாக்கப்பட்டது</p>", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"❌ **பயன்பாட்டுப் பிழை**: {str(e)}")
        st.info("🔄 தயவுசெய்து பக்கத்தை புதுப்பிக்கவும். பிரச்சனை தொடர்ந்தால், ஆதரவைத் தொடர்பு கொள்ளவும்.")

if __name__ == "__main__":
    main()