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
MIN_RESPONSE_LENGTH = 150  # Telugu text tends to be longer

class TeluguChatBot:
    """Telugu language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._translator_cache = {}
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment variables and session state"""
        load_dotenv()
        initialize_session_state()
        apply_language_styles('Telugu')
    
    @st.cache_resource
    def get_model_config(_self):
        """Get cached model configuration"""
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1800,  # Increased for Telugu responses
            "stop_sequences": ["---END---"]
        }
    
    @st.cache_resource
    def get_model(_self):
        """Get cached model instance with error handling"""
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google API key లేదు. దయచేసి environment variables చూడండి.")
            
            genai.configure(api_key=genai_api_key)
            
            system_instruction = """మీరు తెలుగు భాషలో మాట్లాడే సహాయకులు. మీరు ఈ నియమాలను కఠినంగా పాటించాలి:

కఠిన మార్గదర్శకాలు:
1. ఎల్లప్పుడూ సహజమైన, స్వాభావిక తెలుగులో మాత్రమే సమాధానం ఇవ్వండి
2. సరైన తెలుగు వ్యాకరణం మరియు వృత్తిపరమైన పదజాలం ఉపయోగించండి
3. వివరంగా, సమాచారాత్మక సమాధానాలు ఇవ్వండి (కనీసం 150 పదాలు)
4. స్పష్టమైన పేరాగ్రాఫ్‌లతో మీ సమాధానాన్ని నిర్మించండి
5. అవసరమైనప్పుడు విశ్వసనీయ సమాచారాన్ని ఉదహరించండి
6. పునరావృత పదాలు లేదా అనవసర కంటెంట్‌ను నివారించండి
7. ప్రశ్న సంక్లిష్టత ఆధారంగా తగిన స్వരం ఉపయోగించండి
8. వాస్తవిక ప్రశ్నలకు, సందర్భం మరియు నేపథ్య సమాచారం అందించండి
9. అనవసర ముగింపులు లేకుండా సహజంగా సమాధానాలను ముగించండి
10. వినియోగదారులను సంబోధించేటప్పుడు తగిన తెలుగు గౌరవ పదాలు (మీరు) ఉపయోగించండి"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ Model initialization విఫలమైంది: {str(e)}")
            return None
    
    @lru_cache(maxsize=32)
    def get_translator(self, src: str, dest: str) -> Optional[GoogleTranslator]:
        """Get cached translator instance"""
        try:
            return GoogleTranslator(source=src, target=dest)
        except Exception as e:
            st.warning(f"⚠️ Translator initialization error: {str(e)}")
            return None
    
    def translate_text(self, text: str, src: str, dest: str) -> Optional[str]:
        """Translate text with error handling and caching"""
        try:
            translator = self.get_translator(src, dest)
            if translator:
                return translator.translate(text)
            return None
        except Exception as e:
            st.warning(f"⚠️ Translation error: {str(e)}")
            return None
    
    def clean_repeated_text(self, text: str) -> str:
        """Remove repeated words and phrases from Telugu text"""
        try:
            # Split text into words
            words = text.split()
            
            # Remove consecutive repeated words
            cleaned_words = []
            for i, word in enumerate(words):
                if i == 0 or word != words[i-1]:
                    cleaned_words.append(word)
            
            # Remove repeated phrases (up to 3 words)
            final_words = []
            for i in range(len(cleaned_words)):
                # Check for 2-word repetition
                if i >= 2 and cleaned_words[i-1:i+1] == cleaned_words[i-3:i-1]:
                    continue
                # Check for 3-word repetition
                if i >= 4 and cleaned_words[i-2:i+1] == cleaned_words[i-5:i-2]:
                    continue
                final_words.append(cleaned_words[i])
            
            return ' '.join(final_words)
        except Exception as e:
            st.warning(f"⚠️ Text cleaning error: {str(e)}")
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
        """Extract and format valid URLs from response text"""
        try:
            # Clean existing link sections
            text = re.sub(r'\n\nసంబంధిత లింక్‌లు:.*$', '', text, flags=re.DOTALL)
            text = re.sub(r'\n\nవిశ్వసనీయ లింక్‌లు:.*$', '', text, flags=re.DOTALL)
            
            # Clean up markdown-formatted links
            text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\2', text)
            text = re.sub(r'\d+\.\s*https?://[^\s]+', '', text)
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            
            if not urls:
                return None
            
            # Process and deduplicate URLs
            valid_urls = []
            seen_domains = set()
            
            for url in urls[:MAX_LINKS * 2]:
                # Clean URL
                clean_url = url.strip('()[].,!?').rstrip('.')
                clean_url = re.sub(r'\s*\d+\.?$', '', clean_url)
                
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
            formatted_links = "\n\n---\n\n🔗 **సంబంధిత వనరులు:**\n\n"
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
        """Create an optimized prompt for better Telugu responses"""
        return f"""ఈ ప్రశ్నకు తెలుగులో సమగ్రమైన సమాధానం ఇవ్వండి:

{question}

అవసరాలు:
- వివరణాత్మక, బాగా నిర్మించిన సమాధానం ఇవ్వండి (కనీసం 150 పదాలు)
- స్పష్టమైన పేరాగ్రాఫ్‌లు మరియు లాజికల్ ఫ్లోను ఉపయోగించండి
- సంబంధిత సందర్భం మరియు నేపథ్య సమాచారాన్ని చేర్చండి
- తగిన చోట నిర్దిష్ట వాస్తవాలు మరియు ఉదాహరణలను ఉదహరించండి
- వృత్తిపరమైన కానీ సంభాషణాత్మక స్వరాన్ని కొనసాగించండి
- ఖచ్చితత్వం మరియు సహాయకారిత్వంపై దృష్టి పెట్టండి

దయచేసి వేర్వేరు అధికారిక మూలాధారాల నుండి (విద్యా, ప్రభుత్వ, వార్తలు, పరిశోధనా సంస్థలు).
మీ సమాధానం చివరిలో లింక్‌లను సాధారణ URLలుగా ఫార్మాట్ చేయండి."""
    
    def get_direct_telugu_response(self, question: str) -> Optional[str]:
        """Get response directly in Telugu with error handling and optimization"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_telugu" not in st.session_state:
                st.session_state.chat_session_telugu = self._model.start_chat(history=[])
            
            # Create optimized prompt
            prompt = self._create_optimized_prompt(question)
            
            # Get response with timeout handling
            with st.spinner("🤔 ఆలోచిస్తున్నాను..."):
                response = st.session_state.chat_session_telugu.send_message(prompt)
                
                if not response or not response.text:
                    return "⚠️ ఖాలీ సమాధానం వచ్చింది. దయచేసి మీ ప్రశ్నను మరోవిధంగా అడగండి."
                
                response_text = response.text.strip()
                
                # Clean repeated text
                cleaned_response = self.clean_repeated_text(response_text)
                
                # Validate response quality
                if len(cleaned_response) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ సమాధానం చాలా చిన్నగా ఉంది. మళ్లీ ప్రయత్నిస్తున్నాను...")
                    return None
                
                # Extract and append links
                links = self.extract_links(cleaned_response)
                final_response = cleaned_response + (links if links else "")
                
                return final_response
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API కోటా మించిపోయింది. దయచేసి కొంత సేపు తర్వాత ప్రయత్నించండి."
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ నెట్‌వర్క్ లోపం సంభవించింది. దయచేసి మీ కనెక్షన్ చూసి మళ్లీ ప్రయత్నించండి."
            else:
                st.error(f"❌ సమాధానం వచ్చేటప్పుడు లోపం: {str(e)}")
                return None
    
    def get_fallback_response(self, question: str) -> Optional[str]:
        """Fallback response using translation if direct Telugu response fails"""
        try:
            # Translate question to English
            translated_query = self.translate_text(question, src='te', dest='en')
            if not translated_query:
                return None
            
            # Get English response
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            if "chat_session_telugu_fallback" not in st.session_state:
                st.session_state.chat_session_telugu_fallback = self._model.start_chat(history=[])
            
            english_prompt = f"""Answer this question comprehensively in English:
            
            {translated_query}
            
            Provide a detailed response with proper context and examples.
            Include 3-5 reliable reference links from different sources."""
            
            response = st.session_state.chat_session_telugu_fallback.send_message(english_prompt)
            
            if response and response.text:
                # Translate back to Telugu
                telugu_response = self.translate_text(response.text, src='en', dest='te')
                if telugu_response:
                    return self.clean_repeated_text(telugu_response)
            
            return None
            
        except Exception as e:
            st.warning(f"⚠️ Fallback response error: {str(e)}")
            return None
    
    def display_chat_history(self):
        """Display chat history with improved formatting"""
        if "chat_history_telugu" not in st.session_state:
            st.session_state.chat_history_telugu = [
                AIMessage(content="🙏 **నమస్కారం!** నేను మీ తెలుగు సహాయకుడిని. మీ ప్రశ్నలను తెలుగులో అడగండి!")
            ]
        
        for message in st.session_state.chat_history_telugu:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
    
    def handle_user_input(self):
        """Handle user input with validation and processing"""
        user_query = st.chat_input(
            "💬 మీ ప్రశ్నను తెలుగులో ఇక్కడ టైప్ చేయండి...", 
            key="telugu_chat_input",
            max_chars=1000
        )
        
        if user_query and user_query.strip():
            # Validate input
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ దయచేసి మరింత వివరణాత్మక ప్రశ్న అడగండి.")
                return
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Add user message to history
            st.session_state.chat_history_telugu.append(HumanMessage(content=cleaned_query))
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                # Try direct Telugu response first
                result = self.get_direct_telugu_response(cleaned_query)
                
                # Fallback to translation if direct response fails
                if not result:
                    st.info("🔄 ప్రత్యామ్నాయ పద్ధతిని ప్రయత్నిస్తున్నాను...")
                    result = self.get_fallback_response(cleaned_query)
                
                if result:
                    st.markdown(result)
                    st.session_state.chat_history_telugu.append(AIMessage(content=result))
                    
                    # Show performance metrics
                    response_time = time.time() - start_time
                    if response_time > 0:
                        st.sidebar.success(f"⚡ ప్రతిస్పందన సమయం: {response_time:.2f}s")
                else:
                    error_msg = "😔 సమాధానం ఇవ్వడంలో ఇబ్బంది ఎదురవుతోంది. దయచేసి మీ ప్రశ్నను మరోవిధంగా అడగండి లేదా కొంత సేపు తర్వాత ప్రయత్నించండి."
                    st.error(error_msg)
                    st.session_state.chat_history_telugu.append(AIMessage(content=error_msg))
    
    def run_chat_interface(self):
        """Main chat interface"""
        try:
            self.display_chat_history()
            self.handle_user_input()
            
        except Exception as e:
            st.error(f"❌ చాట్ ఇంటర్‌ఫేస్ లోపం: {str(e)}")
            st.info("🔄 దయచేసి పేజీని రిఫ్రెష్ చేసి మళ్లీ ప్రయత్నించండి.")

def main():
    """Main function that respects the st.flag dependency"""
    
    # Check authentication flag
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **యాక్సెస్ నిరాకరించబడింది**: దయచేసి తెలుగు మాడ్యూల్ ఉపయోగించడానికి లాగిన్ చేయండి.")
        st.info("👈 మీ ఖాతాలోకి లాగిన్ అవడానికి సైడ్‌బార్ ఉపయోగించండి.")
        return
    
    # Page configuration
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #ff6b35 0%, #f7931e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
        font-family: 'Noto Sans Telugu', sans-serif;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
    }
    .telugu-text {
        font-family: 'Noto Sans Telugu', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>తెలుగు సహాయక చాట్‌బాట్</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run chatbot
    try:
        chatbot = TeluguChatBot()
        
        
        
        
        # Main chat interface
        chatbot.run_chat_interface()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666; font-family: \"Noto Sans Telugu\", sans-serif;'>తెలుగు భాష మద్దతు కోసం ❤️ తో తయారు చేయబడింది</p>", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"❌ **అప్లికేషన్ లోపం**: {str(e)}")
        st.info("🔄 దయచేసి పేజీని రిఫ్రెష్ చేయండి. సమస్య కొనసాగితే, మద్దతును సంప్రదించండి.")

if __name__ == "__main__":
    main()