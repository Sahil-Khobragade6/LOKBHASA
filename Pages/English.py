import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from functools import lru_cache
import time
import re
from typing import Optional, List
from utils import initialize_session_state, apply_language_styles, save_user_preferences, get_user_preferences

# Constants
MAX_LINKS = 5
RESPONSE_TIMEOUT = 30
MIN_RESPONSE_LENGTH = 100

class EnglishChatBot:
    """English language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment variables and session state"""
        load_dotenv()
        initialize_session_state()
        apply_language_styles('English')
    
    @st.cache_resource
    def get_model_config(_self):
        """Get cached model configuration"""
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1500,  # Increased for better responses
            "stop_sequences": ["---END---"]  # Add stop sequence for better control
        }
    
    @st.cache_resource
    def get_model(_self):
        """Get cached model instance with error handling"""
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google API key not found. Please check your environment variables.")
            
            genai.configure(api_key=genai_api_key)
            
            system_instruction = """You are a helpful, fluent English assistant specialized in providing comprehensive answers.

STRICT GUIDELINES:
1. Always respond in natural, fluent English only
2. Use proper grammar and professional vocabulary
3. Provide detailed, informative responses (minimum 100 words)
4. Structure your answer with clear paragraphs
5. Be accurate and cite reliable information when possible
6. Avoid repetitive phrases or filler content
7. Use appropriate tone based on the question complexity
8. For factual questions, provide context and background
9. End responses naturally without unnecessary closings"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ Model initialization failed: {str(e)}")
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
            cleaned_text = re.sub(r'\n\n(Related|Useful) Links:.*$', '', text, flags=re.DOTALL)
            
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
            formatted_links = "\n\n---\n\n🔗 **Related Resources:**\n\n"
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
        """Create an optimized prompt for better responses"""
        return f"""Answer this question comprehensively in English:

{question}

Requirements:
- Provide a detailed, well-structured response (minimum 100 words)
- Use clear paragraphs and logical flow
- Include relevant context and background information
- Cite specific facts and examples where appropriate
- Maintain professional yet conversational tone
- Focus on accuracy and helpfulness

Please provide 3-5 reliable reference links from different authoritative sources (educational, government, news, research institutions).
Format links as simple URLs at the end of your response."""
    
    def get_response(self, question: str) -> Optional[str]:
        """Get response from the model with error handling and optimization"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_english" not in st.session_state:
                st.session_state.chat_session_english = self._model.start_chat(history=[])
            
            # Create optimized prompt
            prompt = self._create_optimized_prompt(question)
            
            # Get response with timeout handling
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chat_session_english.send_message(prompt)
                
                if not response or not response.text:
                    return "⚠️ I received an empty response. Please try rephrasing your question."
                
                response_text = response.text.strip()
                
                # Validate response quality
                if len(response_text) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ Response seems too short. Trying again...")
                    return None
                
                # Extract and append links
                links = self.extract_links(response_text)
                final_response = response_text + (links if links else "")
                
                return final_response
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API quota exceeded. Please try again later."
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ Network error occurred. Please check your connection and try again."
            else:
                st.error(f"❌ Error generating response: {str(e)}")
                return None
    
    def display_chat_history(self):
        """Display chat history with improved formatting"""
        if "chat_history_english" not in st.session_state:
            st.session_state.chat_history_english = [
                AIMessage(content="👋 **Hello!** I'm your English assistant. Feel free to ask me anything!")
            ]
        
        for message in st.session_state.chat_history_english:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
    
    def handle_user_input(self):
        """Handle user input with validation and processing"""
        user_query = st.chat_input(
            "💬 Ask me anything in English...", 
            key="english_chat_input",
            max_chars=1000
        )
        
        if user_query and user_query.strip():
            # Validate input
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ Please ask a more detailed question.")
                return
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Add user message to history
            st.session_state.chat_history_english.append(HumanMessage(content=cleaned_query))
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                result = self.get_response(cleaned_query)
                
                if result:
                    st.markdown(result)
                    st.session_state.chat_history_english.append(AIMessage(content=result))
                    
                    # Show performance metrics
                    response_time = time.time() - start_time
                    if response_time > 0:
                        st.sidebar.success(f"⚡ Response time: {response_time:.2f}s")
                else:
                    error_msg = "😔 I'm having trouble generating a response. Please try rephrasing your question or try again later."
                    st.error(error_msg)
                    st.session_state.chat_history_english.append(AIMessage(content=error_msg))
    
    def run_chat_interface(self):
        """Main chat interface"""
        try:
            self.display_chat_history()
            self.handle_user_input()
            
        except Exception as e:
            st.error(f"❌ Chat interface error: {str(e)}")
            st.info("🔄 Please refresh the page and try again.")

def main():
    """Main function that respects the st.flag dependency"""
    
    # Check authentication flag
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **Access Denied**: Please log in to use the English module.")
        st.info("👈 Use the sidebar to log in to your account.")
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>English Assistant Chatbot</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run chatbot
    try:
        chatbot = EnglishChatBot()
        
       
        
        # Main chat interface
        chatbot.run_chat_interface()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666;'>Made with ❤️ for English language support</p>", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"❌ **Application Error**: {str(e)}")
        st.info("🔄 Please refresh the page. If the problem persists, contact support.")

if __name__ == "__main__":
    main()