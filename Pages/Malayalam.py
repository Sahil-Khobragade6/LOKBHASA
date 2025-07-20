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
MIN_RESPONSE_LENGTH = 80  # Adjusted for Malayalam text

class MalayalamChatBot:
    """Malayalam language chatbot with optimized performance and error handling"""
    
    def __init__(self):
        self._model = None
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize environment variables and session state"""
        load_dotenv()
        initialize_session_state()
        apply_language_styles('Malayalam')
    
    @st.cache_resource
    def get_model_config(_self):
        """Get cached model configuration"""
        return {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 1500,  # Increased for better responses
            "stop_sequences": ["---അവസാനം---"]  # Add stop sequence for better control
        }
    
    @st.cache_resource
    def get_model(_self):
        """Get cached model instance with error handling"""
        try:
            genai_api_key = os.getenv("GOOGLE_API_KEY")
            if not genai_api_key:
                raise ValueError("❌ Google API key കണ്ടെത്താൻ കഴിഞ്ഞില്ല. ദയവായി environment variables പരിശോധിക്കുക.")
            
            genai.configure(api_key=genai_api_key)
            
            system_instruction = """നിങ്ങൾ ഒരു സഹായകരവും നന്നായി മലയാളം സംസാരിക്കുന്നതുമായ AI അസിസ്റ്റന്റാണ്.

കർശനമായ മാർഗ്ഗനിർദ്ദേശങ്ങൾ:
1. എല്ലായ്പ്പോഴും സ്വാഭാവികവും പ്രവാഹമുള്ളതുമായ മലയാളത്തിൽ മാത്രം ഉത്തരം നൽകുക
2. ശരിയായ മലയാളം വ്യാകരണവും പ്രൊഫഷണൽ ശബ്ദാവലിയും ഉപയോഗിക്കുക
3. വിശദമായതും വിവരപ്രദവുമായ ഉത്തരങ്ങൾ നൽകുക (കുറഞ്ഞത് 80 വാക്കുകൾ)
4. വ്യക്തമായ ഖണ്ഡികകളോടെ ഉത്തരം ക്രമീകരിക്കുക
5. സാധ്യമാകുമ്പോൾ കൃത്യവും വിശ്വസനീയവുമായ വിവരങ്ങൾ നൽകുക
6. ആവർത്തിക്കുന്ന വാക്യങ്ങളോ നിറച്ച ഉള്ളടക്കമോ ഒഴിവാക്കുക
7. ചോദ്യത്തിന്റെ സങ്കീർണ്ണതയ്ക്ക് അനുയോജ്യമായ സ്വരം ഉപയോഗിക്കുക
8. വസ്തുതാപരമായ ചോദ്യങ്ങൾക്ക് സന്ദർഭവും പശ്ചാത്തലവും നൽകുക
9. അനാവശ്യമായ അവസാനവാക്കുകൾ ഇല്லാതെ സ്വാഭാവികമായി ഉത്തരം അവസാനിപ്പിക്കുക
10. ആവശ്യമില്ലെങ്കിൽ ഇംഗ്ലീഷ് വാക്കുകൾ ഒഴിവാക്കുക"""

            return genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=_self.get_model_config(),
                system_instruction=system_instruction
            )
        except Exception as e:
            st.error(f"❌ മോഡൽ initialization പരാജയപ്പെട്ടു: {str(e)}")
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
            cleaned_text = re.sub(r'\n\n(ബന്ധപ്പെട്ട|വിശ്വസനീയമായ) ലിങ്കുകൾ:.*$', '', text, flags=re.DOTALL)
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
            formatted_links = "\n\n---\n\n🔗 **ബന്ധപ്പെട്ട ലിങ്കുകൾ:**\n\n"
            for url in valid_urls:
                domain = self._extract_domain(url)
                if domain:
                    display_name = domain.replace('www.', '')
                    formatted_links += f"• [{display_name}]({url})\n"
            
            return formatted_links.rstrip()
            
        except Exception as e:
            st.warning(f"⚠️ ലിങ്ക് extraction പിശക്: {str(e)}")
            return None
    
    def _create_optimized_prompt(self, question: str) -> str:
        """Create an optimized prompt for better responses"""
        return f"""ഈ ചോദ്യത്തിന് സമഗ്രമായി മലയാളത്തിൽ ഉത്തരം നൽകുക:

{question}

ആവശ്യകതകൾ:
- വിശദവും നന്നായി ക്രമീകരിച്ചതുമായ ഉത്തരം നൽകുക (കുറഞ്ഞത് 80 വാക്കുകൾ)
- വ്യക്തമായ ഖണ്ഡികകളും ലോജിക്കൽ ഫ്ലോയും ഉപയോഗിക്കുക
- പ്രസക്തമായ സന്ദർഭവും പശ്ചാത്തല വിവരങ്ങളും ഉൾപ്പെടുത്തുക
- ആവശ്യമുള്ളിടത്ത് നിർദ്ദിഷ്ട വസ്തുതകളും ഉദാഹരണങ്ങളും ഉദ്ധരിക്കുക
- പ്രൊഫഷണൽ എന്നാൽ സംഭാഷണാത്മകമായ സ്വരം നിലനിർത്തുക
- കൃത്യതയിലും സഹായകരമായതിലും ശ്രദ്ധ കേന്ദ്രീകരിക്കുക

വിവിധ ആധികാരിക ഉറവിടങ്ങളിൽ (വിദ്യാഭ്യാസം, സർക്കാർ, വാർത്ത, ഗവേഷണ സ്ഥാപനങ്ങൾ) നിന്ന് 3-5 വിശ്വസനീയമായ റഫറൻസ് ലിങ്കുകൾ നൽകുക.
നിങ്ങളുടെ ഉത്തരത്തിന്റെ അവസാനത്തിൽ ലിങ്കുകൾ സാധാരണ URL-കളായി ഫോർമാറ്റ് ചെയ്യുക."""
    
    def get_response(self, question: str) -> Optional[str]:
        """Get response from the model with error handling and optimization"""
        try:
            if not self._model:
                self._model = self.get_model()
                if not self._model:
                    return None
            
            # Initialize chat session if needed
            if "chat_session_malayalam" not in st.session_state:
                st.session_state.chat_session_malayalam = self._model.start_chat(history=[])
            
            # Create optimized prompt
            prompt = self._create_optimized_prompt(question)
            
            # Get response with timeout handling
            with st.spinner("🤔 ചിന്തിക്കുന്നു..."):
                response = st.session_state.chat_session_malayalam.send_message(prompt)
                
                if not response or not response.text:
                    return "⚠️ എനിക്ക് ഒരു ശൂന്യമായ ഉത്തരം ലഭിച്ചു. ദയവായി നിങ്ങളുടെ ചോദ്യം മറ്റുവിധത്തിൽ ചോദിക്കാൻ ശ്രമിക്കുക."
                
                response_text = response.text.strip()
                
                # Validate response quality
                if len(response_text) < MIN_RESPONSE_LENGTH:
                    st.warning("⚠️ ഉത്തരം വളരെ ചെറുതായി തോന്നുന്നു. വീണ്ടും ശ്രമിക്കുന്നു...")
                    return None
                
                # Extract and append links
                links = self.extract_links(response_text)
                final_response = response_text + (links if links else "")
                
                return final_response
                
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "limit" in error_msg:
                return "⚠️ API quota കഴിഞ്ഞു. ദയവായി പിന്നീട് ശ്രമിക്കുക."
            elif "network" in error_msg or "connection" in error_msg:
                return "⚠️ നെറ്റ്‌വർക്ക് പിശക് സംഭവിച്ചു. ദയവായി നിങ്ങളുടെ കണക്ഷൻ പരിശോധിച്ച് വീണ്ടും ശ്രമിക്കുക."
            else:
                st.error(f"❌ ഉത്തരം ജനറേറ്റ് ചെയ്യുന്നതിൽ പിശക്: {str(e)}")
                return None
    
    def display_chat_history(self):
        """Display chat history with improved formatting"""
        if "chat_history_malayalam" not in st.session_state:
            st.session_state.chat_history_malayalam = [
                AIMessage(content="👋 **നമസ്കാരം!** ഞാൻ നിങ്ങളുടെ മലയാളം അസിസ്റ്റന്റാണ്. എന്തും ചോദിക്കാൻ മടിക്കേണ്ട!")
            ]
        
        for message in st.session_state.chat_history_malayalam:
            if isinstance(message, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message.content)
    
    def handle_user_input(self):
        """Handle user input with validation and processing"""
        user_query = st.chat_input(
            "💬 നിങ്ങളുടെ ചോദ്യം മലയാളത്തിൽ ഇവിടെ ടൈപ്പ് ചെയ്യൂ...", 
            key="malayalam_chat_input",
            max_chars=1000
        )
        
        if user_query and user_query.strip():
            # Validate input
            cleaned_query = user_query.strip()
            if len(cleaned_query) < 3:
                st.warning("⚠️ ദയവായി കൂടുതൽ വിശദമായ ചോദ്യം ചോദിക്കുക.")
                return
            
            # Record start time for performance tracking
            start_time = time.time()
            
            # Add user message to history
            st.session_state.chat_history_malayalam.append(HumanMessage(content=cleaned_query))
            
            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(cleaned_query)
            
            # Generate and display response
            with st.chat_message("assistant", avatar="🤖"):
                result = self.get_response(cleaned_query)
                
                if result:
                    st.markdown(result)
                    st.session_state.chat_history_malayalam.append(AIMessage(content=result))
                    
                    # Show performance metrics
                    response_time = time.time() - start_time
                    if response_time > 0:
                        st.sidebar.success(f"⚡ പ്രതികരണ സമയം: {response_time:.2f}s")
                else:
                    error_msg = "😔 എനിക്ക് ഉത്തരം ജനറേറ്റ് ചെയ്യുന്നതിൽ പ്രശ്നമുണ്ട്. ദയവായി നിങ്ങളുടെ ചോദ്യം മറ്റുവിധത്തിൽ ചോദിക്കുകയോ പിന്നീട് ശ്രമിക്കുകയോ ചെയ്യുക."
                    st.error(error_msg)
                    st.session_state.chat_history_malayalam.append(AIMessage(content=error_msg))
    
    def run_chat_interface(self):
        """Main chat interface"""
        try:
            self.display_chat_history()
            self.handle_user_input()
            
        except Exception as e:
            st.error(f"❌ ചാറ്റ് ഇന്റർഫേസ് പിശക്: {str(e)}")
            st.info("🔄 ദയവായി പേജ് റിഫ്രഷ് ചെയ്ത് വീണ്ടും ശ്രമിക്കുക.")

def main():
    """Main function that respects the st.flag dependency"""
    
    # Check authentication flag
    if not hasattr(st, 'flag') or not st.flag:
        st.error("🔒 **പ്രവേശനം നിഷേധിച്ചു**: മലയാളം മൊഡ്യൂൾ ഉപയോഗിക്കാൻ ദയവായി ലോഗിൻ ചെയ്യുക.")
        st.info("👈 നിങ്ങളുടെ അക്കൗണ്ടിലേക്ക് ലോഗിൻ ചെയ്യാൻ സൈഡ്ബാർ ഉപയോഗിക്കുക.")
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
    .malayalam-text {
        font-family: 'Noto Sans Malayalam', 'Kartika', 'Rachana', sans-serif;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>മലയാളം സഹായി ചാറ്റ്ബോട്ട്</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run chatbot
    try:
        chatbot = MalayalamChatBot()
        
       
        
        # Main chat interface
        chatbot.run_chat_interface()
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align: center; color: #666;' class='malayalam-text'>മലയാളം ഭാഷാ പിന്തുണയ്ക്കായി ❤️ ഓടെ നിർമ്മിച്ചത്</p>", 
                unsafe_allow_html=True
            )
            
    except Exception as e:
        st.error(f"❌ **ആപ്ലിക്കേഷൻ പിശക്**: {str(e)}")
        st.info("🔄 ദയവായി പേജ് റിഫ്രഷ് ചെയ്യുക. പ്രശ്നം തുടരുകയാണെങ്കിൽ, സപ്പോർട്ടിനെ ബന്ധപ്പെടുക.")

if __name__ == "__main__":
    main()