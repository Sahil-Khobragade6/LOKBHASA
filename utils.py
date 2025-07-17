import streamlit as st
from typing import Dict, Any

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'font_size': 'medium',
            'theme': 'light',
            'last_visited_page': None
        }

def get_language_config(language: str) -> Dict[str, Any]:
    """Get configuration for specific language."""
    configs = {
        'Telugu': {
            'welcome_message': 'తెలుగు భాషలో స్వాగతం',
            'font_family': 'Mandali',
            'direction': 'ltr'
        },
        'Hindi': {
            'welcome_message': 'हिंदी भाषा में स्वागत है',
            'font_family': 'Noto Sans Devanagari',
            'direction': 'ltr'
        },
        'Tamil': {
            'welcome_message': 'தமிழ் மொழியில் வரவேற்கிறோம்',
            'font_family': 'Noto Sans Tamil',
            'direction': 'ltr'
        }
    }
    return configs.get(language, {})

def apply_language_styles(language: str) -> None:
    """Apply language-specific styles."""
    config = get_language_config(language)
    if config:
        st.markdown(f"""
            <style>
                .stApp {{
                    font-family: '{config['font_family']}', sans-serif;
                    direction: {config['direction']};
                }}
            </style>
        """, unsafe_allow_html=True)

def save_user_preferences(preferences: Dict[str, Any]) -> None:
    """Save user preferences to session state."""
    st.session_state.user_preferences.update(preferences)

def get_user_preferences() -> Dict[str, Any]:
    """Get current user preferences."""
    return st.session_state.user_preferences 