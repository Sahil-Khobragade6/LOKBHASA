import streamlit as st
import sqlite3
import time
import re
import hashlib
from functools import lru_cache
from contextlib import contextmanager
from datetime import datetime, timedelta

# ---------------- Enhanced Database Functions ----------------
@contextmanager
def get_db_connection():
    """Context manager for database connections with better error handling"""
    conn = None
    try:
        conn = sqlite3.connect('users.db', timeout=10)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        yield conn
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def create_usertable():
    """Create enhanced users table with additional fields"""
    with get_db_connection() as conn:
        if conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                login_attempts INTEGER DEFAULT 0,
                is_locked BOOLEAN DEFAULT 0,
                preferred_language TEXT DEFAULT 'English'
            )''')
            conn.commit()

def hash_password(password):
    """Hash password with salt for security"""
    salt = "lokbasha_secure_salt_2024"  # In production, use random salt per user
    return hashlib.sha256((password + salt).encode()).hexdigest()

def add_user(username, email, password):
    """Add new user with enhanced validation"""
    try:
        with get_db_connection() as conn:
            if conn:
                c = conn.cursor()
                hashed_password = hash_password(password)
                c.execute('''INSERT INTO users(username, email, password) 
                           VALUES (?, ?, ?)''', (username, email, hashed_password))
                conn.commit()
                return True, "Account created successfully!"
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already exists. Please choose a different one."
        elif "email" in str(e):
            return False, "Email already registered. Please use a different email."
        else:
            return False, "Registration failed. Please try again."
    except Exception as e:
        return False, f"Registration error: {str(e)}"

def login_user(username, password):
    """Enhanced user authentication with attempt tracking"""
    try:
        with get_db_connection() as conn:
            if conn:
                c = conn.cursor()
                
                # Check if account is locked
                c.execute('SELECT login_attempts, is_locked FROM users WHERE username = ?', (username,))
                user_status = c.fetchone()
                
                if user_status and user_status['is_locked']:
                    return False, "Account is temporarily locked due to multiple failed attempts. Please try again later."
                
                # Verify credentials
                hashed_password = hash_password(password)
                c.execute('''SELECT id, username, email, preferred_language 
                           FROM users WHERE username = ? AND password = ?''', 
                         (username, hashed_password))
                user = c.fetchone()
                
                if user:
                    # Reset login attempts and update last login
                    c.execute('''UPDATE users SET login_attempts = 0, is_locked = 0, 
                               last_login = CURRENT_TIMESTAMP WHERE username = ?''', (username,))
                    conn.commit()
                    return True, dict(user)
                else:
                    # Increment login attempts
                    if user_status:
                        attempts = user_status['login_attempts'] + 1
                        is_locked = 1 if attempts >= 5 else 0
                        c.execute('''UPDATE users SET login_attempts = ?, is_locked = ? 
                                   WHERE username = ?''', (attempts, is_locked, username))
                        conn.commit()
                        
                        if is_locked:
                            return False, "Too many failed attempts. Account temporarily locked."
                        else:
                            return False, f"Invalid credentials. {5-attempts} attempts remaining."
                    else:
                        return False, "Invalid username or password."
                        
    except Exception as e:
        return False, f"Login error: {str(e)}"

# ---------------- Validation Functions ----------------
def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username format"""
    if len(username) < 3:
        return False, "Username must be at least 3 characters long"
    if len(username) > 20:
        return False, "Username must be less than 20 characters"
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores"
    return True, "Valid username"

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one number"
    return True, "Strong password"

# ---------------- Session State Management ----------------
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'logged_in': False,
        'username': '',
        'user_email': '',
        'user_id': None,
        'flag': False,
        'selected_language': 'English',
        'current_page': 'home',
        'login_attempts': 0,
        'show_password_tips': False,
        'registration_step': 1,
        'form_data': {},
        'theme': 'light',
        'show_forgot_password': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ---------------- UI Components ----------------
def render_custom_css():
    """Render custom CSS for enhanced UI"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        padding: 0;
    }
    
    /* Login Container */
    .login-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', sans-serif;
    }
    
    .login-card {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        max-width: 450px;
        width: 100%;
        margin: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .login-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styles */
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        color: #718096;
        font-size: 1rem;
        margin: 0;
        margin-bottom: 2rem;
    }
    
    /* Form Styles */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        border: none;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f7fafc;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Feature Cards */
    .feature-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .feature-title {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 12px;
        border: none;
        font-weight: 500;
    }
    
    /* Password Strength Indicator */
    .password-strength {
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 0.875rem;
    }
    
    .strength-weak {
        background-color: #fed7d7;
        color: #c53030;
    }
    
    .strength-medium {
        background-color: #fef5e7;
        color: #d69e2e;
    }
    
    .strength-strong {
        background-color: #d1fae5;
        color: #059669;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #e2e8f0;
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive Design */
    @media (max-width: 640px) {
        .login-card {
            margin: 1rem;
            padding: 2rem;
        }
        
        .app-title {
            font-size: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_password_strength_indicator(password):
    """Render password strength indicator"""
    if not password:
        return
    
    strength_score = 0
    feedback = []
    
    # Check length
    if len(password) >= 8:
        strength_score += 1
    else:
        feedback.append("At least 8 characters")
    
    # Check for uppercase
    if re.search(r'[A-Z]', password):
        strength_score += 1
    else:
        feedback.append("One uppercase letter")
    
    # Check for lowercase
    if re.search(r'[a-z]', password):
        strength_score += 1
    else:
        feedback.append("One lowercase letter")
    
    # Check for numbers
    if re.search(r'\d', password):
        strength_score += 1
    else:
        feedback.append("One number")
    
    # Check for special characters
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        strength_score += 1
    else:
        feedback.append("One special character")
    
    # Display strength
    if strength_score <= 2:
        st.markdown(
            f'<div class="password-strength strength-weak">🔴 Weak Password<br>Missing: {", ".join(feedback)}</div>',
            unsafe_allow_html=True
        )
    elif strength_score <= 3:
        st.markdown(
            f'<div class="password-strength strength-medium">🟡 Medium Password<br>Add: {", ".join(feedback)}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="password-strength strength-strong">🟢 Strong Password</div>',
            unsafe_allow_html=True
        )

def render_features_section():
    """Render features section"""
    st.markdown("### 🌟 Why Choose LokBasha?")
    
    features = [
        ("🗣️", "Multi-Language Support", "Chat in your native language - Telugu, Hindi, Tamil, Malayalam, and more"),
        ("⚡", "Lightning Fast", "Get instant responses powered by advanced AI technology"),
        ("🔒", "Secure & Private", "Your conversations are protected with enterprise-grade security"),
        ("🎯", "Accurate Responses", "Get precise answers with relevant context and examples"),
        ("📚", "Rich Knowledge Base", "Access comprehensive information across various topics"),
        ("🌐", "Always Available", "24/7 availability for all your questions and queries")
    ]
    
    for icon, title, description in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-title">{icon} {title}</div>
            <div class="feature-description">{description}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------- Authentication Interface ----------------
def login_signup():
    """Enhanced login and signup interface"""
    st.set_page_config(
        page_title="LokBasha - Your Native Language AI Assistant",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    render_custom_css()
    
    # Create database table
    create_usertable()
    
    # Main container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-card">
            <div class="login-header">
                <div class="logo">🌐</div>
                <h1 class="app-title">LokBasha</h1>
                <p class="app-subtitle">Your Native Language AI Assistant</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Tab-based interface
        tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab1:
            render_login_form()
        
        with tab2:
            render_signup_form()
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Features section in sidebar or below
    with st.sidebar:
        render_features_section()

def render_login_form():
    """Render enhanced login form"""
    st.markdown("#### Welcome Back!")
    st.markdown("Sign in to continue your journey with LokBasha")
    
    with st.form("login_form", clear_on_submit=False):
        # Username/Email field
        username = st.text_input(
            "👤 Username or Email",
            placeholder="Enter your username or email",
            help="You can use either your username or email address"
        )
        
        # Password field
        password = st.text_input(
            "🔒 Password",
            type="password",
            placeholder="Enter your password",
            help="Enter your account password"
        )
        
        # Remember me checkbox
        remember_me = st.checkbox("Remember me")
        
        # Login button
        login_btn = st.form_submit_button(
            "🚀 Sign In",
            type="primary",
            use_container_width=True
        )
        
        if login_btn:
            if username and password:
                with st.spinner("Signing you in..."):
                    success, result = login_user(username, password)
                    
                    if success:
                        # Store user information
                        st.session_state.logged_in = True
                        st.session_state.username = result['username']
                        st.session_state.user_email = result['email']
                        st.session_state.user_id = result['id']
                        st.session_state.selected_language = result['preferred_language']
                        st.session_state.flag = True
                        
                        st.success("✅ Welcome back! Redirecting to your dashboard...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
            else:
                st.warning("⚠️ Please fill in all fields")
    
    # Forgot password button (outside the form)
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("🔄 Forgot Password?", help="Click to reset your password"):
            st.session_state.show_forgot_password = True
            st.rerun()
    
    # Show forgot password section if requested
    if st.session_state.show_forgot_password:
        st.markdown("---")
        st.markdown("#### 🔄 Password Reset")
        
        with st.form("forgot_password_form"):
            email_reset = st.text_input(
                "📧 Enter your email address",
                placeholder="Enter the email associated with your account"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.form_submit_button("Send Reset Link", type="primary"):
                    if email_reset and validate_email(email_reset):
                        st.info("🔄 Password reset feature is coming soon! Please contact support for assistance.")
                    else:
                        st.error("Please enter a valid email address")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_forgot_password = False
                    st.rerun()

def render_signup_form():
    """Render enhanced signup form"""
    st.markdown("#### Create Your Account")
    st.markdown("Join thousands of users exploring knowledge in their native language")
    
    with st.form("signup_form", clear_on_submit=False):
        # Username field
        new_user = st.text_input(
            "👤 Choose Username",
            placeholder="Enter a unique username",
            help="3-20 characters, letters, numbers, and underscores only"
        )
        
        # Email field
        email = st.text_input(
            "📧 Email Address",
            placeholder="Enter your email address",
            help="We'll use this for important account notifications"
        )
        
        # Password field
        new_pass = st.text_input(
            "🔒 Create Password",
            type="password",
            placeholder="Create a strong password",
            help="At least 8 characters with uppercase, lowercase, and numbers"
        )
        
        # Show password strength if password is entered
        if new_pass:
            render_password_strength_indicator(new_pass)
        
        # Confirm password field
        confirm_pass = st.text_input(
            "🔒 Confirm Password",
            type="password",
            placeholder="Confirm your password"
        )
        
        # Language preference
        preferred_language = st.selectbox(
            "🌐 Preferred Language",
            ["English", "Telugu", "Hindi", "Tamil", "Malayalam"],
            help="Choose your preferred language for the interface"
        )
        
        # Terms and conditions
        terms_accepted = st.checkbox(
            "I agree to the Terms of Service and Privacy Policy",
            help="Please accept our terms to create an account"
        )
        
        # Signup button
        signup_btn = st.form_submit_button(
            "📝 Create Account",
            type="primary",
            use_container_width=True
        )
        
        if signup_btn:
            # Validation
            if not all([new_user, email, new_pass, confirm_pass]):
                st.warning("⚠️ Please fill in all fields")
                return
            
            if not terms_accepted:
                st.warning("⚠️ Please accept the Terms of Service to continue")
                return
            
            # Validate username
            username_valid, username_msg = validate_username(new_user)
            if not username_valid:
                st.error(f"❌ {username_msg}")
                return
            
            # Validate email
            if not validate_email(email):
                st.error("❌ Please enter a valid email address")
                return
            
            # Validate password
            password_valid, password_msg = validate_password(new_pass)
            if not password_valid:
                st.error(f"❌ {password_msg}")
                return
            
            # Check password confirmation
            if new_pass != confirm_pass:
                st.error("❌ Passwords do not match")
                return
            
            # Create account
            with st.spinner("Creating your account..."):
                success, message = add_user(new_user, email, new_pass)
                
                if success:
                    st.success("✅ Account created successfully! Please login to continue.")
                    st.balloons()
                    
                    # Clear form
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ {message}")

# ---------------- Main App Functions ----------------
@lru_cache(maxsize=1)
def get_language_options():
    """Get available language options (cached)"""
    return ["English", "Telugu", "Hindi", "Tamil", "Malayalam"]

def main_app(username):
    """Main application interface after login"""
    st.set_page_config(
        page_title="LokBasha Dashboard",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced sidebar
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {username}!")
        
        # User profile section
        with st.expander("👤 Profile", expanded=False):
            st.write(f"**Username:** {st.session_state.username}")
            st.write(f"**Email:** {st.session_state.user_email}")
            st.write(f"**Language:** {st.session_state.selected_language}")
        
        # Language selector
        st.markdown("### 🌐 Select Language")
        current_index = get_language_options().index(st.session_state.selected_language)
        language = st.selectbox(
            "Choose your preferred language:",
            get_language_options(),
            index=current_index,
            key="language_selector"
        )
        
        if language != st.session_state.selected_language:
            st.session_state.selected_language = language
            st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", type="primary", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("👋 Logged out successfully!")
            time.sleep(1)
            st.rerun()
    
    # Main content area
    st.title("🌐 LokBasha: Your Native Language AI Assistant")
    
    # Dashboard content based on selected language
    if st.session_state.selected_language != "English":
        st.info(f"🗣️ You have selected **{st.session_state.selected_language}**. The chatbot module will load here.")
        
        # Here you would load the specific language module
        # For demonstration, showing a placeholder
        st.markdown(f"""
        ### 🚀 {st.session_state.selected_language} Chat Assistant
        
        The {st.session_state.selected_language} language module would be loaded here.
        You can now start chatting in your native language!
        """)
        
    else:
        # English welcome screen
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🙏 Welcome to LokBasha!
            
            Thank you for joining our community of learners who prefer to explore knowledge in their native language.
            
            #### 🚀 Getting Started:
            1. **Select a Language** from the sidebar
            2. **Start Chatting** in your preferred language  
            3. **Ask Questions** and get detailed responses
            4. **Explore Topics** with cultural context
            
            #### ✨ Features Available:
            - 🗣️ **Natural Conversations** in your language
            - 🔄 **Real-time Responses** with AI assistance
            - 📚 **Rich Information** with relevant examples
            - 🌐 **Cultural Context** for better understanding
            """)
        
        with col2:
            st.markdown("### 🌏 Available Languages")
            for lang in get_language_options()[1:]:
                st.markdown(f"🔸 {lang}")
            
            

# ---------------- Main Entry Point ----------------
def main():
    """Main application entry point"""
    initialize_session_state()
    
    if st.session_state.logged_in:
        st.flag = True
        main_app(st.session_state.username)
    else:
        st.flag = False
        login_signup()

if __name__ == "__main__":
    main()