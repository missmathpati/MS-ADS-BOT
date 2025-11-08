import os
import json
import re
import uuid
from pathlib import Path
import numpy as np
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Page configuration
st.set_page_config(
    page_title="AskADS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for animations and styling
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Animated header */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glowMaroon {
        0%, 100% {
            box-shadow: 0 0 5px rgba(128, 0, 32, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(128, 0, 32, 0.8), 0 0 30px rgba(128, 0, 32, 0.6);
        }
    }
    
    /* Chat container for alignment */
    .stChatMessageContainer {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        max-width: 56rem; /* max-w-4xl equivalent - matches input bar */
        margin: 0 auto;
        padding: 0 1rem;
        width: 100%;
        box-sizing: border-box;
    }
    
    /* Ensure proper alignment on all screen sizes */
    @media (max-width: 1024px) {
        .stChatMessageContainer {
            padding: 0 1rem;
        }
    }
    
    @media (max-width: 768px) {
        .stChatMessageContainer {
            padding: 0 0.75rem;
        }
    }
    
    @media (max-width: 480px) {
        .stChatMessageContainer {
            padding: 0 0.5rem;
        }
    }
    
    @media (max-width: 360px) {
        .stChatMessageContainer {
            padding: 0 0.375rem;
        }
    }
    
    /* Message alignment - User messages left */
    [data-testid="stChatMessage"][aria-label*="user"] {
        margin-left: 0 !important;
        margin-right: auto !important;
        max-width: 70% !important;
        align-self: flex-start !important;
    }
    
    /* Message alignment - Assistant messages right */
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        margin-left: auto !important;
        margin-right: 0 !important;
        max-width: 70% !important;
        align-self: flex-end !important;
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
    }
    
    /* Header styling - Maroon gradient theme */
    .header-container {
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(128, 0, 32, 0.3);
        animation: fadeInDown 0.8s ease-out;
        border: 3px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5);
        animation: pulse 2s ease-in-out infinite;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.75rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Bot icon animation */
    .bot-icon {
        font-size: 4.5rem;
        text-align: center;
        animation: pulse 2s ease-in-out infinite;
        display: block;
        margin-bottom: 1rem;
        filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3));
        position: relative;
        z-index: 1;
    }
    
    /* Chat message animations */
    .stChatMessage {
        animation: slideIn 0.5s ease-out;
        margin-bottom: 1.5rem;
    }
    
    /* Message bubbles */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* User message styling - Left aligned */
    [data-testid="stChatMessage"][aria-label*="user"] {
        background: linear-gradient(135deg, rgba(128, 0, 32, 0.08) 0%, rgba(160, 0, 48, 0.05) 100%);
        border-left: 4px solid #800020;
        text-align: left;
        position: relative;
    }
    
    [data-testid="stChatMessage"][aria-label*="user"]::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 15px 0 0 15px;
    }
    
    /* Assistant message styling - Right aligned */
    [data-testid="stChatMessage"][aria-label*="assistant"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        border-right: 4px solid #800020;
        text-align: right;
        position: relative;
    }
    
    [data-testid="stChatMessage"][aria-label*="assistant"]::after {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 0 15px 15px 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        color: white;
        border: 2px solid white;
        border-radius: 12px;
        padding: 0.6rem 1.8rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(128, 0, 32, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(128, 0, 32, 0.5);
        background: linear-gradient(135deg, #a00030 0%, #c00040 50%, #e00050 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f8f8 100%);
        border-right: 4px solid #800020;
    }
    
    /* Input bar container - sticky positioning */
    [data-testid="stChatInputContainer"] {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 100 !important;
        background: linear-gradient(to top, rgba(255, 255, 255, 0.98) 0%, rgba(255, 255, 255, 0.95) 100%);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1rem 1rem !important;
        padding-bottom: calc(1rem + env(safe-area-inset-bottom)) !important;
        margin-top: 1rem !important;
        border-top: 1px solid rgba(128, 0, 32, 0.15);
        box-shadow: 0 -4px 20px rgba(128, 0, 32, 0.08);
        width: 100% !important;
        box-sizing: border-box !important;
    }
    
    /* Input wrapper - aligned with chat container */
    [data-testid="stChatInput"] {
        max-width: 56rem !important; /* max-w-4xl - matches chat container */
        width: 100% !important;
        margin: 0 auto !important;
        padding: 0 !important;
        border-radius: 1rem !important; /* rounded-2xl */
        border: 2px solid rgba(128, 0, 32, 0.25) !important;
        background: linear-gradient(135deg, #ffffff 0%, #fafafa 100%) !important;
        box-shadow: 0 4px 20px rgba(128, 0, 32, 0.12) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: visible !important;
        box-sizing: border-box !important;
    }
    
    /* Gradient border effect on focus - cleaner approach */
    [data-testid="stChatInput"]:focus-within {
        border-color: #800020 !important;
        border-width: 2px !important;
        box-shadow: 0 6px 30px rgba(128, 0, 32, 0.25),
                    0 0 0 4px rgba(128, 0, 32, 0.1) !important;
        transform: translateY(-2px);
    }
    
    /* Textarea styling */
    [data-testid="stChatInput"] textarea {
        color: #1a1a1a !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        padding: 0.75rem 1rem !important;
        padding-right: 3rem !important;
        min-height: 48px !important;
        max-height: 140px !important;
        resize: none !important;
        background: transparent !important;
        border: none !important;
        outline: none !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif !important;
        width: 100% !important;
        box-sizing: border-box !important;
        -webkit-appearance: none !important;
        -moz-appearance: none !important;
        appearance: none !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: rgba(128, 0, 32, 0.5) !important;
        font-style: italic;
    }
    
    /* Disabled state */
    [data-testid="stChatInput"]:has(textarea:disabled) {
        opacity: 0.6 !important;
        cursor: not-allowed !important;
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%) !important;
    }
    
    [data-testid="stChatInput"] textarea:disabled {
        cursor: not-allowed !important;
    }
    
    /* Send button styling */
    [data-testid="stChatInput"] button {
        position: absolute !important;
        right: 0.5rem !important;
        bottom: 0.5rem !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        border-radius: 0.75rem !important;
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%) !important;
        border: none !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 8px rgba(128, 0, 32, 0.3) !important;
        z-index: 10 !important;
    }
    
    [data-testid="stChatInput"] button:hover:not(:disabled) {
        background: linear-gradient(135deg, #a00030 0%, #c00040 50%, #e00050 100%) !important;
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 4px 12px rgba(128, 0, 32, 0.4) !important;
    }
    
    [data-testid="stChatInput"] button:active:not(:disabled) {
        transform: translateY(0) scale(0.98) !important;
    }
    
    [data-testid="stChatInput"] button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
        background: linear-gradient(135deg, #cccccc 0%, #bbbbbb 100%) !important;
    }
    
    /* Loading/thinking state - spinner animation */
    [data-testid="stChatInput"] button:disabled::after {
        content: '';
        position: absolute;
        width: 1rem;
        height: 1rem;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-top-color: white;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    /* Tablet responsive adjustments (768px - 1024px) */
    @media (max-width: 1024px) and (min-width: 769px) {
        [data-testid="stChatInputContainer"] {
            padding: 0.875rem 1rem !important;
            padding-bottom: calc(0.875rem + env(safe-area-inset-bottom)) !important;
        }
        
    [data-testid="stChatInput"] {
            max-width: 100% !important;
            border-radius: 0.875rem !important;
        }
        
        [data-testid="stChatInput"] textarea {
            padding: 0.7rem 0.9rem !important;
            padding-right: 2.8rem !important;
        }
        
        [data-testid="stChatInput"] button {
            width: 2.4rem !important;
            height: 2.4rem !important;
            right: 0.45rem !important;
            bottom: 0.45rem !important;
        }
    }
    
    /* Mobile responsive adjustments (up to 768px) */
    @media (max-width: 768px) {
        [data-testid="stChatInputContainer"] {
            padding: 0.75rem 0.75rem !important;
            padding-bottom: calc(0.75rem + env(safe-area-inset-bottom)) !important;
            margin-top: 0.75rem !important;
        }
        
        [data-testid="stChatInput"] {
            max-width: 100% !important;
            padding: 0 !important;
            border-radius: 0.875rem !important;
            border-width: 1.5px !important;
        }
        
        [data-testid="stChatInput"] textarea {
            font-size: 16px !important; /* Prevents zoom on iOS */
            padding: 0.625rem 0.875rem !important;
            padding-right: 2.75rem !important;
            min-height: 44px !important; /* Better touch target */
        }
        
        [data-testid="stChatInput"] button {
            width: 2.25rem !important;
            height: 2.25rem !important;
            right: 0.5rem !important;
            bottom: 0.5rem !important;
            min-width: 2.25rem !important;
            min-height: 2.25rem !important;
        }
    }
    
    /* Small mobile devices (up to 480px) */
    @media (max-width: 480px) {
        [data-testid="stChatInputContainer"] {
            padding: 0.625rem 0.5rem !important;
            padding-bottom: calc(0.625rem + env(safe-area-inset-bottom)) !important;
        }
        
        [data-testid="stChatInput"] {
            border-radius: 0.75rem !important;
        }
        
        [data-testid="stChatInput"] textarea {
            font-size: 16px !important;
            padding: 0.5rem 0.75rem !important;
            padding-right: 2.5rem !important;
            min-height: 44px !important;
        }
        
        [data-testid="stChatInput"] button {
            width: 2rem !important;
            height: 2rem !important;
            right: 0.5rem !important;
            bottom: 0.5rem !important;
        }
    }
    
    /* Extra small devices (up to 360px) */
    @media (max-width: 360px) {
        [data-testid="stChatInputContainer"] {
            padding: 0.5rem 0.375rem !important;
            padding-bottom: calc(0.5rem + env(safe-area-inset-bottom)) !important;
        }
        
        [data-testid="stChatInput"] textarea {
            padding: 0.5rem 0.625rem !important;
            padding-right: 2.25rem !important;
            font-size: 16px !important;
        }
        
        [data-testid="stChatInput"] button {
            width: 1.875rem !important;
            height: 1.875rem !important;
            right: 0.375rem !important;
            bottom: 0.375rem !important;
        }
    }
    
    /* Landscape orientation on mobile */
    @media (max-width: 768px) and (orientation: landscape) {
        [data-testid="stChatInputContainer"] {
            padding: 0.5rem 0.75rem !important;
            padding-bottom: calc(0.5rem + env(safe-area-inset-bottom)) !important;
        }
        
        [data-testid="stChatInput"] textarea {
            min-height: 40px !important;
            max-height: 100px !important;
        }
    }
    
    /* Large screens (above 1024px) - ensure proper centering */
    @media (min-width: 1025px) {
        [data-testid="stChatInputContainer"] {
            padding-left: max(1rem, calc((100% - 56rem) / 2)) !important;
            padding-right: max(1rem, calc((100% - 56rem) / 2)) !important;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        [data-testid="stChatInputContainer"] {
            background: linear-gradient(to top, rgba(26, 26, 26, 0.98) 0%, rgba(26, 26, 26, 0.95) 100%);
            border-top-color: rgba(128, 0, 32, 0.3);
        }
        
        [data-testid="stChatInput"] {
            background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%) !important;
            border-color: rgba(128, 0, 32, 0.3) !important;
        }
        
        [data-testid="stChatInput"] textarea {
            color: #e5e5e5 !important;
        }
        
        [data-testid="stChatInput"] textarea::placeholder {
            color: rgba(128, 0, 32, 0.6) !important;
        }
        
        [data-testid="stChatInput"]:has(textarea:disabled) {
            background: linear-gradient(135deg, #1a1a1a 0%, #151515 100%) !important;
        }
    }
    
    /* Ensure no horizontal scrolling */
    body {
        overflow-x: hidden;
    }
    
    /* Prevent text selection issues on mobile */
    [data-testid="stChatInput"] textarea {
        -webkit-tap-highlight-color: transparent;
        -webkit-touch-callout: none;
    }
    
    /* Touch-friendly button sizing on mobile */
    @media (hover: none) and (pointer: coarse) {
        [data-testid="stChatInput"] button {
            min-width: 44px !important;
            min-height: 44px !important;
        }
    }
    
    /* Ensure proper width constraints */
    [data-testid="stChatInputContainer"],
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] textarea {
        max-width: 100% !important;
    }
    
    /* Accessibility improvements */
    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
    }
    
    [data-testid="stChatInput"] button:focus-visible {
        outline: 3px solid rgba(128, 0, 32, 0.5) !important;
        outline-offset: 2px !important;
    }
    
    /* Smooth auto-grow for textarea */
    [data-testid="stChatInput"] textarea {
        overflow-y: auto !important;
        scrollbar-width: thin;
        scrollbar-color: rgba(128, 0, 32, 0.3) transparent;
    }
    
    [data-testid="stChatInput"] textarea::-webkit-scrollbar {
        width: 6px;
    }
    
    [data-testid="stChatInput"] textarea::-webkit-scrollbar-track {
        background: transparent;
    }
    
    [data-testid="stChatInput"] textarea::-webkit-scrollbar-thumb {
        background: rgba(128, 0, 32, 0.3);
        border-radius: 3px;
    }
    
    [data-testid="stChatInput"] textarea::-webkit-scrollbar-thumb:hover {
        background: rgba(128, 0, 32, 0.5);
    }
    
    /* Source cards */
    .source-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 5px solid #800020;
        animation: slideIn 0.5s ease-out;
        transition: all 0.3s ease;
        box-shadow: 0 3px 12px rgba(128, 0, 32, 0.15);
    }
    
    .source-card:hover {
        transform: translateX(8px);
        box-shadow: 0 6px 20px rgba(128, 0, 32, 0.25);
        border-left-width: 6px;
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(128, 0, 32, 0.15);
        border: 2px solid rgba(128, 0, 32, 0.1);
    }
    
    /* Loading spinner */
    .spinner {
        border: 4px solid rgba(128, 0, 32, 0.1);
        border-top: 4px solid #800020;
        border-radius: 50%;
        width: 35px;
        height: 35px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #800020;
        margin-top: 3rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        border-top: 4px solid #800020;
        border-radius: 15px;
        box-shadow: 0 -4px 15px rgba(128, 0, 32, 0.1);
        position: relative;
    }
    
    .footer::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 15px 15px 0 0;
    }
    
    /* Answer container */
    .answer-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f8f8 100%);
        padding: 2rem 2.5rem;
        border-radius: 15px;
        border-left: 6px solid #800020;
        box-shadow: 0 6px 20px rgba(128, 0, 32, 0.15);
        margin: 1rem 0;
        line-height: 1.75;
        text-align: left;
        position: relative;
        color: #1a1a1a;
        font-size: 1rem;
        max-width: 100%;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .answer-container::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 6px;
        background: linear-gradient(180deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 15px 0 0 15px;
    }
    
    /* Typography improvements */
    .answer-container p {
        margin: 0 0 1rem 0;
        padding: 0;
        text-align: left;
    }
    
    .answer-container p:last-child {
        margin-bottom: 0;
    }
    
    /* Numbered lists - professional formatting */
    .answer-container ol {
        margin: 1rem 0;
        padding-left: 2rem;
        list-style-position: outside;
        counter-reset: item;
    }
    
    .answer-container ol > li {
        margin: 0.75rem 0;
        padding-left: 0.5rem;
        line-height: 1.7;
        text-align: left;
    }
    
    .answer-container ol > li::marker {
        font-weight: 600;
        color: #800020;
    }
    
    /* Unordered lists (bullets) */
    .answer-container ul {
        margin: 0.75rem 0;
        padding-left: 2rem;
        list-style-type: disc;
        list-style-position: outside;
    }
    
    .answer-container ul > li {
        margin: 0.5rem 0;
        padding-left: 0.5rem;
        line-height: 1.7;
        text-align: left;
    }
    
    /* Nested lists - proper indentation */
    .answer-container ul ul,
    .answer-container ol ol,
    .answer-container ul ol,
    .answer-container ol ul {
        margin: 0.5rem 0;
        padding-left: 1.75rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .answer-container ul ul {
        list-style-type: circle;
    }
    
    .answer-container ul ul ul {
        list-style-type: square;
    }
    
    /* Bold text in lists */
    .answer-container li strong {
        color: #800020;
        font-weight: 600;
    }
    
    /* Headings in answers */
    .answer-container h1,
    .answer-container h2,
    .answer-container h3,
    .answer-container h4 {
        margin: 1.5rem 0 1rem 0;
        color: #800020;
        font-weight: 600;
        text-align: left;
        line-height: 1.4;
    }
    
    .answer-container h1 {
        font-size: 1.5rem;
        border-bottom: 2px solid rgba(128, 0, 32, 0.2);
        padding-bottom: 0.5rem;
    }
    
    .answer-container h2 {
        font-size: 1.25rem;
    }
    
    .answer-container h3 {
        font-size: 1.1rem;
    }
    
    /* Links in answers */
    .answer-container a {
        color: #800020;
        text-decoration: none;
        border-bottom: 1px solid rgba(128, 0, 32, 0.3);
        transition: all 0.2s ease;
    }
    
    .answer-container a:hover {
        color: #a00030;
        border-bottom-color: #a00030;
    }
    
    /* Citations in text [1], [2], etc. */
    .answer-container :is(p, li) {
        text-align: left;
    }
    
    /* Better spacing for paragraphs */
    .answer-container > *:first-child {
        margin-top: 0;
    }
    
    .answer-container > *:last-child {
        margin-bottom: 0;
    }
    
    /* Code blocks if any */
    .answer-container code {
        background: rgba(128, 0, 32, 0.1);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
        color: #800020;
    }
    
    .answer-container pre {
        background: rgba(128, 0, 32, 0.05);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #800020;
        overflow-x: auto;
        margin: 1rem 0;
    }
    
    .answer-container pre code {
        background: transparent;
        padding: 0;
    }
    
    /* Blockquotes */
    .answer-container blockquote {
        margin: 1rem 0;
        padding: 1rem 1.5rem;
        border-left: 4px solid #800020;
        background: rgba(128, 0, 32, 0.05);
        border-radius: 0 8px 8px 0;
        font-style: italic;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .answer-container {
            padding: 1.5rem 1.75rem;
            font-size: 0.95rem;
        }
        
        .answer-container ol,
        .answer-container ul {
            padding-left: 1.5rem;
        }
        
        .answer-container ul ul,
        .answer-container ol ol {
            padding-left: 1.25rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .answer-container {
            background: linear-gradient(135deg, #2a2a2a 0%, #1f1f1f 100%);
            color: #e5e5e5;
        }
        
        .answer-container li strong {
            color: #c00040;
        }
        
        .answer-container h1,
        .answer-container h2,
        .answer-container h3,
        .answer-container h4 {
            color: #c00040;
        }
        
        .answer-container a {
            color: #c00040;
        }
        
        .answer-container code {
            background: rgba(192, 0, 64, 0.2);
            color: #c00040;
        }
    }
    
    /* Citation section */
    .citation-section {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 2px solid rgba(128, 0, 32, 0.2);
        background: linear-gradient(135deg, rgba(128, 0, 32, 0.05) 0%, transparent 100%);
        border-radius: 10px;
        padding: 1rem;
    }
    
    .citation-header {
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .citation-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 10px;
        border-right: 4px solid #800020;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(128, 0, 32, 0.1);
        justify-content: flex-end;
        position: relative;
    }
    
    .citation-item::after {
        content: '';
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 0 10px 10px 0;
    }
    
    .citation-item:hover {
        transform: translateX(-5px);
        box-shadow: 0 4px 12px rgba(128, 0, 32, 0.2);
    }
    
    .citation-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        color: white;
        border-radius: 50%;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(128, 0, 32, 0.3);
    }
    
    /* Citation styling */
    .citation-link {
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .citation-link:hover {
        opacity: 0.8;
        text-decoration: underline;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f8f8f8;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #a00030 0%, #c00040 50%, #e00050 100%);
    }
</style>
""", unsafe_allow_html=True)

# Paths
ART_DIR = Path("rag_index")
CHROMA_DIR = ART_DIR / "chroma_db"
META_PATH = ART_DIR / "meta.jsonl"

# Model configuration
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANKER_NAME = "BAAI/bge-reranker-base"

# Priority hints for boosting
EDU_PRIORITY_HINTS = [
    "/education/masters-programs/ms-in-applied-data-science",
    "/education/masters-programs",
    "/education/",
]

LOW_PRIORITY_HINTS = [
    "/news-events/news/",
    "/news-events/events/",
    "/news-events/insights/",
    "/research/",
    "/people/",
]

INTENT_KEYWORDS = {
    "admissions": ["admission", "admissions", "apply", "application", "requirements", "prereq", "prerequisite", "deadline", "GRE", "TOEFL", "IELTS", "resume", "statement", "letters"],
    "curriculum": ["core", "course", "courses", "curriculum", "credit", "unit", "track", "specialization", "elective"],
    "capstone": ["capstone", "project", "showcase", "practicum"],
}

# PII regex patterns
PII_EMAIL = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PII_PHONE = re.compile(r'\b(?:\+?\d{1,2}\s*)?(?:\(?\d{3}\)?[\s.-]*)?\d{3}[\s.-]?\d{4}\b')

# System prompt
SYSTEM_PROMPT = """You are a helpful assistant for the University of Chicago MS in Applied Data Science.
Answer ONLY from the provided context. Prefer content from the Education section and the program page.
If top results are news, events, insights, research, or people pages, treat them as lower priority unless the question asks for them.
If the required information is not present in the provided context, say you don't know and suggest checking the official MS-ADS page.
Keep answers specific and concise. Always include bracketed citations like [1], [2] with URLs.
Redact personal emails/phones if present in context.
"""


# Optional reranker (lazy loading)
_reranker = None
def get_reranker(name=RERANKER_NAME):
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(name)
        except Exception as e:
            _reranker = False
            st.sidebar.warning(f"Reranker could not be loaded: {e}")
    return _reranker if _reranker else None


# E5 Embedder class for ChromaDB
class E5Embedder(embedding_functions.EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        v = self.model.encode(["passage: " + x for x in input],
                             normalize_embeddings=True, convert_to_numpy=True)
        return v.tolist()


@st.cache_resource(show_spinner="Loading RAG index and models...")
def load_chroma_and_meta(chroma_dir: Path, embed_model_name: str):
    """Load ChromaDB collection, metadata, embedding model, and TF-IDF vectorizer."""
    if not chroma_dir.exists():
        raise FileNotFoundError(
            f"Missing ChromaDB directory. "
            f"Ensure {chroma_dir} exists in the rag_index/ folder."
        )
    
    # Load embedding model
    model = SentenceTransformer(embed_model_name)
    
    # Setup ChromaDB
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Try new API (ChromaDB 0.4.0+)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        # Access default tenant and database
        try:
            db = client.get_or_create_database("default_database")
            collection = db.get_or_create_collection(
                name="msads_e5",
                metadata={"hnsw:space": "cosine"},
                embedding_function=E5Embedder(model)
            )
        except AttributeError:
            # Fallback: try direct collection access (older API or different version)
            collection = client.get_or_create_collection(
                name="msads_e5",
                metadata={"hnsw:space": "cosine"},
                embedding_function=E5Embedder(model)
            )
    except Exception as e:
        # If that fails, try with explicit tenant
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                tenant="default_tenant",
                database="default_database"
            )
            collection = client.get_or_create_collection(
                name="msads_e5",
                metadata={"hnsw:space": "cosine"},
                embedding_function=E5Embedder(model)
            )
        except Exception:
            # Last resort: try without tenant/database params
            client = chromadb.PersistentClient(path=str(chroma_dir))
            collection = client.get_or_create_collection(
                name="msads_e5",
                metadata={"hnsw:space": "cosine"},
                embedding_function=E5Embedder(model)
            )
    
    # Load metadata if available
    META = []
    id_to_meta = {}
    id_order = []
    
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta_list = [json.loads(line) for line in f]
        
        for i, m in enumerate(meta_list):
            doc_id = m.get("id", str(uuid.uuid4()))
            m["text"] = m.get("text", "")
            m["_id"] = doc_id
            META.append(m)
            id_to_meta[doc_id] = m
            id_order.append(doc_id)
    else:
        # Build from ChromaDB if meta.jsonl doesn't exist
        all_data = collection.get(include=["metadatas", "documents", "ids"])
        for i, doc_id in enumerate(all_data["ids"]):
            meta = all_data["metadatas"][i] if all_data["metadatas"] else {}
            meta["text"] = all_data["documents"][i] if all_data["documents"] else ""
            meta["_id"] = doc_id
            META.append(meta)
            id_to_meta[doc_id] = meta
            id_order.append(doc_id)
    
    # Build TF-IDF vectorizer
    DOC_TEXTS = [
        ((m.get("title", "") + " " + m.get("section", "") + " " + 
          m.get("url", "") + " " + m.get("text", "")).strip())
        for m in META
    ]
    tfidf = TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1, 2))
    X = tfidf.fit_transform(DOC_TEXTS)
    
    return collection, META, id_to_meta, id_order, model, tfidf, X


def ann_dense_chroma(query: str, collection, model, topn: int):
    """Dense retrieval using ChromaDB."""
    qvec = model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    res = collection.query(
        query_embeddings=[qvec],
        n_results=topn,
        include=["metadatas", "documents", "distances"]
    )
    ids = res["ids"][0]
    sims = [1.0 - d for d in res["distances"][0]]  # cosine similarity
    return ids, sims


def bm25_like_indices(query: str, tfidf, X, id_order, topn: int):
    """Sparse retrieval using TF-IDF."""
    qv = tfidf.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:topn]
    return [id_order[i] for i in idx], sims[idx]


def mmr_select(q_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: list[str], k: int = 6, lambda_: float = 0.55):
    """Maximal Marginal Relevance for diversity."""
    selected, pool = [], list(range(len(cand_ids)))
    sim_q = (cand_vecs @ q_vec.reshape(-1, 1)).ravel()
    while len(selected) < min(k, len(pool)):
        if not selected:
            i = int(np.argmax(sim_q[pool]))
            selected.append(pool[i])
            pool.pop(i)
            continue
        sel_vecs = cand_vecs[selected]
        cand_idx = np.array(pool)
        diversity = (cand_vecs[cand_idx] @ sel_vecs.T).max(axis=1)
        scores = lambda_ * sim_q[cand_idx] - (1 - lambda_) * diversity
        i = int(np.argmax(scores))
        selected.append(cand_idx[i])
        pool.remove(cand_idx[i])
    return [cand_ids[i] for i in selected]


def _boost_score(url: str, section: str, base: float) -> float:
    """Boost scores based on URL patterns and section."""
    b = base
    if any(h in url for h in EDU_PRIORITY_HINTS):
        b += 0.20
    if section == "education":
        b += 0.10
    if any(h in url for h in LOW_PRIORITY_HINTS):
        b -= 0.20
    return b


def _intent(query: str):
    """Detect query intent from keywords."""
    ql = query.lower()
    for k, toks in INTENT_KEYWORDS.items():
        if any(t in ql for t in toks):
            return k
    return None


def retrieve_hybrid(query: str, collection, id_to_meta, id_order, model, 
                    tfidf, X, k: int, shortlist: int, use_reranker: bool):
    """Hybrid retrieval combining dense (ChromaDB) and sparse (TF-IDF) methods with MMR."""
    # Dense retrieval (ChromaDB)
    dense_ids, _ = ann_dense_chroma(query, collection, model, topn=shortlist)
    
    # Sparse retrieval (TF-IDF)
    sparse_ids, _ = bm25_like_indices(query, tfidf, X, id_order, topn=shortlist)
    
    # Reciprocal Rank Fusion
    def rrf(id_lists, c=60):
        score = {}
        for lst in id_lists:
            for r, did in enumerate(lst):
                score[did] = score.get(did, 0.0) + 1.0 / (c + r + 1)
        return score
    
    fused = rrf([list(dense_ids), list(sparse_ids)])
    
    # Apply boosts based on intent and URL patterns
    want = _intent(query)
    items = []
    for did, base in fused.items():
        m = id_to_meta.get(did, {})
        url = m.get("url", "") or ""
        sec = m.get("section", "") or ""
        if want not in ("capstone",) and ("/news-events/" in url or "/research/" in url):
            base -= 0.25
        boosted = _boost_score(url, sec, base)
        items.append((did, boosted))
    
    items.sort(key=lambda x: -x[1])
    
    # MMR on a larger pool
    pool = [did for did, _ in items[:max(k, 30)]]
    cand_texts = [id_to_meta[did].get("text", "") for did in pool]
    cand_vecs = model.encode(["passage: " + t for t in cand_texts], normalize_embeddings=True, convert_to_numpy=True)
    q_vec = model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True).ravel().astype(np.float32)
    mmr_ids = mmr_select(q_vec, cand_vecs, pool, k=max(k, 10), lambda_=0.55)
    
    hits = [dict(id_to_meta[did]) | {"_id": did} for did in mmr_ids]
    
    # Apply reranker if enabled
    if use_reranker:
        rr = get_reranker()
        if rr:
            pairs = [(query, h.get("text", "")) for h in hits]
            scores = rr.predict(pairs)
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            hits.sort(key=lambda x: -x["rerank_score"])
    
    return hits[:k]


def scrub(text: str) -> str:
    """Redact PII from text."""
    text = PII_EMAIL.sub("[redacted-email]", text)
    text = PII_PHONE.sub("[redacted-phone]", text)
    return text


_SENT_SPLIT = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')


def compress_text_for_query(text: str, query: str, model, top_sentences: int = 8):
    """Compress text by selecting most relevant sentences."""
    sents = [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]
    if not sents:
        return text or ""
    qv = model.encode(["query: " + query], normalize_embeddings=True, convert_to_numpy=True)
    sv = model.encode(["passage: " + s for s in sents], normalize_embeddings=True, convert_to_numpy=True)
    sims = (sv @ qv.T).ravel()
    keep = np.argsort(-sims)[:min(top_sentences, len(sents))]
    keep.sort()
    return " ".join(sents[i] for i in keep)


def long_context_reorder(hits):
    """Reorder hits for better context placement."""
    if len(hits) <= 2:
        return hits
    L, R, out = 0, len(hits) - 1, []
    while L <= R:
        out.append(hits[L])
        L += 1
        if L <= R:
            out.append(hits[R])
            R -= 1
    return out


def build_context(hits, query: str, model):
    """Build context string from retrieved hits with compression."""
    hits = long_context_reorder(hits)
    blocks = []
    for i, h in enumerate(hits, 1):
        title = (h.get('title', '') or '').strip()
        url = (h.get('url', '') or '').strip()
        sect = (h.get('section', '') or '').strip()
        txt = compress_text_for_query(h.get('text', '') or '', query, model, top_sentences=8)
        txt = scrub(txt)
        blocks.append(f"[{i}] {title} • {sect} | {url}\n{txt}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, hits, oai_client, model, temperature: float = 0.2, 
                    model_name: str = "gpt-4o-mini"):
    """Generate answer using OpenAI API."""
    context = build_context(hits, query, model)
    user_prompt = f"Question: {query}\n\nUse the context to answer with bracket citations.\n\nContext:\n{context}"
    
    try:
        resp = oai_client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )
        return resp.choices[0].message.content.strip(), context
    except Exception as e:
        return f"Error generating answer: {str(e)}", None


# Sidebar configuration with maroon gradient styling
st.sidebar.markdown("""
<div style='text-align: center; padding: 1.25rem; background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%); border-radius: 12px; margin-bottom: 1rem; border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 4px 15px rgba(128,0,32,0.3);'>
    <h2 style='color: white; margin: 0; font-weight: 700;'>⚙️ Configuration</h2>
</div>
""", unsafe_allow_html=True)

# Check for OpenAI API key
# Initialize session state for API key if not exists
if "api_key" not in st.session_state:
    env_key = os.getenv("OPENAI_API_KEY", "")
    st.session_state.api_key = env_key

# Get API key from environment first, then check session state
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Check session state
    api_key = st.session_state.api_key if st.session_state.api_key else ""
    
    # Use session state to persist the API key
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        value=api_key,
        key="openai_api_key_input",
        help="Enter your OpenAI API key or set OPENAI_API_KEY environment variable"
    )
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
else:
    # If API key is in environment, use it and update session state
    st.session_state.api_key = api_key

if not api_key:
    st.sidebar.error("⚠️ Please provide an OpenAI API key to use this app.")
    st.stop()

try:
    from openai import OpenAI
    if api_key:
        oai = OpenAI(api_key=api_key)
    else:
        oai = None
except Exception as e:
    st.sidebar.error(f"OpenAI client initialization error: {e}")
    oai = None
    st.stop()

# Model settings with maroon gradient styling
st.sidebar.markdown("""
<div style='background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%); padding: 0.9rem; border-radius: 12px; margin: 1rem 0; border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 3px 12px rgba(128,0,32,0.25);'>
    <h3 style='color: white; margin: 0; text-align: center; font-weight: 700;'>🎛️ Model Settings</h3>
</div>
""", unsafe_allow_html=True)
USE_RERANKER = st.sidebar.toggle("Use reranker (CrossEncoder)", value=True)
TEMPERATURE = st.sidebar.slider("Generation temperature", 0.0, 1.0, 0.2, 0.1)
TOP_K = st.sidebar.slider("k (final retrieved)", 3, 12, 6, 1)
SHORTLIST = st.sidebar.slider("Shortlist (pre-rerank)", 10, 100, 60, 5)

# Load ChromaDB and models
try:
    collection, META, id_to_meta, id_order, embed_model, tfidf, X = load_chroma_and_meta(
        CHROMA_DIR, EMBED_MODEL_NAME
    )
    doc_count = collection.count()
    st.sidebar.markdown(f"""
    <div style='background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%); color: white; padding: 1.25rem; border-radius: 12px; text-align: center; margin-top: 1rem; border: 2px solid rgba(255,255,255,0.2); box-shadow: 0 4px 15px rgba(128,0,32,0.3);'>
        <strong style='font-size: 1.1rem;'>✅ Loaded {doc_count} documents</strong>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    st.error(f"❌ Could not load artifacts: {e}")
    st.info(
        "Make sure you have:\n"
        f"- {CHROMA_DIR} (ChromaDB directory)\n"
        f"- {META_PATH} (optional, metadata file)\n"
        "in the `rag_index/` folder."
    )
    st.stop()

# Main UI with animated header
st.markdown("""
<div class="header-container">
    <div class="bot-icon">🎓</div>
    <h1 class="header-title">AskADS</h1>
    <p class="header-subtitle">Our smart assistant for the MS-ADS program</p>
</div>
""", unsafe_allow_html=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(f"""
            <div class="answer-container">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            srcs = (message.get("sources") or [])[:5]
            if srcs:
                citations_html = '<div class="citation-section">'
                citations_html += '<div class="citation-header">📚 References</div>'
                for i, s in enumerate(srcs, 1):
                    title = s.get('title') or '(no title)'
                    url = s.get('url') or ''
                    citations_html += '<div class="citation-item">'
                    citations_html += f'<span class="citation-number">{i}</span>'
                    citations_html += f'<a class="citation-link" href="{url}" target="_blank">{title}</a>'
                    citations_html += '</div>'
                citations_html += '</div>'
                st.markdown(citations_html, unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input with custom styling
prompt = st.chat_input("Ask AskADS anything about the MS-ADS program...")

if prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Retrieve relevant documents
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching knowledge base..."):
            hits = retrieve_hybrid(
                prompt,
                collection,
                id_to_meta,
                id_order,
                embed_model,
                tfidf,
                X,
                k=TOP_K,
                shortlist=SHORTLIST,
                use_reranker=USE_RERANKER
            )
            
            # Query augmentation if no education pages found
            if not any(("/education/" in (h.get("url", "") or "") or 
                       "ms-in-applied-data-science" in (h.get("url", "") or "")) 
                      for h in hits):
                aug = prompt + ' program site education admissions curriculum "MS in Applied Data Science"'
                hits = retrieve_hybrid(
                    aug,
                    collection,
                    id_to_meta,
                    id_order,
                    embed_model,
                    tfidf,
                    X,
                    k=TOP_K,
                    shortlist=SHORTLIST,
                    use_reranker=USE_RERANKER
                )
        
        with st.spinner("🤖 Generating answer..."):
            ans, _ctx = generate_answer(prompt, hits, oai, embed_model, temperature=TEMPERATURE)
            st.markdown(f"""
            <div class="answer-container">
                {ans}
            </div>
            """, unsafe_allow_html=True)
        
        # Subtle citations (top 5)
        top5 = hits[:5]
        if top5:
            citations_html = '<div class="citation-section">'
            citations_html += '<div class="citation-header">📚 References</div>'
            for i, h in enumerate(top5, 1):
                title = (h.get('title') or '(no title)')
                url = h.get('url') or ''
                citations_html += '<div class="citation-item">'
                citations_html += f'<span class="citation-number">{i}</span>'
                citations_html += f'<a class="citation-link" href="{url}" target="_blank">{title}</a>'
                citations_html += '</div>'
            citations_html += '</div>'
            st.markdown(citations_html, unsafe_allow_html=True)
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": ans,
        "sources": [{"title": h.get("title", ""), "url": h.get("url", ""), 
                     "section": h.get("section", "")} for h in hits[:5]]
    })

# Sidebar info with styling
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: white; padding: 1.25rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(128,0,32,0.15); border: 2px solid rgba(128,0,32,0.1);'>
    <h3 style='background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-top: 0; font-weight: 700;'>📊 Index Info</h3>
    <p style='color: #333;'><strong>Documents:</strong> {}</p>
    <p style='color: #333;'><strong>Database:</strong> ChromaDB</p>
    <p style='color: #333;'><strong>Collection:</strong> msads_e5</p>
</div>
""".format(collection.count()), unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p style='font-size: 1.1rem;'><strong style='background: linear-gradient(135deg, #800020 0%, #a00030 50%, #c00040 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;'>🎓 AskADS</strong> | Powered by GEN AI Group 7</p>
    <p style='font-size: 0.9rem; margin-top: 0.5rem; color: #666;'>University of Chicago MS in Applied Data Science</p>
</div>
""", unsafe_allow_html=True)

