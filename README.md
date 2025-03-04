# VisionaryX--A-Multi-Modal-LLM- #

# Overview #

VisionaryX- a multi-modal AI assistant is a Streamlit-based application that combines text, image, and document processing into a single platform.

# Features #

 * Text Generation - Generate human-like responses using Google’s Gemini 1.5 Flash.
 * Text-to-Image - Convert text prompts into images using Hugging Face models.
 * Image-to-Text - Extract text from images using Gemini 1.5 Flash.
 * PDF-to-Text - Extract and process text from PDFs efficiently using FAISS.

# Local Development #
  ### 1. Set up API Keys:
    
     **Create a .env file and add:**
     '''bash
       GEMINI_API_KEY=your_google_api_key  
       HUGGINGFACE_API_KEY=your_huggingface_api_key
    
  ### 2. Run the Application:
     
     '''bash
      streamlit run application.py

# Project Structure #
 * application.py - Main application script.
 * requirements.txt - Python packages required for working of the app.
 * README.md - Project documentation.


# Dependencies #

 * Streamlit
 * langchain
 * google.generativeai
 * python-dotenv
 * huggingface_hub
 * pypdf

# Acknowledgments #
 * Google Gemini: For providing the underlying language model
 * Streamlit: For the user interface framework
   




           





