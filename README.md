# VisionaryX--A-Multi-Modal-LLM- #

# Overview #

VisionaryX- a multi-modal AI assistant is a Streamlit-based application that combines text, image, and document processing into a single platform.

# Features #

 * Text Generation - Generate human-like responses using Google’s Gemini 1.5 Flash.
 * Text-to-Image - Convert text prompts into images using Hugging Face models.
 * Image-to-Text (OCR) - Extract text from images using Gemini 1.5 Flash.
 * PDF-to-Text - Extract and process text from PDFs efficiently using FAISS.

# Installation #
   Follow these informations to set up and run this project on your local machine.
   Note: This project requires Python 3.10 
     1. # Clone the Repository:
         ##
         git clone https://github.com/your-username/VisionaryX - A Multi-Modal LLM.git  
         cd VisionaryX--A Multi-Modal LLM

     2. # Install Dependencies:
          ##
          pip install -r requirements.txt 
          
     3. # Set up API Keys: 
          * Create a .env file and add the following keys:
          ##
          GEMINI_API_KEY=your_google_api_key  
          HUGGINGFACE_API_KEY=your_huggingface_api_key  

     4. # Run the Application:
         ##
         streamlit run application.py

#  Usage #

  1. Text Generation: Enter a prompt, and the chatbot will generate responses.
  2. Text-to-Image: Input a text description, and an AI-generated image will appear.
  3. Image-to-Text (OCR): Upload an image to extract text from it.
  4. PDF-to-Text: Upload a PDF, and the text content will be retrieved.

# Dependencies #

 * Streamlit
 * langchain
 * google.generativeai
 * python-dotenv
 * huggingface_hub
 * pypdf
   




           





