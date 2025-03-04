from langchain_community.llms import HuggingFaceHub
import streamlit as st
from huggingface_hub import InferenceClient
import google.generativeai as genai
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configure Google Gemini API and Hugging Face token
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="VisionaryX", page_icon="🔮", layout="wide")
st.title("Welcome to **VisionaryX** 🧠")

# Initialize session state for conversation history
if "text_conversation_history" not in st.session_state:
    st.session_state.text_conversation_history = []
if "pdf_conversation_history" not in st.session_state:
    st.session_state.pdf_conversation_history = []
    
# User and Bot icons
user_image = "user_5_logo.png"  
bot_image = "bot_8_logo.png"    

# -------------------- Gemini 1.5 Flash functions --------------------
#For Text-Generation function
def get_gemini_text_response(question):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(question)
    return response.text

#For Image + Text Generation function
def get_gemini_image_response(input_text, image):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if input_text != "":
        response = model.generate_content([input_text, image])
    else:
        response = model.generate_content(image)
    return response.text

# -------------------- Hugging Face text-to-image function --------------------
def generate_image_from_text(prompt):
    client = InferenceClient(token=hf_token)
    image = client.text_to_image(prompt)
    return image

# -------------------- PDF Q&A Vector Embedding functions --------------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the
    answer is not in the context, say "answer is not available in the context". Don't
    guess.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.pdf_conversation_history.append((user_question, response["output_text"]))
    st.write("Reply: ", response["output_text"])

#--------Authentication for the App------------
def creds_entered():
    if st.session_state["user"].strip() == "admin" and st.session_state["passwd"].strip() == "admin":
        st.session_state["authenticated"] = True
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please Enter Password")
        elif not st.session_state["user"]:
            st.warning("Please Enter Username")
        else:
            st.error("Invalid Username/Password")

def authenticate_user():
    # Ensure session state variables are initialized
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = ""
    if "passwd" not in st.session_state:
        st.session_state["passwd"] = ""

    if not st.session_state["authenticated"]:
        # Center alignment with custom HTML/CSS for input fields and button
        st.markdown(
            """
            <style>
                .login-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    height: 10vh;
                }
                .stTextInput {
                    width: 300px;
                }
                .stButton > button {
                    width: 150px;
                    font-size: 16px;
                    margin-top: 20px;
                }
            </style>
            <div class="login-container">
            """,
            unsafe_allow_html=True,
        )

        st.text_input("Username:", key="user", placeholder="Enter your username", max_chars=15)
        st.text_input("Password:", key="passwd", placeholder="Enter your password", type="password", max_chars=15)

        if st.button("Login"):
            creds_entered()

        st.markdown("</div>", unsafe_allow_html=True)

        return False
    else:
        return True



      
if authenticate_user():

# -------------------- Main Streamlit UI --------------------
    st.sidebar.header("User Dashboard")
    activity = st.sidebar.radio("Choose an Activity", ("Home", "Image-Based-Query", "Text-to-Image", "Chat with PDF"))
    # "Clear Interaction History" button in the sidebar
    if st.sidebar.button("Clear Interaction History"):
        st.session_state.text_conversation_history = []
        st.session_state.pdf_conversation_history = []
        st.sidebar.success("Interaction history cleared!")
    # Handle "Home" page
    if activity == "Home":
        st.subheader("Welcome to VisionaryX - Your AI Assistant")
        st.write("""
            This app allows you to interact with **VisionaryX** for tasks like generating images, 
            chatting with uploaded PDFs, and more. Use the sidebar to choose your activity.
        """)

        # Ask Me Anything Section : Text Generation Section
        st.write("### Ask Me Anything! 🤖")
        user_question = st.text_input("Type your question here:")

        if user_question:
            with st.spinner("Thinking..."):
                response = get_gemini_text_response(user_question)
                st.write("**VisionaryX Response:**", response)

    # Handle "Image-Based Query" page
    elif activity == "Image-Based-Query":
        st.subheader("Image-Based Query 📸")

        # Upload image file
        uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
        image = None
        input_text = st.text_input("Enter your question:", key="input_prompt")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)  # Open the image file
            
            # Resize the image to a fixed width and height
            fixed_size = (300, 300)  # Set the desired width and height in pixels
            resized_image = image.resize(fixed_size)
            st.image(image, caption="Uploaded Image", use_column_width=False)  # Display the uploaded image

        # When "Tap to Ask" is pressed
        if st.button("Tap to Ask"):
            with st.spinner("Generating Response..."):
                # Default placeholder text for image-only queries
                default_text = "Explain this image"

                # Case 1: Both image and text are provided
                if image is not None and input_text != "":
                    response = get_gemini_image_response(input_text, image)
                # Case 2: Only the image is provided; use default text
                elif image is not None:
                    response = get_gemini_image_response(default_text, image)
                # Case 3: Only text is provided (no image)
                elif input_text != "":
                    response = get_gemini_text_response(input_text)
                else:
                    st.error("Please provide either a text prompt or an image!")

                # Display the response if it exists
                if response:
                    st.write("**VisionaryX Response:**", response)


    # Handle "Text-to-Image" page
    elif activity == "Text-to-Image":
        st.subheader("Text-to-Image Generation 🖼️")
        prompt_for_image = st.text_input("Enter prompt for image generation:", "e.g., 'sunset over a beach'")

        if st.button("Generate Image"):
            with st.spinner("Generating Image..."):
                generated_image = generate_image_from_text(prompt_for_image)
                resized_image = generated_image.resize((400, 400))
                st.image(resized_image, caption="Generated Image", use_column_width=False)

                img_byte_arr = io.BytesIO()
                resized_image.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()
                st.download_button(label="Download Image", data=img_byte_arr, file_name="generated_image.png", mime="image/png")

    # Handle "Chat with PDF" page
    elif activity == "Chat with PDF":
        st.subheader("Chat with PDF 📄")

        uploaded_file = st.file_uploader("Upload PDF File(s)", accept_multiple_files=True)

        if uploaded_file is not None:
            st.write("Uploaded PDF(s):", [file.name for file in uploaded_file])

        # Process PDF button
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(uploaded_file)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("PDF Processed Successfully!")

        user_question = st.text_input("Ask a question about the uploaded PDF:")

        if user_question:
            user_input(user_question)

        st.write("### PDF Interaction History:")
        for question, reply in st.session_state.pdf_conversation_history:
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.image(user_image, width=30)
            with col2:
                st.write(f"**You:** {question}")
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.image(bot_image, width=30)
            with col2:
                st.write(f"**VisionaryX:** {reply}")
