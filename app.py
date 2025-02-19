import streamlit as st
import fitz
from PIL import Image
import os
import io
import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from langchain.vectorstores import FAISS
from langchain.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to extract text and images from the PDF
def load_pdf_with_images(pdf_path):
    doc = fitz.open(pdf_path)
    text_data = []
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_data.append(page.get_text("text"))
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(io.BytesIO(image_bytes))
            img_path = f"page-{page_num+1}-img-{img_index+1}.png"
            img.save(img_path)
            images.append(img_path)

    return text_data, images
def generate_image_embeddings(images):
    embeddings = []
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            normalized = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embeddings.append(normalized.cpu().numpy())
    return np.vstack(embeddings)
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #008000;
        color: #333333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("📄 PDF INTERACTION QA BOT ")
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
if uploaded_file is not None:
    with st.spinner("Processing your PDF..."):
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.read())
        texts, images = load_pdf_with_images("uploaded_file.pdf")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_chunks = text_splitter.create_documents(texts)
        text_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        text_db = FAISS.from_documents(text_chunks, text_embeddings)
        image_embeddings = generate_image_embeddings(images)
        dimension = image_embeddings.shape[1]
        image_index = faiss.IndexFlatL2(dimension)
        image_index.add(image_embeddings)
        os.environ['GROQ_API_KEY'] = '#'
        llm = ChatGroq(model_name='llama-3.1-8b-instant')
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
        retriever = text_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            memory=memory,
            retriever=retriever
        )
        st.success("PDF processed successfully!")
        tab1, tab2 = st.tabs(["💬 Ask Questions", "📸 Explore Images"])
        with tab1:
            query = st.text_input("Ask a question about the PDF or images:")
            submit_button = st.button("Submit Query")
        
            if submit_button and query:
                with st.spinner("Searching for answers..."):
                    query_inputs = processor(text=[query], return_tensors="pt")
                    with torch.no_grad():
                        query_features = model.get_text_features(**query_inputs)
                        query_normalized = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
                    query_array = query_normalized.cpu().numpy()
                    k = 3
                    distances, indices = image_index.search(query_array, k)
                    result = qa({"question": query})
                    answer = result['answer']
                    st.write("### 💡 Answer:", answer)
                    st.info("Check the '📸 Explore Images' tab to view related images!")

        with tab2:
            st.subheader("📸 Extracted Images")
            for img_path in images:
                st.image(img_path, use_container_width=True)
    st.sidebar.subheader("📜 Extracted Text")
    for i, text in enumerate(texts):
        with st.sidebar.expander(f"Page {i + 1}"):
            st.write(text)
else:
    st.info("Upload a PDF file to get started!")
