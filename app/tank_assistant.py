import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message


# To extract text from PDF file
def extract_text_from_pdf(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into small chunks
def text_to_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Create Vector Store
def create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


# Initiate conversation chain
def initiate_conversation(vector_store):
    chat_model = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Chat interaction query-response
def process_user_query(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, chat in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(chat.content, is_user=True)
        else:
            message(chat.content)


def main():
    st.set_page_config(page_title='GenAI_Assistant')
    current_dir = os.path.dirname(__file__)
    openai_key_path = os.path.join(current_dir, 'config', 'openai_key')

    # Read OpenAI API key
    with open(openai_key_path, 'r') as file:
        openai_api_key = file.read().strip()
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Add banner image
    image_path = os.path.join(current_dir, 'images', 'header.png')
    st.image(str(image_path))

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # main UI
    st.header('Get Your Tank Inspection Queries Resolved')
    user_question = st.text_input('User Input')
    if user_question:
        process_user_query(user_question)

    with st.sidebar:  # Sidebar UI
        st.subheader('Tank Inspection Queries')
        st.write('Ask your query related to tank inspection')

        # Option to choose between uploading or using already uploaded documents
        st.subheader('\n')
        option = st.radio("Choose your option", ("Upload New Knowledge Base", "Existing Knowledge Base"))

        if option == "Upload Knowledge Base (.pdf)":
            uploaded_files = st.file_uploader('Please Upload PDFs', accept_multiple_files=True)
            if uploaded_files:
                pdf_docs = uploaded_files
        else:
            pdf_file_paths = [
                "knowledge_base/tank_qa_1.pdf"
            ]
            pdf_docs = [open(pdf_path, "rb") for pdf_path in pdf_file_paths]
        if st.button('Proceed'):
            with st.spinner(
                    'Processing Knowledge base...'):  # processing to be done after pdf documents uploaded by user
                # calling Extract text function from PDFs
                raw_text = extract_text_from_pdf(pdf_docs)

                # calling Split text into manageable chunks function
                text_chunks = text_to_chunks(raw_text)

                # calling Create vector store for text chunks function
                vector_store = create_vector_store(text_chunks)

                # calling Initialize conversation chain function for chat interaction
                st.session_state.conversation = initiate_conversation(vector_store)


if __name__ == '__main__':
    main()
