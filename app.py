import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorStore):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        temperature=0.5,
        model_kwargs={"max_length": 512},
        timeout=10000
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please process your PDFs first.")
        return None

    user_question = user_question.strip()
    if not user_question:
        st.warning("Please enter a question.")
        return None

    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response.get('chat_history', [])
        return response
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None

def main():
    load_dotenv()
    st.set_page_config(page_title='Chat with Multiple PDFs', page_icon='pdf.png')
    st.markdown("<h1 style='display: inline';>Chat with Multiple PDFs</h1>",unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.markdown(f"<h4>Hello, I am a bot. How can I help you?</h4>", unsafe_allow_html=True)
            
    user_query = st.text_input("Ask a question?")


    if user_query:
        with st.spinner("Getting response..."):
            # Display the user query
            with st.chat_message("Human"):
                st.markdown(user_query)

            # Get the response from the function
            response = handle_userinput(user_query)

            if response is not None:
                # Append user query to chat history only once
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                
                # Display AI response from the last response
                if response:  # Ensure response exists
                    ai_message = response.get('answer')  # Adjust based on your response structure
                    st.session_state.chat_history.append(AIMessage(content=ai_message))

                    with st.chat_message("AI"):
                        st.markdown(ai_message)


    with st.sidebar:
        st.subheader("Your Docs:")
        pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vectorStore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorStore)
                    st.success("Processing Done.")
                else:
                    st.warning("No text found in the uploaded PDFs.")

if __name__ == '__main__':
    main()
