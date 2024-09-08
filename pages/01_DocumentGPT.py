import streamlit as st
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import openai
import os

st.set_page_config(
    page_title="RAG with Streamlit",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"

    folder_path = os.path.dirname(file_path)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_content = file.read()

    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    # OpenAIEmbeddings애 api_key를 전달하였습니다. api-key가 환경변수에 저장되어 있는게 아니라면 직접 전달해줘야하는 것 같습니다.
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
      save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# Open API Key의 유효성을 검사하는 함수입니다. openai.Model.list()는 api_key가 유효할 시 사용 가능한 모델들을 반환하고 반대의 경우 에러를 발생시킵니다.
def validate_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        return False

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

        Context: {context}
        """,
    ),
    ("human", "{question}"),
])

st.title("Document GPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    api_key = st.text_input("Please enter your OpenAI API key", type="password")
    is_valid = validate_api_key(api_key)
    # Open API Key가 유효하지 않을시 file uploader를 비활성화합니다.
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type = ["pdf", "txt", "docx"],
        disabled = not is_valid
    )
    st.link_button("Go to Git repo", "https://github.com/Kosto1221/FULLSTACK-GPT/blob/main/pages/01_DocumentGPT.py")
    with st.expander("See Full code"):
        st.markdown("""python
            import streamlit as st
            from langchain.document_loaders import UnstructuredFileLoader
            from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
            from langchain.storage import LocalFileStore
            from langchain.text_splitter import CharacterTextSplitter
            from langchain.vectorstores.faiss import FAISS
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
            from langchain.chat_models import ChatOpenAI
            from langchain.callbacks.base import BaseCallbackHandler
            import openai

            st.set_page_config(
                page_title="RAG with Streamlit",
                page_icon="📃",
            )

            class ChatCallbackHandler(BaseCallbackHandler):
                message = ""

                def on_llm_start(self, *args, **kwargs):
                    self.message_box = st.empty()

                def on_llm_end(self, *args, **kwargs):
                    save_message(self.message, "ai")

                def on_llm_new_token(self, token, *args, **kwargs):
                    self.message += token
                    self.message_box.markdown(self.message)

            st.cache_data(show_spinner="Embedding file...")
            def embed_file(file):
                file_path = f"./.cache/files/{file.name}"

                folder_path = os.path.dirname(file_path)
                
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                file_content = file.read()

                with open(file_path, "wb") as f:
                    f.write(file_content)
                cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
                splitter = CharacterTextSplitter.from_tiktoken_encoder(
                    separator="\\n",
                    chunk_size=600,
                    chunk_overlap=100,
                )
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load_and_split(text_splitter=splitter)
                # OpenAIEmbeddings애 api_key를 전달하였습니다.
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
                vectorstore = FAISS.from_documents(docs, cached_embeddings)
                retriever = vectorstore.as_retriever()
                return retriever

            def save_message(message, role):
                st.session_state["messages"].append({"message": message, "role": role})

            def send_message(message, role, save=True):
                with st.chat_message(role):
                    st.markdown(message)
                if save:
                save_message(message, role)

            def paint_history():
                for message in st.session_state["messages"]:
                    send_message(
                        message["message"],
                        message["role"],
                        save=False,
                    )

            def format_docs(docs):
                return "\\n\\n".join(document.page_content for document in docs)


            # Open API Key의 유효성을 검사하는 함수입니다.
            def validate_api_key(api_key):
                try:
                    openai.api_key = api_key
                    openai.Model.list()
                    return True
                except Exception:
                    return False

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    '''
                    Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

                    Context: {context}
                    ''',
                ),
                ("human", "{question}"),
            ])

            st.title("Document GPT")

            st.markdown(
                '''
            Welcome!

            Use this chatbot to ask questions to an AI about your files!

            Upload your files on the sidebar.
            '''
            )

            with st.sidebar:
                api_key = st.text_input("Please enter your OpenAI API key", type="password")
                is_valid = validate_api_key(api_key)
                # Open API Key가 유효하지 않을시 file uploader를 비활성화시킵니다.
                file = st.file_uploader(
                    "Upload a .txt .pdf or .docx file",
                    type = ["pdf", "txt", "docx"],
                    disabled = not is_valid
                )
                st.link_button("Go to Git repo", "hhttps://github.com/Kosto1221/FULLSTACK-GPT/blob/main/pages/01_DocumentGPT.py")
                with st.expander("See Full code"):
                    st.write('''
                            코드 전문
                    ''')

            if file:
                retriever = embed_file(file)
                send_message("I'm ready! Ask away!", "ai", save=False)
                paint_history()
                message = st.chat_input("Ask anything about your file...")
                if message:
                    send_message(message, "human")
                    chain = (
                        {
                            "context": retriever | RunnableLambda(format_docs),
                            "question": RunnablePassthrough(),
                        }
                        | prompt
                        | ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0.1,
                            streaming=True,
                            callbacks=[ChatCallbackHandler()],
                            openai_api_key=api_key,
                        )
                    )
                    with st.chat_message("ai"):
                        chain.invoke(message)

            else:
                st.session_state["messages"] = []     


        """)

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                streaming=True,
                callbacks=[ChatCallbackHandler()],
                openai_api_key=api_key,
            )
        )
        with st.chat_message("ai"):
            chain.invoke(message)

else:
    st.session_state["messages"] = []
    





