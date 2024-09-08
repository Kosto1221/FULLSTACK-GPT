from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
import openai

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def validate_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        return False
    
def reset_quiz():
    run_quiz_chain.clear()
     
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 {difficulty} questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Context: {context}
""",
            )
        ]
    )

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = questions_prompt | llm
    response =  chain.invoke({"context": format_docs(_docs), "difficulty": difficulty})
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Please enter your OpenAI API key", type="password")
    is_valid = validate_api_key(api_key)

    if api_key and is_valid:
        st.success("Validation successful", icon="✅")
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
            index = None,
        )
        difficulty = st.selectbox(
            "Choose the level of the questions",
            (
                "easy",
                "moderate",
                "difficult",
            ),
            index = None,
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        elif choice == "Wikipedia Article":
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
    elif api_key:
        st.warning("Validation failed", icon="⚠️")

    st.link_button("Go to Git repo", "https://github.com/Kosto1221/FULLSTACK-GPT/blob/main/pages/02_QuizGPT.py")
    with st.expander("See Full code"):
        st.markdown("""python
        from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import json
from langchain.retrievers import WikipediaRetriever
import openai

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

def format_docs(docs):
    return "\\n\\n".join(document.page_content for document in docs)

def validate_api_key(api_key):
    try:
        openai.api_key = api_key
        openai.Model.list()
        return True
    except Exception:
        return False
    
def reset_quiz():
    run_quiz_chain.clear()
     
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 {difficulty} questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Context: {context}
''',
            )
        ]
    )

@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    chain = questions_prompt | llm
    response =  chain.invoke({"context": format_docs(_docs), "difficulty": difficulty})
    response = response.additional_kwargs["function_call"]["arguments"]
    return json.loads(response)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    topic = None
    api_key = st.text_input("Please enter your OpenAI API key", type="password")
    is_valid = validate_api_key(api_key)

    if api_key and is_valid:
        st.success("Validation successful", icon="✅")
        choice = st.selectbox(
            "Choose what you want to use.",
            (
                "File",
                "Wikipedia Article",
            ),
            index = None,
        )
        difficulty = st.selectbox(
            "Choose the level of the questions",
            (
                "easy",
                "moderate",
                "difficult",
            ),
            index = None,
        )
        if choice == "File":
            file = st.file_uploader(
                "Upload a .docx , .txt or .pdf file",
                type=["pdf", "txt", "docx"],
            )
            if file:
                docs = split_file(file)
        elif choice == "Wikipedia Article":
            topic = st.text_input("Search Wikipedia...")
            if topic:
                docs = wiki_search(topic)
    elif api_key:
        st.warning("Validation failed", icon="⚠️")

    st.link_button("Go to Git repo", "https://github.com/Kosto1221/FULLSTACK-GPT/blob/main/pages/02_QuizGPT.py")
    with st.expander("See Full code"):
        st.markdown('''python
 
        ''')


if not docs:
    st.markdown(
        '''
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    '''
    )
else:
    llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    openai_api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )
    
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)

    total_questions = len(response["questions"])
    correct_answers = 0
    
    with st.form("questions_forms"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                label_visibility="collapsed",
                index=None,
            )
            if value:
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_answers += 1                   
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
        
        button = st.form_submit_button(disabled = correct_answers == total_questions)

    if button:
        if correct_answers == total_questions:
            st.button(f":green[{correct_answers} out of {total_questions} questions correct. well done!]", use_container_width=True, disabled=True)
            st.balloons()
        else:            
            st.button(f":red[{correct_answers} out of {total_questions} questions correct. Click to retry!]", use_container_width=True, on_click=reset_quiz)
        """)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    openai_api_key=api_key,
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )
    
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)

    total_questions = len(response["questions"])
    correct_answers = 0
    
    with st.form("questions_forms"):
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                label_visibility="collapsed",
                index=None,
            )
            if value:
                if {"answer": value, "correct": True} in question["answers"]:
                    correct_answers += 1                   
                    st.success("Correct!")
                elif value is not None:
                    st.error("Wrong!")
        
        button = st.form_submit_button(disabled = correct_answers == total_questions)

    if button:
        if correct_answers == total_questions:
            st.button(f":green[{correct_answers} out of {total_questions} questions correct. well done!]", use_container_width=True, disabled=True)
            st.balloons()
        else:            
            st.button(f":red[{correct_answers} out of {total_questions} questions correct. Click to retry!]", use_container_width=True, on_click=reset_quiz)

