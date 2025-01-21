import streamlit as st
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import openai

class ChatCallbackhandler(BaseCallbackHandler):

    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5. 0 being not helpful to the user and 5 being helpful to the user.

    Make sure to include the answer's score.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}
    """
)

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"]
            } for doc in docs
        ]
    }

choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY following pre-existing answers to answer the user's question.
            
            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}")
    ]
)

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n" for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed
        }
    )

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")

def save_message(message, role):
     st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
       save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )
    loader = SitemapLoader(
        url, 
        filter_urls=[
            r"^https://developers\.cloudflare\.com/ai-gateway/.*",
            r"^https://developers\.cloudflare\.com/vectorize/.*",
            r"^https://developers\.cloudflare\.com/workers-ai/.*"
        ],
        parsing_function=parse_page
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

def get_vectorstore(docs, api_key):
    return FAISS.from_documents(docs, OpenAIEmbeddings(api_key=api_key)).as_retriever()

with st.sidebar:
    input_disabled = True
    api_key = st.text_input("Insert OpenAI API Key", type="password")
    if api_key:
        try:
            openai.api_key = api_key
            openai.models.list()
            input_disabled=False
            llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.1)
        except openai.AuthenticationError:
            st.error("Invalid API key. Please check and try again.")   
    url = st.text_input("Write down a sitemap path", placeholder="https://example.xml", disabled=input_disabled)
    st.link_button("Visit repository", "https://github.com/Kosto1221/fullstack-gpt")

if api_key and llm and url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    else:
        docs = load_website(url)
        retriever = get_vectorstore(docs, api_key)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask a question to the website.")
        if message:
            send_message(message, "human")
            chain = {"docs": retriever, "question": RunnablePassthrough()} | RunnableLambda(get_answers) | RunnableLambda(choose_answer)
            result = chain.invoke(message).content
            send_message(result, "ai")

else:
   if "messages" not in st.session_state:
        st.session_state["messages"] = []





# 챗봇 만들기, 스트리밍, 히스토리, question cache, 비슷한 질문도 cache, question -> list애서 확인 -> , list, llm애게 전달