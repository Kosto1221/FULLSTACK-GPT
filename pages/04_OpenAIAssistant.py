import streamlit as st
from langchain.tools import WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
import openai as client
import json

st.cache_data.clear()

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'assistant_id' not in st.session_state:
    st.session_state['assistant_id'] = None

def WikipediaSearchTool(query):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return wikipedia.run(query)

def DuckDuckGoSearchTool(query):
    ddg = DuckDuckGoSearchRun()
    return ddg.run(query)

def URLReaderTool(urls):
    if isinstance(urls, dict):
        urls = urls.get("url", "")
    if isinstance(urls, str):
        urls = [urls] 
    loader = WebBaseLoader(urls)
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return text

functions_map = {
    "WikipediaSearchTool": WikipediaSearchTool,
    "DuckDuckGoSearchTool": DuckDuckGoSearchTool,
    "URLReaderTool": URLReaderTool,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "WikipediaSearchTool",
            "description": "Use this tool to find the urls for query. It takes a query as an argument and returns at least one url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "urls for query",
                    }
                },
                "required": ["query"],
            }
        } 
    },
    {
        "type": "function",
        "function": {
            "name": "DuckDuckGoSearchTool",
            "description": "Use this tool to find the urls for query. It takes a query as an argument and returns at least one url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "urls for query",
                    }
                },
                "required": ["query"],
            }
        } 
    },
    {
        "type": "function",
        "function": {
            "name": "URLReaderTool",
            "description": "Use this tool to parse contents from URLs given by DuckDuckGoSearchTool and WikipediaSearchTool",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "string",
                        "desctription": "The URLs' list given by DuckDuckGoSearchTool and WikipediaSearchTool."
                    }
                },
                "required": ["urls"],
            }
        }
    },

]

@st.cache_data
def get_assistants(_client):
    assistant = _client.beta.assistants.create(
        name="Research Assistant",
        instructions="You are a personal Research Assistant. You help users do research on topics.",
        model="gpt-4o-mini",
        tools=functions,
    )
    return assistant.id


def validate_api_key(api_key):
    try:
        client.api_key = api_key
        client.Model.list()
        return True
    except Exception:
        return False

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

st.set_page_config(page_title="OpenAI Assistant", page_icon="🔬")

st.title("OpenAI Assistant")

with st.sidebar:
    api_key = st.text_input("Please enter your OpenAI API key", type="password")
    is_valid = validate_api_key(api_key)
    if api_key and is_valid:
        st.success("Validation successful", icon="✅")
    elif api_key:
        st.warning("Validation failed", icon="⚠️")
    st.link_button("Go to Git repo", "https://github.com/Kosto1221/FULLSTACK-GPT/blob/main/pages/04_OpenAIAssistant.py")

if api_key and is_valid:
    paint_history()
    query = st.chat_input("Please enter the topic you wish to research")

    if query:
        send_message(role="user", message=query)

        if st.session_state['assistant_id'] is None:
            st.session_state['assistant_id'] = get_assistants(_client=client)

        assistant_id = st.session_state['assistant_id']

        thread = client.beta.threads.create()

        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Research about {query}"
        )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant_id)

        while run.status != 'completed':
            if run.required_action.type == 'submit_tool_outputs':
                tool_outputs = []
                for tool in run.required_action.submit_tool_outputs.tool_calls:
                    function = tool.function
                    tool_id = tool.id
                    tool_name = function.name
                    tool_inputs = json.loads(function.arguments)
                    print(f"Calling function: {function.name} with arg {function.arguments}")
                    tool_output = functions_map[tool_name]
                    tool_outputs.append({
                        "tool_call_id": tool_id,
                        "output": tool_output(tool_inputs)
                    })

                run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                    thread_id=thread.id,
                    run_id=run.id,
                    tool_outputs=tool_outputs)
            else:
                run = client.beta.threads.runs.poll(thread_id=thread.id, run_id=run.id)

        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            messages_list = list(messages)
            ai_message = messages_list[0].content[0].text.value
            send_message(role="ai", message=ai_message)
        else:
            send_message(role="ai", message="Failed to find what you requested. Please try again.", save=False)
else:
    st.session_state['messages'] = []
