import os
from dotenv import load_dotenv
from langchain_core.tools import tool, create_retriever_tool
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
# THE FIX: Importing from the modern langchain.agents module
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

def setup_rag_tool():
    """Loads local markdown, embeds it, and returns a RAG retriever tool."""
    loader = TextLoader("knowledge.md")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    retriever = vectorstore.as_retriever()
    
    rag_tool = create_retriever_tool(
        retriever,
        "autostream_knowledge_base",
        "Search this tool for any questions regarding AutoStream pricing, features, or company policies."
    )
    return rag_tool

@tool
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Call this tool ONLY when you have collected the user's Name, Email, and Content Creator Platform.
    Do not guess these values.
    """
    print(f"\n[BACKEND TRIGGERED] Lead captured successfully: {name}, {email}, {platform}\n")
    return "Lead successfully captured in the backend."

def build_agent():
    """Initializes the LangChain agent with memory and tools."""
  
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    rag_tool = setup_rag_tool()
    tools = [rag_tool, mock_lead_capture]

    system_instruction = """You are a conversational AI agent for AutoStream, a SaaS product for automated video editing.
    
    Your core responsibilities based on user intent:
    1. Casual Greeting: Be polite, concise, and ask how you can help.
    2. Product/Pricing Inquiry: Use the 'autostream_knowledge_base' tool to fetch accurate information. DO NOT hallucinate pricing.
    3. High-Intent Lead: If a user expresses interest in buying, signing up, or trying a plan, shift to lead capture.
    
    Lead Capture Rules:
    - You must collect three distinct pieces of information: Name, Email, and Content Creator Platform (e.g., YouTube, Instagram).
    - Ask for missing information conversationally.
    - ONLY when you have all three pieces of information, call the 'mock_lead_capture' tool.
    - Never call 'mock_lead_capture' prematurely.
    """

    memory = MemorySaver()
 
    agent_executor = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=system_instruction,
        checkpointer=memory
    )
    
    return agent_executor

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is missing. Please check your .env file.")
        exit(1)

    print("Setting up AutoStream Agent...")
    
    try:
        agent = build_agent()
    except Exception as e:
        print(f"\nFailed to build the RAG pipeline or Agent. Error details:\n{e}")
        exit(1)

    config = {"configurable": {"thread_id": "user_session_001"}}
    
    print("\nAgent ready! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        try:
            response = agent.invoke(
                {"messages": [("user", user_input)]}, 
                config=config
            )

            last_message = response['messages'][-1]

            if isinstance(last_message.content, list):
                text_response = "".join([block['text'] for block in last_message.content if block['type'] == 'text'])
            else:
                text_response = last_message.content

            print(f"Agent: {text_response}")
            
        except Exception as e:
            print(f"An error occurred during generation: {e}")