# GENAI_agent
# AutoStream Conversational AI Agent

This repository contains a real-world, stateful Conversational AI Agent built for AutoStream, a fictional SaaS product for automated video editing. The agent handles intent routing, performs RAG over a local knowledge base for pricing inquiries, and executes a backend lead capture tool when high intent is detected.

## 1. How to run the project locally

1. Clone this repository to your local machine.
2. Ensure you have Python 3.9+ installed.
3. Create a virtual environment and activate it:
   - Mac/Linux: `python3 -m venv venv && source venv/bin/activate`
   - Windows: `python -m venv venv && venv\Scripts\activate`
4. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt
Create a .env file in the root directory and add your Google Gemini API key:

Plaintext
GOOGLE_API_KEY=your_api_key_here
Run the agent:

Bash
python agent.py

## 2. Architecture Explanation
For this project, I chose LangGraph combined with Gemini 2.5 Flash (via langchain.agents) rather than standard LangChain chains or AutoGen. Building an AI agent intended for commercial deployment requires deterministic execution, reliable tool-calling loops, and robust conversational memory. LangGraph's state machine architecture natively supports cyclical agent logic (the ReAct pattern), allowing the LLM to dynamically reason about whether it needs to query the RAG vector database or trigger the lead capture tool based on the conversation state.

State management is handled using LangGraph's MemorySaver checkpointer. Every interaction is tied to a specific thread_id, which acts as a session identifier. As the user converses, MemorySaver appends new inputs and tool outputs to an internal messages state list. When the LLM evaluates the state to decide its next action, it references this entire memory buffer. This ensures the agent perfectly retains context across 5-6 conversation turns, enabling it to remember previously provided details (like a user's content platform) without asking for them repeatedly before safely executing the final mock_lead_capture tool.

## 3. WhatsApp Deployment Integration
To deploy this agent to WhatsApp, I would utilize the Meta Cloud API alongside a Python backend framework like FastAPI.

Webhook Setup: I would create a /webhook endpoint in FastAPI handling both GET requests (for Meta's initial verification challenge) and POST requests (for receiving live user messages).

State & Memory Integration: When a user messages the WhatsApp number, the Meta webhook pushes a JSON payload containing the user's phone number and message text. I would extract the user's phone number and pass it into LangGraph as the thread_id. This guarantees persistent, isolated memory for every unique WhatsApp user.

Execution & Response: The backend passes the incoming text to the agent.invoke() method with the corresponding thread_id. Once the LLM finishes reasoning and generates its final text state, the backend fires a POST request back to the Meta Graph API's /messages endpoint, delivering the agent's response directly to the user's phone.
