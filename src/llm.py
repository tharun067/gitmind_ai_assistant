from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, MessagesState,START
from dotenv import load_dotenv
from typing import Literal, TypedDict, Annotated
import os

from src.vector_store_git_loader import VectorStoreManager
from src.prompts import (
    get_code_index_agent_messages,
    get_documentation_agent_messages,
    get_qa_agent_messages,
    get_history_agent_messages
)

load_dotenv()
retriever = VectorStoreManager().as_retriever()

llm = None


class ExtendedMessagesState(MessagesState):
    """Extended state with conversation memory"""
    answer: str = ""
    conversation_history: list = []

def start_node(state: ExtendedMessagesState):
    """Entry node that prepares the state with conversation history"""
    # Keep only last 5 message pairs (10 messages total)
    messages = state.get("messages", [])
    if len(messages) > 10:
        messages = messages[-10:]
    
    return {
        "messages": messages,
        "conversation_history": messages[-10:] if len(messages) > 0 else []
    }

def router(state: ExtendedMessagesState) -> Literal["code_index_agent", "documentation_agent", "history_agent", "qa_agent"]:
    """Enhanced router with better keyword detection and context awareness"""
    
    # Get the latest user message
    user_question = state["messages"][-1].content.lower()
    
    # Define keyword patterns for each agent
    code_keywords = ["code", "function", "method", "class", "implementation", 
                     "snippet", "script", "module", "import", "def ", "async"]
    
    doc_keywords = ["documentation", "docs", "readme", "guide", "tutorial",
                    "reference", "api", "manual", "how to", "instructions"]
    
    history_keywords = ["previous", "earlier", "before", "last time", 
                        "remind me", "recall", "history", "conversation"]
    
    # Check for keywords in the user question
    if any(keyword in user_question for keyword in code_keywords):
        return "code_index_agent"
    elif any(keyword in user_question for keyword in doc_keywords):
        return "documentation_agent"
    elif any(keyword in user_question for keyword in history_keywords):
        return "history_agent"
    else:
        return "qa_agent"
    

def code_index_agent(state: ExtendedMessagesState):
    """Agent to handle code-related queries"""
    user_question = state["messages"][-1].content
    context = retriever.get_relevant_documents(user_question)

    ### Include conversation history in the prompt
    messages = get_code_index_agent_messages(user_question, context)

    
    reply = llm.invoke(messages)
    return {
        "messages": [("assistant", reply.content)],
        "answer": reply.content
    }

def documentation_agent(state: ExtendedMessagesState):
    """Agent to handle documentation-related queries"""
    user_question = state["messages"][-1].content
    context = retriever.get_relevant_documents(user_question)

    ### Include conversation history in the prompt
    messages = get_documentation_agent_messages(user_question, context)

    
    reply = llm.invoke(messages)
    return {
        "messages": [("assistant", reply.content)],
        "answer": reply.content
    }

def qa_agent(state: ExtendedMessagesState):
    """Agent to handle QA-related queries"""
    user_question = state["messages"][-1].content
    context = retriever.get_relevant_documents(user_question)

    ### Include conversation history in the prompt
    messages = get_qa_agent_messages(user_question, context)

    reply = llm.invoke(messages)
    return {
        "messages": [("assistant", reply.content)],
        "answer": reply.content
    }

def history_agent(state: ExtendedMessagesState):
    """Agent to handle history-related queries"""
    user_question = state["messages"][-1].content
    context = retriever.get_relevant_documents(user_question)

    ### Include conversation history in the prompt
    messages = get_history_agent_messages(user_question, context)


    reply = llm.invoke(messages)
    return {
        "messages": [("assistant", reply.content)],
        "answer": reply.content
    }


def agent_build(api_key: str = None):
    """Build the state graph for the agents"""
    global llm

    if api_key:
        llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
    else:
        # Try to get from environment variable
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key is required. Provide it as parameter or set GROQ_API_KEY environment variable")
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)
    builder = StateGraph(ExtendedMessagesState)

    builder.add_node("start", start_node)

    ## Add all agent nodes
    builder.add_node("code_index_agent", code_index_agent)
    builder.add_node("documentation_agent", documentation_agent)
    builder.add_node("qa_agent", qa_agent)
    builder.add_node("history_agent", history_agent)

    ## Set entry point to start node
    builder.add_edge(START, "start")

    ## Add routing logic
    builder.add_conditional_edges(
        "start",
        router,
        {
            "code_index_agent": "code_index_agent",
            "documentation_agent": "documentation_agent",
            "qa_agent": "qa_agent",
            "history_agent": "history_agent"
        },
    )

    ## Add terminal nodes
    builder.add_edge("code_index_agent", END)
    builder.add_edge("documentation_agent", END)
    builder.add_edge("qa_agent", END)
    builder.add_edge("history_agent", END)

    graph = builder.compile()

    return graph

"""
def main():
    # Main function with conversation memory
    graph = agent_build()
    conversation_messages = []
    
    print("=" * 60)
    print("Chat with your Git repository!")
    print("Type 'exit' to end the session or 'clear' to reset memory")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "exit":
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == "clear":
            conversation_messages = []
            print("\n[Memory cleared]")
            continue
        
        if not user_input:
            continue
        
        # Add user message to conversation history
        conversation_messages.append(("user", user_input))
        
        # Keep only last 10 messages (5 exchanges)
        if len(conversation_messages) > 10:
            conversation_messages = conversation_messages[-10:]
        
        # Invoke the graph with conversation history
        try:
            response = graph.invoke({
                "messages": conversation_messages,
                "conversation_history": conversation_messages
            })
            
            # Extract answer
            last_message = response["messages"][-1]
            answer = last_message.content if hasattr(last_message, "content") else last_message["content"]
            
            # Add assistant response to conversation history
            conversation_messages.append(("assistant", answer))
            
            # Keep only last 10 messages
            if len(conversation_messages) > 10:
                conversation_messages = conversation_messages[-10:]
            
            print(f"\nBot: {answer}")
            
        except Exception as e:
            print(f"\n[Error]: {str(e)}")
            # Remove the last user message if there was an error
            if conversation_messages and len(conversation_messages) > 0:
                if conversation_messages[-1][0] == "user":
                    conversation_messages.pop()

if __name__ == "__main__":
    main()

"""