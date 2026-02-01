print("Chatbot started")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# 1. Model

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)


# 2. Internet Tool

search_tool = TavilySearchResults(max_results=3)

print("🤖 ChatGPT-style AI Assistant is running!")
print("Ask me anything. Type 'exit' to quit.\n")


# 3. Conversation Memory

chat_history = [
    SystemMessage(
        content="You are a helpful, friendly, all-purpose AI assistant like ChatGPT."
    )
]

# 4. Chat Loop

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    chat_history.append(HumanMessage(content=user_input))


    # 5. Decide if search is needed

    router_prompt = f"""
    Decide if this question needs real-time or factual information
    (weather, news, prices, current events).
    Answer only YES or NO.

    Question: {user_input}
    """

    decision = llm.invoke(
        [HumanMessage(content=router_prompt)]
    ).content.strip().upper()


    # 6. Use tools if needed

    if decision == "YES":
        search_result = search_tool.invoke({"query": user_input})

        tool_context = SystemMessage(
            content=f"Use the following live information to answer accurately:\n{search_result}"
        )

        messages = chat_history + [tool_context]

    else:
        messages = chat_history


    # 7. Get response

    response = llm.invoke(messages)
    print("Bot:", response.content)

    chat_history.append(AIMessage(content=response.content))
