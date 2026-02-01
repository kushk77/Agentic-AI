print("Chatbot file started")

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5
)

# Web search tool
search_tool = TavilySearchResults(max_results=3)

print("🤖 Agentic AI Chatbot with Internet is running!")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("👋 Goodbye!")
        break

    # WEATHER LOGIC
    if "weather" in user_input.lower() or "temperature" in user_input.lower():
        search_result = search_tool.invoke(
            {"query": f"current weather in {user_input}"}
        )

        messages = [
            HumanMessage(
                content=(
                    "Using the following live data, answer clearly in CELSIUS only.\n"
                    f"{search_result}"
                )
            ),
            HumanMessage(content=user_input)
        ]
    else:
        messages = [HumanMessage(content=user_input)]

    response = llm.invoke(messages)
    print("Bot:", response.content)
