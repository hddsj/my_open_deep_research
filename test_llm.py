"""Test script to verify DeepSeek LLM integration works correctly."""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from my_deep_research.configuration import Configuration

# Step 1: Load .env file (API keys)
load_dotenv()

# Step 2: Create Configuration instance (uses .env values + defaults)
config = Configuration.from_runnable_config()
print(f"Configuration loaded:")
print(f"  research_model: {config.research_model}")
print(f"  search_api: {config.search_api}")
print(f"  max_researcher_iterations: {config.max_researcher_iterations}")
print()

# Step 3: Initialize the LLM using LangChain
model = init_chat_model(config.research_model, model_provider="deepseek")

# Step 4: Send a test message
print("Sending test message to DeepSeek...")
response = model.invoke("What is LangGraph? Answer in 2 sentences.")

# Step 5: Print the response
print(f"\nResponse from {config.research_model}:")
print(response.content)
