# https://www.geeksforgeeks.org/introduction-to-langchain/
# Generate and Store Your API Key
#   You need to generate your API key from the OpenAI platform by signing up and creating an account. 
#   Once you have the API key, create a .env file in your project directory and add your API key to it like this:

OPENAI_KEY='your_api_key'

# Set Up Your Python Script
#   Next, create a new Python file named lang.py. In this file, you'll use LangChain to generate responses with OpenAI. Start by importing the necessary libraries:
#   This code loads the environment variables from the .env file, where your OpenAI API key is stored.

import os
import openai
import langchain
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_KEY", None)

# Initialize the OpenAI Model
from langchain.llms import OpenAI

# Initialize OpenAI LLM with a temperature of 0.9 for randomness
llm = OpenAI(temperature=0.9, openai_api_key=api_key)

# Generate a Response
# Now that the model is initialized, you can generate a response by passing a simple prompt to it. In this case, we’ll ask, "Suggest me a skill that is in demand?"

response=llm.predict("Suggest me a skill that is in demand?")
print(response)




