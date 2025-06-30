# https://python.langchain.com/docs/tutorials/llm_chain/

# After you sign up at the link above, make sure to set your environment variables to start logging traces:
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
export LANGSMITH_PROJECT="default" # or any other project name

import getpass
import os

try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

# Using Language Models
pip install -qU "langchain[google-genai]"

import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

# Let's first use the model directly.
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

model.invoke(messages)

# all these are the same as above
model.invoke("Hello")

model.invoke([{"role": "user", "content": "Hello"}])

model.invoke([HumanMessage("Hello")])

# Prompt Templates
# Let's create a prompt template here. It will take in two user variables:

#     language: The language to translate text into
#     text: The text to translate

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

# Note that ChatPromptTemplate supports multiple message roles in a single template. We format the language parameter into the system message, and the user text into a user message.

# The input to this prompt template is a dictionary. We can play around with this prompt template by itself to see what it does by itself

prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
prompt.to_messages()

# Finally, we can invoke the chat model on the formatted prompt:
response = model.invoke(prompt)
print(response.content)











