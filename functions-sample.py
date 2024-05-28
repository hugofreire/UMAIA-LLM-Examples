
## pip install langchain-experimental

from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage

model = OllamaFunctions(model="llama3:70b", format="json")

model = model.bind_tools(
    tools=[
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, " "e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["location"],
            },
        },
        {
            "name": "return_response",
            "description": "execute this funcction when you already have the response ",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "your response data",
                    },
                },
                "required": ["response"],
            },
        }
    ],
)

response = model.invoke("whats the capital of Portugal?")
print(response.additional_kwargs['function_call'])
