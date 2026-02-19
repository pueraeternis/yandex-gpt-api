import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

# Fix path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

from src.clients.wrapper import get_openai_client
from src.config import config, logger

# --- 1. Define Real Python Functions ---


def get_current_weather(location: str, unit: str = "celsius"):
    """Mock function to get weather."""
    logger.info(f"--> Tool called: get_current_weather({location}, {unit})")
    if "moscow" in location.lower():
        return json.dumps({"location": "Moscow", "temperature": "-5", "unit": unit})
    if "dubai" in location.lower():
        return json.dumps({"location": "Dubai", "temperature": "30", "unit": unit})
    return json.dumps({"location": location, "temperature": "unknown"})


# --- 2. Describe Tools for the AI ---

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]


def run_tools_demo():
    client = get_openai_client()

    # User asks a question that requires an external tool
    user_prompt = "What is the weather like in Moscow today?"
    print(f"User: {user_prompt}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
    ]

    # 1. First Call: Model decides if it needs a tool
    logger.info("Sending initial request...")
    response = client.chat.completions.create(
        model=config.model_uri,
        messages=cast("list[ChatCompletionMessageParam]", messages),
        tools=cast("list[ChatCompletionToolUnionParam]", TOOLS_SCHEMA),
        tool_choice="auto",  # Let the model decide
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # The model wants to call a function!
        print(f"AI: I need to call a function: {tool_calls[0].function.name}")  # pyright: ignore[reportAttributeAccessIssue]

        # Append the model's request to history
        messages.append(response_message)  # pyright: ignore[reportArgumentType]

        # 2. Execute the function locally
        for tool_call in tool_calls:
            function_name = tool_call.function.name  # pyright: ignore[reportAttributeAccessIssue]
            function_args = json.loads(tool_call.function.arguments)  # pyright: ignore[reportAttributeAccessIssue]

            if function_name == "get_current_weather":
                function_response = get_current_weather(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius"),
                )

                # 3. Feed the result back to the model
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    },
                )

        # 4. Second Call: Model generates the final answer
        logger.info("Sending results back to model...")
        final_response = client.chat.completions.create(
            model=config.model_uri,
            messages=cast("list[ChatCompletionMessageParam]", messages),
            tools=cast("list[ChatCompletionToolUnionParam]", TOOLS_SCHEMA),
        )

        print(f"AI Final Answer: {final_response.choices[0].message.content}")

    else:
        print("AI didn't call any tools.")


if __name__ == "__main__":
    run_tools_demo()
