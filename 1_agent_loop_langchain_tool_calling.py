from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATION = 10
MODEL = "qwen3:1.7b"

# Tools


@tool
def get_product_price(product: str) -> float:
    """Lookup a price of the product in the catalogue"""
    print(f"Excueting get prodcut price for the prodcut {product}")
    prices = {"laptop": 1200.99, "headphone": 234.89, "keyboard": 123.55}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier on a price and return a final price.
    Available tiers are bronze, silver and gold
    """
    print(
        f" >>> Executing apply_discount(price={price}, discount_tier={discount_tier})"
    )
    discount_percentages = {"bronze": 5, "silver": 15, "gold": 25}
    discount = discount_percentages.get(discount_tier, 1)
    return round(price * (1 - discount / 100), 2)


# Agent Loop
@traceable(name="Langchain Agent loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question is : {question}")
    print("=" * 60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpfull Shopping assistant."
                "You have access to product catalogue tool."
                "and a discount tool \n \n"
                "STRICT rules - You must follow these exactly : \n"
                "1. NEVER guess or assume any product price"
                "You MUST call get_product_price first to get the real price. "
                "2. Only call  apply_discount tool AFTER you receive  a price"
                "from get_product_price, pass the exact price you receive from  "
                "get_product_price - Do not pass a made-up number. \n"
                "3. NEVER calculate dsicount yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "Ask them with tier to use - DO NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]
    for interation in range(1, MAX_ITERATION + 1):
        print(f"\n ===== Iteration number : {interation} ======")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        # If no tool_calls , then we have the final answer
        if not tool_calls:
            print(f"Final answer is : {ai_message.content}")
            return ai_message.content
        # Process only the first tool_call
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"[Tool Call selected] : {tool_name} with args :{tool_args}")

        tool_to_use = tool_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f" Tool {tool_name} not found.")

        observation = tool_to_use.invoke(tool_args)

        print(f"[Tool Result] : {observation}")

        messages.append(ai_message)

        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
    print("ERROR: Max iteration reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello langchain Agent (.bind_tools)")
    print()
    result = run_agent(
        "What is the price of a laptop after applying the silver discount ?"
    )
    print(result)
