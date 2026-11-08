from dotenv import load_dotenv

load_dotenv()


import re
import inspect

from langsmith import traceable

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Lookup a price of the product in the catalogue"""
    print(f"Excueting get prodcut price for the prodcut {product}")
    prices = {"laptop": 1200.99, "headphone": 234.89, "keyboard": 123.55}
    return prices.get(product, 0)


@traceable(run_type="tool")
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

tools = {
    "get_product_price" : get_product_price,
    "apply_discount": apply_discount
}

def get_tool_description(tools_dict):
    discriptions = []
    for tool_name, tool_function in tools_dict.item():
        descr