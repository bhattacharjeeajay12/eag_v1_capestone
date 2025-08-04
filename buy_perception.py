from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_provider import default_llm

class ProductAttributes(BaseModel):
    color: Optional[str] = None
    gender: Optional[Literal["male", "female", "unisex"]] = None
    age: Optional[int] = None
    age_group: Optional[Literal["kids", "teen", "adult", "senior"]] = None
    size: Optional[Literal["XS", "S", "M", "L", "XL", "XXL"]] = None

class BuySearch(BaseModel):
    product_name: str = Field(..., description="Product category or name")
    attributes: Optional[ProductAttributes] = None

system_prompt = """
You are a buy-query perception assistant.
Extract structured product details:
- product_name
- attributes: color, gender, age, age_group, size
{format_instructions}
"""

def get_buy_chain():
    parser = PydanticOutputParser(pydantic_object=BuySearch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "User query: {user_query}")]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | default_llm.chat_model | parser

def extract_buy_details(query: str, perception_chain) -> BuySearch:
    return perception_chain.invoke({"user_query": query})
