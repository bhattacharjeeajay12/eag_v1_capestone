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

class ReplacementDetails(BaseModel):
    product_name: str
    attributes: Optional[ProductAttributes] = None

class ReturnSearch(BaseModel):
    order_id: Optional[str] = None
    product_name: Optional[str] = None
    order_date: Optional[str] = None
    attributes: Optional[ProductAttributes] = None
    return_action: Literal["return_only", "replace", "refund"] = None
    replacement_product: Optional[ReplacementDetails] = None

system_prompt = """
You are a return-query perception assistant.
Extract:
- order_id, product_name, order_date
- attributes: color, gender, age, size
- return_action: return_only / replace / refund
- replacement_product if applicable
{format_instructions}
"""

def get_return_chain():
    # Perception chain
    parser = PydanticOutputParser(pydantic_object=ReturnSearch)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "User query: {user_query}")]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | default_llm.chat_model | parser

def extract_return_details(query: str, perception_chain) -> ReturnSearch:
    return perception_chain.invoke({"user_query": query})
