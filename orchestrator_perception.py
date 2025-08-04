from pydantic import BaseModel, Field
from typing import Optional, Literal
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_provider import default_llm

class QueryClassification(BaseModel):
    query_type: Optional[Literal["buy", "return"]] = Field(
        None, description="Classify if query is about buying or returning"
    )
    enhanced_query: Optional[str] = Field(
        None, description="Clear standalone query summarizing user intent"
    )

system_prompt = """
You are an expert ecommerce orchestrator assistant.
Your job is to classify if a query is about:
- Buying products (query_type="buy")
- Returning/replacing/refunding products (query_type="return")

Output:
- query_type: "buy", "return", or null if unclear.
- enhanced_query: clear standalone summary.
{format_instructions}
"""

def get_orchestrator_chain():
    parser = PydanticOutputParser(pydantic_object=QueryClassification)
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "User query: {user_query}")]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | default_llm.chat_model | parser

def classify_query(user_query: str, perception_chain) -> QueryClassification:
    return perception_chain.invoke({"user_query": user_query})
