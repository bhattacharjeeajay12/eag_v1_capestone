from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from llm_provider import default_llm
from memory import ConversationMemory
import json

class ProductAttributes(BaseModel):
    """Common product attributes for both buy and return"""
    color: Optional[str] = Field(None, description="Preferred color of the product if mentioned")
    gender: Optional[Literal["male", "female", "unisex"]] = Field(
        None, description="Gender for whom the product is intended"
    )
    age: Optional[int] = Field(
        None, description="Numeric age if mentioned (e.g., 5 for a 5-year-old child)"
    )
    age_group: Optional[str] = Field(
        None, description="Age group for the product: 'kids', 'teen', 'adult', or 'senior'"
    )
    size: Optional[Literal["XS", "S", "M", "L", "XL", "XXL"]] = Field(
        None, description="Size of the product (XS, S, M, L, XL, XXL)"
    )

class ReplacementDetails(BaseModel):
    """Details for replacement product"""
    product_name: str = Field(..., description="Name or category of the replacement product")
    attributes: Optional['ProductAttributes'] = Field(
        None, description="Attributes of the replacement product (color, size, gender, age)"
    )

class ReturnSearch(BaseModel):
    """Return, replace, or refund search parameters"""
    order_id: Optional[str] = Field(None, description="The order ID for the return/replace/refund")
    product_name: Optional[str] = Field(None, description="The product name being returned")
    order_date: Optional[str] = Field(None, description="The order date if mentioned (e.g., '2 weeks ago')")
    attributes: Optional['ProductAttributes'] = Field(
        None, description="Attributes of the returned product (color, gender, age, size)"
    )
    return_action: Literal["return_only", "replace", "refund"] = Field(
        "return_only", description="The type of return action: return_only, replace, or refund"
    )
    replacement_product: Optional[ReplacementDetails] = Field(
        None, description="If return_action is 'replace', details of the replacement product"
    )



class BuySearch(BaseModel):
    """Buy search parameters"""
    product_name: str = Field(..., description="The product name or category to explore or purchase")
    attributes: Optional[ProductAttributes] = Field(
        None, description="Common product attributes like color, gender, size, age"
    )


class EntitySearch(BaseModel):
    """Ecommerce entity search for buy or return queries"""
    query_type: Optional[Literal["buy", "return"]] = Field(
        None, description="The type of query: 'buy' for purchase/exploration or 'return' for return-related actions"
    )
    enhanced_query: Optional[str] = Field(
        None, description="A standalone question summarizing the user's intent clearly"
    )
    buy_info: Optional[BuySearch] = Field(None, description="Details for buy queries")
    return_info: Optional[ReturnSearch] = Field(None, description="Details for return queries")

    # # optional
    # # 1️⃣ Pre-validation: Normalize keys from raw input
    # @model_validator(mode="before")
    # @classmethod
    # def normalize_keys(cls, values):
    #     """Handle raw LLM outputs or legacy keys (buy → buy_info)."""
    #     if isinstance(values, dict):
    #         if "buy" in values and "buy_info" not in values:
    #             values["buy_info"] = values.pop("buy")
    #         if "return" in values and "return_info" not in values:
    #             values["return_info"] = values.pop("return")
    #     return values
    #
    # # 2️⃣ Post-validation: Enforce query_type consistency
    # @model_validator(mode="after")
    # def ensure_query_type_consistency(self):
    #     """Infer or default query_type after validation."""
    #     if not self.query_type:
    #         if self.buy_info and not self.return_info:
    #             self.query_type = "buy"
    #         elif self.return_info and not self.buy_info:
    #             self.query_type = "return"
    #         else:
    #             self.query_type = None  # Ambiguous or empty case
    #     return self


# ---- PROMPT ----
system_prompt = """
You are an expert ecommerce assistant. Your job is to extract structured search parameters for **buying products** or **returning orders** from a user's query, using the current query and chat history.

### Your tasks:
1. Determine if the query is about "buy" or "return".
2. Build an enhanced_query: a standalone, clear summary of the user's intent (using chat_history if available).
3. Extract parameters based on query type.

### For "buy" queries:
- query_type: "buy"
- buy_info:
  - product_name: main product category or name (e.g., "birthday gifts", "jeans pants")
  - attributes:
      - color: preferred color if mentioned (e.g., "red", "blue")
      - gender: intended gender (male/female/unisex)
      - age: numeric age if specified (e.g., "5" for "5-year-old")
      - age_group: derived age category ('kids', 'teen', 'adult', 'senior')
      - size: product size (XS, S, M, L, XL, XXL)

### For "return" queries:
- query_type: "return"
- return_info:
  - order_id: extracted if mentioned
  - product_name: extracted if mentioned
  - order_date: extracted if relative date is mentioned (e.g., "2 weeks ago")
  - attributes: details of returned product (color, gender, size, age)
  - return_action:
      - "return_only": if just returning
      - "replace": if returning and requesting a replacement
      - "refund": if returning and requesting a refund
  - replacement_product: if "replace", include:
      - product_name
      - attributes: color, gender, size, age of the new product

### Rules:
- Use chat_history to fill missing details (e.g., if user says "that order").
- If any value is not mentioned, set it to null.
- Never hallucinate product names/order IDs.
- Always output the JSON structure.

### Output Format:
Return a single JSON object following this schema:
{format_instructions}
"""

# ---- Perception Chain ----
def get_perception_chain(default_llm=default_llm) -> RunnableSequence:
    parser = PydanticOutputParser(pydantic_object=EntitySearch)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "The user query is: {user_query}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | default_llm.chat_model | parser


# ---- Processing Queries ----
def process_entity_query(
    user_query: str,
    conversation_memory: ConversationMemory = None,
    perception_chain: RunnableSequence = None,
) -> EntitySearch:
    if conversation_memory is None:
        conversation_memory = ConversationMemory()

    if perception_chain is None:
        perception_chain = get_perception_chain()

    chat_history = conversation_memory.get_langchain_messages()
    conversation_memory.add_human_message(user_query)
    result = perception_chain.invoke({"user_query": user_query, "chat_history": chat_history})
    conversation_memory.add_ai_message(result.model_dump())
    conversation_memory.save()

    return result


# ---- Testing ----
def test_entity_chain(queries: list[str], conversation_id: str = "ecom-test") -> list[EntitySearch]:
    memory = ConversationMemory(conversation_id=conversation_id)
    perception_chain = get_perception_chain()

    results = []  # ✅ Collect results in a list

    for query in queries:
        print(f"\nQuery: {query}")
        result = process_entity_query(query, memory, perception_chain)
        print("Result:", json.dumps(result.model_dump(), indent=2))
        results.append(result)  # ✅ Store the result

    return results  # ✅ Return results list


if __name__ == "__main__":
    test_queries = [
        # return
        "I want to return an order.",
        "That order which I ordered 2 weeks back.",
        "I want to return order ID 98765, the blue jeans size L I bought for my dad",
        "I want to return my size M red shoes and get a refund",
        "I want to return my jeans and replace them with black chinos size L",

        # neutral
        "I ordered a shoe.",
        "The orderID is 45363.",

        #buy
        "I need pink shoes for my 5-year-old daughter",
        "I want to buy a blue shirt for men",
        # "I want to explore birthday gifts.",
        # "I want to buy pens.",
        # "I need to look at jeans pants.",
        # "I am looking for half sleeve shirts.",
        # "I want to buy a red t-shirt size M for my 12-year-old son",
    ]
    results = test_entity_chain(test_queries, "ecom-session")
    CHECK = 1
