from pydantic import BaseModel, Field
from typing import Literal, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence
from llm_provider import default_llm
from memory import ConversationMemory
import json


class FlightSearch(BaseModel):
    """Flight search parameters"""

    departure_id: str = Field(..., description="The IATA code of the departure airport")
    arrival_id: str = Field(..., description="The IATA code of the arrival airport")


class HotelSearch(BaseModel):
    """Hotel search parameters"""

    location: str = Field(..., description="The location of the hotel")
    adults: int = Field(1, description="The number of adults")


class TravelSearch(BaseModel):
    """Travel search parameters that can include flight, hotel, or both"""

    search_type: Literal["flight", "hotel", "combined"] = Field(
        ...,
        description="Type of search to perform (flight, hotel, or combined for both)",
    )
    # Common fields
    currency: str = Field(
        "USD", description="The currency of the prices (USD, EUR, GBP, etc.)"
    )
    start_date: str = Field(
        ...,
        description="The start date in YYYY-MM-DD format (outbound flperception.pyight or check-in date)",
    )
    end_date: str = Field(
        None,
        description="The end date in YYYY-MM-DD format (return flight or check-out date)",
    )

    # Specific search components
    flight: Optional[FlightSearch] = Field(
        None,
        description="Flight search details, required for flight or combined searches",
    )
    hotel: Optional[HotelSearch] = Field(
        None,
        description="Hotel search details, required for hotel or combined searches",
    )

    enhanced_query: Optional[str] = Field(
        None,
        description="A standalone question that captures the user's intent and basis of which the search parameters are derived",
    )


system_prompt = """
You are an expert travel assistant. Your job is to extract structured search parameters for flight and hotel searches from a user's query, using both the current query and the chat_history for context.

**Your tasks:**
1. Analyze the chat_history and the current user query to determine the user's intent.
2. Build an enhanced_query: a standalone, unambiguous question that summarizes the user's intent, using chat_history to fill in any missing details.
3. Extract the following parameters as a JSON object:
    - search_type: "flight", "hotel", or "combined"
    - currency: The currency for prices (e.g., USD, EUR, GBP)
    - start_date: Outbound flight or hotel check-in date (YYYY-MM-DD)
    - end_date: Return flight or hotel check-out date (YYYY-MM-DD)
    - enhanced_query: The standalone question you constructed

    For flight or combined searches, include a "flight" object:
        - departure_id: IATA code of the departure airport (e.g., JFK)
        - arrival_id: IATA code of the arrival airport

    For hotel or combined searches, include a "hotel" object:
        - location: Hotel location (city or area)
        - adults: Number of adults

**Rules:**
- Use chat_history to resolve ambiguity or fill in missing information. If information is not available, set the value to null or "unknown".
- Only set search_type to "combined" if the user explicitly requests both a flight and a hotel in the same query. If the request is inferred from chat_history, set search_type to "flight" or "hotel" as appropriate.
- Always return all required fields in the output, even if some are null or "unknown".
- Do not guess or hallucinate values. If a value cannot be determined, use null or "unknown".
- If the user's intent is unclear, set ambiguous fields to "unknown" and explain the ambiguity in the enhanced_query.
- search_type can only be combined if the user explicitly asks for both a flight and hotel in the original user_query.
- Always prioritize the user's intent for search_type mentioned in the user_query and then reconsolidate the enhanced_query based on the chat_history.


**Output Format:**
Return a single JSON object following this schema:
{format_instructions}

**Historical Context:**
Always refer to chat_history to improve the current search if relevant or ambiguous. Use context from previous searches to fill in missing information and to improve the enhanced_query.
"""


def get_perception_chain(default_llm=default_llm) -> RunnableSequence:
    """
    Creates and returns the perception chain for extracting travel search parameters.

    Args:
        default_llm: The default LLM to use for the chain.

    Returns:
        A LangChain runnable sequence that takes a user query and chat history,
        and returns a TravelSearch object.
    """

    # Set up a parser
    parser = PydanticOutputParser(pydantic_object=TravelSearch)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "The user query is: {user_query}"),
        ]
    )

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # Create and return the chain
    return prompt | default_llm.chat_model | parser


def process_travel_query(
    user_query: str,
    conversation_memory: ConversationMemory = None,
    perception_chain: RunnableSequence = None,
) -> TravelSearch:
    """
    Process a travel query using conversation history for context.

    Args:
        user_query: The user's query about travel.
        conversation_memory: Optional ConversationMemory object with chat history.
        perception_chain: Optional perception chain to use. If None, a new one is created.

    Returns:
        A TravelSearch object with extracted parameters.
    """
    # Initialize a new conversation if one wasn't provided
    if conversation_memory is None:
        conversation_memory = ConversationMemory()

    # Use provided chain or create a new one
    if perception_chain is None:
        perception_chain = get_perception_chain()

    # Get chat history in the format expected by LangChain
    chat_history = conversation_memory.get_langchain_messages()

    # Debug information
    print(f"Processing query: {user_query}")
    if chat_history:
        print(f"Chat history length: {len(chat_history)} messages")
    else:
        print("No chat history yet")

    # Add the current query to memory BEFORE processing
    conversation_memory.add_human_message(user_query)

    # Process the query with the chain
    result = perception_chain.invoke(
        {"user_query": user_query, "chat_history": chat_history}
    )

    # Store the result directly using model_dump()
    conversation_memory.add_ai_message(result.model_dump())

    # Save the conversation
    conversation_memory.save()

    return result


def test_perception_chain(
    queries: list[str], conversation_id: str = "travel-test"
) -> None:
    """
    Test the perception chain with a list of queries.

    Args:
        queries: List of queries to process in sequence.
        conversation_id: Optional ID for the conversation memory.
    """
    # Create a memory object and perception chain
    memory = ConversationMemory(conversation_id=conversation_id)
    perception_chain = get_perception_chain()

    results = []

    # Process each query in sequence
    for i, query in enumerate(queries, 1):
        print(f"\n=== Example {i}: {query[:50]}{'...' if len(query) > 50 else ''} ===")
        result = process_travel_query(query, memory, perception_chain)
        results.append(result)
        print(f"Result: {result.model_dump()}\n")

    # Print the conversation history
    print("\n=== Conversation History ===")
    msg_tuples = memory.get_message_tuples()
    print("Number of messages:", len(msg_tuples))

    for i, (role, content) in enumerate(msg_tuples, 1):
        if role == "human":
            print(f"\nHuman #{i}: {content}")
        elif role == "ai":
            # Try to parse the AI response as JSON
            try:
                response_data = json.loads(content)
                search_type = response_data.get("search_type", "unknown")
                print(f"AI Response #{i}: {search_type} search")

                if search_type == "flight" or search_type == "combined":
                    flight_data = response_data.get("flight", {})
                    if flight_data:
                        print(
                            f"  Flight: {flight_data.get('departure_id')} → {flight_data.get('arrival_id')}"
                        )
                        print(
                            f"  Dates: {response_data.get('start_date')} to {response_data.get('end_date')}"
                        )

                if search_type == "hotel" or search_type == "combined":
                    hotel_data = response_data.get("hotel", {})
                    if hotel_data:
                        print(f"  Hotel Location: {hotel_data.get('location')}")
                        print(
                            f"  Dates: {response_data.get('start_date')} to {response_data.get('end_date')}"
                        )
                        print(f"  Adults: {hotel_data.get('adults')}")
            except json.JSONDecodeError:
                # If not valid JSON, just print the first 100 characters
                print(
                    f"AI Response #{i}: {content[:100]}..."
                    if len(content) > 100
                    else f"AI Response #{i}: {content}"
                )

    return results


if __name__ == "__main__":
    # Demo queries to test the perception chain
    test_queries = [
        "I want to search for flights from New York to Los Angeles on 2025-05-01 with return on 2025-05-05",
        # "I need a hotel in Los Angeles for 2 adults from May 1 to May 5, 2025",
        # "I want to search for flights from New York to Los Angeles on 2025-05-01 "
        # "with return on 2025-05-05 and a hotel for 2 adults in Los Angeles for the same dates",
    ]

    # Run the test
    results = test_perception_chain(test_queries, "travel-session")
    check=1