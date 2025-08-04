from return_perception import get_return_chain, extract_return_details
from memory import ConversationMemory

class ReturnAgent:
    def __init__(self):
        self.perception_chain = get_return_chain()

    def handle(self, query: str):
        print("[ReturnAgent] Processing return query...")
        result = extract_return_details(query, self.perception_chain)
        print("[ReturnAgent] Parsed Details:", result.model_dump())

        # Simulated return processing
        if result.return_action == "refund":
            print(f"[ReturnAgent] Initiating refund for Order ID: {result.order_id}")
        elif result.return_action == "replace":
            print(f"[ReturnAgent] Replacing {result.product_name} with {result.replacement_product.product_name}")
        else:
            print(f"[ReturnAgent] Processing return for Order ID: {result.order_id}")
        return result

if __name__ == "__main__":
    user_query_list = test_queries = [
        # return
        "I want to return an order.",
        "That order which I ordered 2 weeks back.",
        "I want to return order ID 98765, the blue jeans size L I bought for my dad",
        "I want to return my size M red shoes and get a refund",
        "I want to return my jeans and replace them with black chinos size L",

        # neutral
        "I ordered a shoe.",
        "The orderID is 45363.",
    ]

    agent = ReturnAgent()
    perception_list = []
    for query in user_query_list:
        perception_list.append({"query": query, "result": agent.handle(query)})
    chk=1
