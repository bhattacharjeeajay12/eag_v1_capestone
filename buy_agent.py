from buy_perception import get_buy_chain, extract_buy_details

class BuyAgent:
    def __init__(self):
        self.perception_chain = get_buy_chain()

    def handle(self, query: str):
        print("[BuyAgent] Processing buy query...")
        result = extract_buy_details(query, self.perception_chain)
        print("[BuyAgent] Parsed Details:", result.model_dump())

        # Simulated product search
        print(f"[BuyAgent] Searching for {result.product_name} with attributes {result.attributes}")
        # TODO: Integrate product catalog API here
        return result

if __name__ == "__main__":
    user_query_list = test_queries = [
        # buy
        "I need pink shoes for my 5-year-old daughter",
        "I want to buy a blue shirt for men",
        "I want to explore birthday gifts.",
        "I want to buy pens.",
        "I need to look at jeans pants.",
        "I am looking for half sleeve shirts.",
        "I want to buy a red t-shirt size M for my 12-year-old son",
    ]

    agent = BuyAgent()
    perception_list = []
    for query in user_query_list:
        perception_list.append({"query": query, "result": agent.handle(query)})
    chk = 1