from orchestrator_perception import get_orchestrator_chain, classify_query
from buy_agent import BuyAgent
from return_agent import ReturnAgent

def main():
    print("E-commerce Orchestrator Running...")
    perception_chain = get_orchestrator_chain()

    while True:
        user_query = input("\nUser: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Step 1: Orchestrator perception - classify query type
        classification = classify_query(user_query, perception_chain)

        query_type = classification.query_type
        print(f"[Orchestrator] Query Type: {query_type}")

        # Step 2: Route to appropriate agent
        if query_type == "buy":
            agent = BuyAgent()
            agent.handle(user_query)
        elif query_type == "return":
            agent = ReturnAgent()
            agent.handle(user_query)
        else:
            print("[Orchestrator] Unable to classify query. Please clarify your request.")

if __name__ == "__main__":
    main()
