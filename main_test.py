from orchestrator_perception import get_orchestrator_chain, classify_query
from buy_agent import BuyAgent
from return_agent import ReturnAgent

def main(ques_list):
    print("E-commerce Orchestrator Running...")
    perception_chain = get_orchestrator_chain()

    # while True:
    results = []
    for user_query in ques_list:
        # user_query = input("\nUser: ").strip()
        result = {}
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Step 1: Orchestrator perception - classify query type
        classification = classify_query(user_query, perception_chain)
        result["classification"] = classification.query_type

        query_type = classification.query_type
        print(f"[Orchestrator] Query Type: {query_type}")

        # Step 2: Route to appropriate agent
        if query_type == "buy":
            agent = BuyAgent()
            perception_result = agent.handle(user_query)
        elif query_type == "return":
            agent = ReturnAgent()
            perception_result = agent.handle(user_query)
        else:
            print("[Orchestrator] Unable to classify query. Please clarify your request.")
            perception_result = None
        result["perception"] = perception_result
        results.append(result)
    return results

if __name__ == "__main__":
    ques_list = [
        "I want to return the shows which I returned last week",
        "exit"
    ]
    perception_result = main(ques_list)
    chk = 1
