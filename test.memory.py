# test_memory.py
from app.agent import build_agent
from app.config import validate_config

if __name__ == "__main__":
    validate_config()
    agent = build_agent()

    session_cfg = {"configurable": {"session_id": "demo"}}

    r1 = agent.invoke({"input": "Je m'appelle Diallo. Retien mon nom.", "hint": ""}, config=session_cfg)
    print("Tour 1:", r1.get("output") or r1)

    r2 = agent.invoke({"input": "Comment je m'appelle ?", "hint": ""}, config=session_cfg)
    print("Tour 2:", r2.get("output") or r2)
