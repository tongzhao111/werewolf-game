import asyncio
import os
import sys
import gc

from pydantic import BaseModel, Field

# ==================== 屏蔽 httpcore2 的已知清理 bug ====================
_TARGET_MSG = "generator didn't stop after athrow"

def _is_noise(exc):
    return isinstance(exc, RuntimeError) and _TARGET_MSG in str(exc)

def _unraisable_hook(u):
    if _is_noise(u.exc_value):
        return
    sys.__unraisablehook__(u)

sys.unraisablehook = _unraisable_hook

def _loop_exception_handler(loop, context):
    exc = context.get("exception")
    if exc is not None and _is_noise(exc):
        return
    loop.default_exception_handler(context)
# =====================================================================

from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from agentscope.message import UserMsg


# ★ 定义结构化输出的数据模型
class VoteDecision(BaseModel):
    """投票决策的输出格式"""
    target: str = Field(description="你要投票淘汰的玩家名字")
    reason: str = Field(description="你投票给他的简要理由，一句话")


def create_agent(name, system_prompt):
    return Agent(
        name=name,
        system_prompt=system_prompt,
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(
                api_key=os.environ["DEEPSEEK_API_KEY"]
            ),
            model="deepseek-chat",
        ),
    )


async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_loop_exception_handler)

    caocao = create_agent(
        "曹操",
        "你是曹操，三国时期的枭雄，说话霸气、多疑。回答要简短。",
    )
    liubei = create_agent(
        "刘备",
        "你是刘备，三国时期的仁君，说话温和、谦逊。回答要简短。",
    )
    sunquan = create_agent(
        "孙权",
        "你是孙权，三国时期东吴之主，说话沉稳、果断。回答要简短。",
    )

    players = [caocao, liubei, sunquan]
    player_names = [a.name for a in players]

    print("=" * 60)
    print(f"【投票阶段】存活玩家：{player_names}")
    print("=" * 60)

    # 让每个 Agent 输出结构化的投票决策
    votes = []
    for agent in players:
        print(f"\n--- {agent.name} 正在思考 ---")
        reply = await agent.reply(
            UserMsg(
                name="主持人",
                content=f"请投票淘汰一名玩家。候选：{[n for n in player_names if n != agent.name]}",
            ),
            structured_schema=VoteDecision,  # ★ 指定结构化模型
        )
        # 从 reply 中取出结构化数据
        decision = reply.structured_output
        votes.append((agent.name, decision))
        print(f"{agent.name} 的投票：{decision}")

    # 统计票数
    print("\n" + "=" * 60)
    print("【投票统计】")
    print("=" * 60)

    tally = {}
    for voter, decision in votes:
        target = decision["target"] if isinstance(decision, dict) else decision.target
        tally[target] = tally.get(target, 0) + 1

    for target, count in tally.items():
        print(f"{target}: {count} 票")

    # 找出得票最高的
    winner = max(tally, key=tally.get)
    print(f"\n淘汰：{winner}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        gc.collect()
        gc.collect()