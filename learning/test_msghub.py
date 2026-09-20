import asyncio
import os
import sys
import gc

from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from agentscope.message import UserMsg, AssistantMsg

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


def create_agent(name, system_prompt):
    """创建一个配置了 DeepSeek 模型的 Agent"""
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


async def broadcast(participants, msg, exclude=None):
    """把消息发给群里的其他人（写入各自上下文，不触发回复）。

    AgentScope 2.0 已移除 `MsgHub`，这里用 `Agent.observe()` 实现等价的
    「广播」效果：消息只会被存入上下文，等 agent 下一次 `reply` 时看到。
    """
    for agent in participants:
        if agent is exclude:
            continue
        await agent.observe(msg)


async def speak(agent, participants, prompt):
    """让某个 agent 发言，并把发言广播给群里其他参与者。

    Args:
        agent: 当前发言者。
        participants: 当前群成员列表。
        prompt: 触发发言的内容（以 system 身份提问）。
    """
    reply = await agent.reply(UserMsg(name="system", content=prompt))
    text = reply.get_text_content() or ""
    print(f"\n[{agent.name}]: {text}")
    await broadcast(participants, reply, exclude=agent)
    return reply


async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_loop_exception_handler)

    # 创建三个角色
    caocao = create_agent(
        "曹操",
        "你是曹操，三国时期的枭雄，说话霸气、多疑，喜欢用反问句。回答要简短，一两句话。",
    )
    liubei = create_agent(
        "刘备",
        "你是刘备，三国时期的仁君，说话温和、谦逊，喜欢谈仁义。回答要简短，一两句话。",
    )
    sunquan = create_agent(
        "孙权",
        "你是孙权，三国时期东吴之主，说话沉稳、果断，善于权衡利弊。回答要简短，一两句话。",
    )

    participants = [caocao, liubei, sunquan]

    # ==================== 核心：模拟 MsgHub 的群聊 ====================
    print("=" * 60)
    print("【群聊开始】")
    print("=" * 60)

    # 群公告：所有参与者都会收到这条消息
    # 注意：observe() 只接受 role 为 user/assistant 的消息，所以这里用 UserMsg
    await broadcast(
        participants,
        UserMsg(
            name="system",
            content="现在你们三人聚在一起，请依次介绍一下自己，"
            "并谈谈对当前天下大势的看法。",
        ),
    )

    # 顺序发言：三人依次发言，每人的发言自动广播给其他人
    for agent in participants:
        await speak(
            agent,
            participants,
            "轮到你自我介绍了，请介绍一下自己，并谈谈对当前天下大势的看法。",
        )

    # ---- 动态管理参与者示例 ----
    print("\n" + "=" * 60)
    print("【动态管理】孙权离开了群聊...")
    print("=" * 60)

    # 从群聊中移除孙权
    participants.remove(sunquan)

    # 模拟孙权离开的消息，广播给剩下的曹操和刘备
    await broadcast(
        participants,
        AssistantMsg(name="孙权", content="我有事先走了，你们继续聊。"),
    )

    # 曹操和刘备继续对话
    print("\n" + "=" * 60)
    print("【继续对话】曹操和刘备继续交谈...")
    print("=" * 60)

    await speak(caocao, participants, "刘备，你怎么看孙权刚才的话？")
    await speak(liubei, participants, "曹操，你觉得孙权为何突然离开？")
    # ====================================================================

    print("\n" + "=" * 60)
    print("【群聊结束】")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        gc.collect()
        gc.collect()
