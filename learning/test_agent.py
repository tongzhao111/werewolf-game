import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from agentscope.message import UserMsg


async def main() -> None:
    agent = Agent(
        name="Assistant",
        system_prompt="你是一个乐于助人的助手，回答要简洁。",
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(
                api_key=os.environ["DEEPSEEK_API_KEY"]
            ),
            model="deepseek-chat",
        ),
    )

    user_msg = UserMsg(name="User", content="你好，请用一句话介绍你自己。")
    reply_msg = await agent.reply(user_msg)

    if reply_msg.content:
        print("智能体回复：", reply_msg.content[0].text)


asyncio.run(main())