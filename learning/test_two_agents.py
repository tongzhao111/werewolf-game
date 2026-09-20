import asyncio
import os


from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from agentscope.message import UserMsg



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


async def main():
    # 创建两个角色
    caocao = create_agent(
        "曹操",
        "你是曹操，三国时期的枭雄，说话霸气、多疑，喜欢用反问句。回答要简短，一两句话。",
    )
    liubei = create_agent(
        "刘备",
        "你是刘备，三国时期的仁君，说话温和、谦逊，喜欢谈仁义。回答要简短，一两句话。",
    )

    # 第 1 轮：曹操先发言
    print("=" * 50)
    print("【主持人】：曹操，你怎么看当前的天下大势？")
    reply1 = await caocao.reply(
        UserMsg(name="主持人", content="曹操，你怎么看当前的天下大势？")
    )
    print(f"\n【曹操】: {reply1.content[0].text}")

    # 第 2 轮：把曹操的话传给刘备
    print("\n" + "=" * 50)
    print("【主持人】：刘备，你怎么回应曹操？")
    reply2 = await liubei.reply(
        UserMsg(name="曹操", content=reply1.content[0].text)
    )
    print(f"\n【刘备】: {reply2.content[0].text}")

    # 第 3 轮：曹操再回应
    print("\n" + "=" * 50)
    print("【主持人】：曹操，你怎么回应刘备？")
    reply3 = await caocao.reply(
        UserMsg(name="刘备", content=reply2.content[0].text)
    )
    print(f"\n【曹操】: {reply3.content[0].text}")



asyncio.run(main())