"""
三国狼人杀 · Streamlit 可视化界面
运行：streamlit run app.py
"""

import streamlit as st
import asyncio
import os
import sys
import gc
import re

# 确保能 import 同目录的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from werewolf_game import (
    WerewolfGame,
    _loop_exception_handler,
)


# ==================== 沉默日志器 ====================
# 把原版 GameLogger 的 print 改成"只收集不打印"，防止 Streamlit 后台刷屏
class SilentLogger:
    def __init__(self, game_index):
        self.game_index = game_index
        self.lines = []

    def log(self, text=""):
        self.lines.append(str(text))

    def save(self, directory="game_logs"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"game_{self.game_index:03d}.txt")
        with open(path, "w", encoding="utf-8", errors="ignore") as f:
            f.write("\n".join(self.lines))
        return path


import werewolf_game
werewolf_game.GameLogger = SilentLogger


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="三国狼人杀",
    page_icon="🐺",
    layout="wide",
)

st.title("🐺 三国狼人杀 · 多智能体对战")
st.caption("基于 AgentScope 2.0 + DeepSeek · AI 自动博弈")


# ==================== 日志渲染 ====================
def render_log(log_lines):
    """把日志渲染成漂亮的对话流"""
    for line in log_lines:
        line = line.strip()
        if not line:
            continue

        # 玩家发言：[名字]: 内容
        match = re.match(r'^\[([^\]]+)\]: (.+)$', line)
        if match:
            speaker, content = match.groups()
            is_human = speaker == "你"
            with st.chat_message(
                "user" if is_human else "assistant",
                avatar="🧑" if is_human else "🤖",
            ):
                st.markdown(f"**{speaker}**\n\n{content}")
            continue

        # 分隔线
        if line.startswith("===") or line.startswith("###"):
            st.divider()
            continue

        # 各种事件
        if line.startswith("🎮"):
            st.subheader(line)
        elif line.startswith("🐺 狼人阵营胜利"):
            st.error(line)
        elif line.startswith("🎉 好人阵营胜利"):
            st.success(line)
        elif line.startswith("🔮"):
            st.info(line)
        elif line.startswith("🌙"):
            st.info(line)
        elif line.startswith("☀️"):
            st.info(line)
        elif line.startswith("🩸") or line.startswith("💀"):
            st.warning(line)
        elif line.startswith("🧪") or line.startswith("☠️"):
            st.info(line)
        elif line.startswith("🛡️"):
            st.info(line)
        elif line.startswith("🗳️"):
            st.markdown(f"**{line}**")
        elif line.startswith("⚖️"):
            st.markdown(f"**{line}**")
        elif line.startswith("💬"):
            st.markdown(f"*{line}*")
        elif line.startswith("   "):
            # 缩进行（子事件）用灰色小字
            st.caption(line)
        else:
            st.text(line)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("⚙️ 游戏设置")
    num_games = st.slider("对局数", 1, 10, 1, help="跑多局看胜率，1 局约 3-8 分钟")

    st.divider()
    st.markdown("### 🎭 角色配置")
    st.markdown(
        """
        | 角色 | 数量 | 能力 |
        |------|------|------|
        | 🐺 狼人 | 2 | 夜晚击杀 |
        | 🔮 预言家 | 1 | 每晚查验 |
        | 🧪 女巫 | 1 | 解药/毒药 |
        | 🛡️ 守卫 | 1 | 每晚守护 |
        | 👤 村民 | 3 | 推理投票 |
        """
    )

    st.divider()
    start = st.button("🎮 开始游戏", type="primary", use_container_width=True)


# ==================== 检查 API Key ====================
if "DEEPSEEK_API_KEY" not in os.environ:
    st.error("❌ 未检测到 DEEPSEEK_API_KEY 环境变量")
    st.code('export DEEPSEEK_API_KEY="sk-你的key"', language="bash")
    st.stop()


# ==================== 运行游戏 ====================
if start:
    st.session_state.pop("results", None)

    progress_bar = st.progress(0, text="初始化...")

    player_names = [
        "刘备", "曹操", "孙权", "诸葛亮",
        "司马懿", "周瑜", "张飞", "关羽",
    ]

    async def run_one_game(idx):
        game = WerewolfGame(player_names, game_index=idx)
        result = await game.run_single_game()
        return result, list(game.logger.lines)

    async def run_all():
        results = []
        for i in range(1, num_games + 1):
            progress_bar.progress(
                int((i - 1) / num_games * 100),
                text=f"第 {i}/{num_games} 局进行中...（每局约 3-8 分钟）",
            )
            result, log_lines = await run_one_game(i)
            results.append({
                "index": i,
                "result": result,
                "log_lines": log_lines,
            })
        return results

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.set_exception_handler(_loop_exception_handler)
    try:
        results = loop.run_until_complete(run_all())
    finally:
        loop.close()
        gc.collect()

    progress_bar.progress(100, text="✅ 全部完成！")
    st.session_state["results"] = results


# ==================== 展示结果 ====================
if "results" in st.session_state:
    results = st.session_state["results"]

    # 统计概览
    st.divider()
    st.header("📊 对局统计")

    total = len(results)
    wolf_wins = sum(1 for r in results if r["result"]["winner"] == "werewolves")
    good_wins = total - wolf_wins
    avg_rounds = sum(r["result"]["rounds"] for r in results) / total

    col1, col2, col3 = st.columns(3)
    col1.metric("🐺 狼人胜", f"{wolf_wins} 局", f"{wolf_wins/total*100:.0f}%")
    col2.metric("🎉 好人胜", f"{good_wins} 局", f"{good_wins/total*100:.0f}%")
    col3.metric("⚡ 平均回合", f"{avg_rounds:.1f}")

    # 每局日志
    st.divider()
    st.header("📜 对局日志")

    if total == 1:
        render_log(results[0]["log_lines"])
    else:
        tabs = st.tabs([f"第 {r['index']} 局" for r in results])
        for tab, r in zip(tabs, results):
            with tab:
                render_log(r["log_lines"])