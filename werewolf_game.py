import asyncio
import os
import sys
import gc
import random
from typing import Optional

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


# ==================== 结构化输出模型 ====================
class VoteDecision(BaseModel):
    target: str = Field(description="你要投票淘汰的玩家名字")
    reason: str = Field(description="简要理由，一句话")

class CheckDecision(BaseModel):
    target: str = Field(description="你要查验身份的玩家名字")

class GuardAction(BaseModel):
    target: str = Field(description="你要守护的玩家名字")

class WitchAction(BaseModel):
    use_antidote: bool = Field(
        description="是否使用解药救活今晚被狼人杀害的玩家。若解药已用或没人被杀，请填 false。"
    )
    use_poison: bool = Field(
        description="是否使用毒药毒死一名玩家。若毒药已用，请填 false。"
    )
    poison_target: Optional[str] = Field(
        description="毒药目标玩家名字。不用毒药请填 None。",
        default=None,
    )


# ==================== 日志 ====================
class GameLogger:
    def __init__(self, game_index):
        self.game_index = game_index
        self.lines = []
        self._print = print

    def log(self, text=""):
        self.lines.append(str(text))
        self._print(text)

    def save(self, directory="game_logs"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"game_{self.game_index:03d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        return path


# ==================== 容错 ====================
async def safe_reply(agent, user_msg, structured_schema=None, logger=None):
    try:
        if structured_schema is not None:
            return await agent.reply(user_msg, structured_schema=structured_schema)
        return await agent.reply(user_msg)
    except Exception as e:
        err_msg = f"   ⚠️ {agent.name} 调用失败：{type(e).__name__}: {e}"
        if logger:
            logger.log(err_msg)
        else:
            print(err_msg)
        return None


def _extract_text(reply):
    if reply is None or not reply.content:
        return None
    first = reply.content[0]
    return first.text if hasattr(first, "text") else str(first)


def _extract_structured(reply):
    if reply is None:
        return None
    return reply.structured_output


# ==================== Agent 创建 ====================
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


def build_role_prompt(name, role, all_names):
    base = f"你是{name}。在这场三国狼人杀中你的身份是【{role}】。\n"

    if role == "狼人":
        base += (
            "你是狼人阵营，目标是消灭所有好人。\n"
            "夜晚你可以与其他狼人（如果有）协商击杀一名玩家。\n"
            "白天你要伪装成好人，误导大家投票，不要暴露身份。\n"
            "【狼人战术】\n"
            "- 如果有人跳预言家指认你，你可以反驳他是'悍跳'，或者反咬他是狼。\n"
            "- 优先和另一个狼人配合，一唱一和分散好人注意力。\n"
        )

    elif role == "预言家":
        base += (
            "你是预言家，属于好人阵营。\n"
            "每晚你可以查验一名玩家的身份（狼人/好人）。\n"
            "★ 关键策略：\n"
            "1. 如果你第一晚查验到【狼人】，第二天必须立即跳预言家大声宣布："
            "'我是预言家，昨夜我查验了XXX，他是狼人！请大家跟我一起投他。'\n"
            "2. 如果你第一晚查验到【好人】，第二天可以视情况决定是否跳。\n"
            "3. 【绝对禁止】不要说出'昨夜狼人刀的是XXX'——你不可能知道狼人刀谁。\n"
            "4. 被反驳时要坚定立场：'我确实是预言家，请好人信任我。'\n"
            "5. 如果你被投票出局，遗言必须再次强调查验结果。\n"
        )


    elif role == "女巫":

        base += (

            "你是女巫，属于好人阵营。\n"

            "你有一瓶解药（救活一名被狼人杀害的玩家）和一瓶毒药（毒死一名玩家），各限一次。\n"

            "★ 关键策略：\n"

            "1. 第一晚通常建议使用解药，救活被杀的玩家。\n"

            "2. 使用解药后不要急着公开身份，先观察预言家有没有跳出来。\n"

            "3. 【绝对禁止】不要说出'昨夜狼人刀的是XXX'，否则会被当成狼人。\n"

            "4. 【毒药使用规则——非常重要】毒药是最后的武器，不是常规手段。\n"

            "   - 第 1 晚绝对不要使用毒药（没有任何信息，必然毒错）。\n"

            "   - 只有满足以下任一条件，才考虑使用毒药：\n"

            "     (a) 已有一个真预言家跳出来，明确指认某人是狼人。\n"

            "     (b) 你自己通过观察，非常确定某人是狼人（有具体证据）。\n"

            "   - 如果你没有 90% 以上的把握，宁可不用毒药，把毒药留到后面更关键的回合。\n"

            "   - 毒错一个好人 = 帮助狼人少杀一次，代价极高。\n"

            "5. 记住：不用毒药不是失败，用错毒药才是失误。\n"

        )


    elif role == "守卫":

        base += (

            "你是守卫，属于好人阵营。\n"

            "每晚你可以守护一名玩家，被守护的人当晚不会被狼人杀死。\n"

            "★ 关键策略（按优先级）：\n"

            "1. 【最高优先级】如果场上有人跳预言家，立刻守护他。\n"

            "   预言家是狼人的头号目标，多活一晚就多一次查验，价值极高。\n"

            "2. 【次高优先级】如果没人跳预言家，优先守护自己。\n"

            "   你活着才能持续守护别人；你死了，好人就少一层保险。\n"

            "3. 【第三优先级】守护那些'发言积极、明显是好人'的玩家。\n"

            "   例如：仔细分析局势、给出具体推理、被多人认可的人。\n"

            "4. 【规则提醒】不要连续两晚守护同一个人。\n"

            "   如果今晚想守护的人，昨晚已经守护过，就改守自己或别人。\n"

            "5. 如果你被投票出局，遗言可以说明你守护过谁，帮助好人推理。\n"

            "★ 核心原则：守卫的价值在于'挡掉狼刀'，不是随便猜。\n"

            "  当你不知道守谁时，守自己（至少保证自己活着）。\n"

        )

    else:  # 村民
        base += (
            "你是普通村民，属于好人阵营。\n"
            "★ 关键策略：\n"
            "1. 【不要盲从】不要因为某个人说得大声就跟着投票。要有自己的推理。\n"
            "2. 【识别真假预言家】看谁的跳预言家时机更自然、谁在积极给其他玩家'定身份'。\n"
            "3. 【观察狼人特征】以下行为高度可疑：急着带节奏投票、不提具体证据、"
            "谁在预言家跳出来后立刻反驳。\n"
            "4. 【发言要有依据】引用前面玩家的具体发言进行赞同或反驳。\n"
            "5. 如果不知道该投谁，投那个'发言最急切、最想带节奏'的人。\n"
        )

    base += (
        f"\n其他玩家：{all_names}。请保持{name}的性格说话，回答要简短（一两句话）。\n"
        "★ 重要：发言时请针对前面玩家说的话进行回应（赞同、反驳或补充），不要自说自话。"
    )
    return base


# ==================== 游戏主类 ====================
class WerewolfGame:
    def __init__(self, player_names, game_index=1):
        self.player_names = list(player_names)
        self.game_index = game_index
        self.logger = GameLogger(game_index)
        self.players = {}
        self.roles = {}
        self.alive = set(self.player_names)
        self.werewolves = set()
        self.seer = None
        self.witch = None
        self.guard = None
        self.antidote_available = True
        self.poison_available = True
        self.last_guarded = None  # ★ 守卫上一晚守护过谁
        self.round_num = 0
        self.last_words_history = []
        # 统计
        self.poison_hit_wolf = 0
        self.poison_hit_good = 0
        self.antidote_used = False
        self.poison_used = False
        self.guard_saved_night = 0  # ★ 守卫成功救人的次数

    def setup_roles(self, num_werewolves=2):
        """8 人局：2 狼 + 1 预言家 + 1 女巫 + 1 守卫 + 3 村民"""
        names = self.player_names.copy()
        random.shuffle(names)

        wolves = names[:num_werewolves]
        seer = names[num_werewolves]
        witch = names[num_werewolves + 1]
        guard = names[num_werewolves + 2]
        villagers = names[num_werewolves + 3:]

        self.werewolves = set(wolves)
        self.seer = seer
        self.witch = witch
        self.guard = guard
        self.roles = {}
        for w in wolves:
            self.roles[w] = "狼人"
        self.roles[seer] = "预言家"
        self.roles[witch] = "女巫"
        self.roles[guard] = "守卫"
        for v in villagers:
            self.roles[v] = "村民"

        for name in self.player_names:
            self.players[name] = create_agent(
                name,
                build_role_prompt(name, self.roles[name], self.player_names),
            )

        self.logger.log(f"\n🎮 角色分配完成（只有狼人知道自己是谁）")
        self.logger.log(
            f"   （调试信息）狼人：{self.werewolves}，预言家：{self.seer}，"
            f"女巫：{self.witch}，守卫：{self.guard}\n"
        )

    def check_game_over(self):
        wolves_alive = self.werewolves & self.alive
        good_alive = self.alive - self.werewolves

        if not wolves_alive:
            return "good"
        if len(wolves_alive) >= len(good_alive):
            return "werewolves"
        return None

    async def werewolf_phase(self):
        wolves_alive = [self.players[n] for n in self.alive if n in self.werewolves]
        if not wolves_alive:
            return None

        self.logger.log(f"\n🌙 夜晚：狼人正在行动...")

        kill_votes = []
        for wolf in wolves_alive:
            candidates = [n for n in self.alive if n not in self.werewolves]
            if not candidates:
                continue
            reply = await safe_reply(
                wolf,
                UserMsg(
                    name="系统",
                    content=f"你是狼人。请选择今晚要击杀的目标。候选：{candidates}",
                ),
                structured_schema=VoteDecision,
                logger=self.logger,
            )
            decision = _extract_structured(reply)
            if decision is None:
                target = random.choice(candidates)
                decision = {"target": target, "reason": "（API 错误，默认选择）"}
                self.logger.log(f"   ⚠️ {wolf.name} 未返回有效决策，默认击杀 {target}")
            kill_votes.append((wolf.name, decision))

        tally = {}
        for voter, decision in kill_votes:
            target = decision["target"] if isinstance(decision, dict) else decision.target
            if target in self.alive and target not in self.werewolves:
                tally[target] = tally.get(target, 0) + 1

        if not tally:
            return None

        killed = max(tally, key=tally.get)
        self.logger.log(f"   🩸 狼人决定击杀：{killed}")
        return killed

    async def seer_phase(self):
        if self.seer not in self.alive:
            return None

        seer_agent = self.players[self.seer]
        self.logger.log(f"\n🔮 预言家正在查验...")

        candidates = [n for n in self.alive if n != self.seer]
        if not candidates:
            return None

        reply = await safe_reply(
            seer_agent,
            UserMsg(
                name="系统",
                content=f"你是预言家。请选择今晚要查验的玩家。候选：{candidates}",
            ),
            structured_schema=CheckDecision,
            logger=self.logger,
        )
        raw = _extract_structured(reply)
        if raw is None:
            target = random.choice(candidates)
            self.logger.log(f"   ⚠️ 预言家未返回有效决策，默认查验 {target}")
        else:
            target = raw["target"] if isinstance(raw, dict) else raw.target

        if target not in self.alive:
            return None

        is_wolf = target in self.werewolves
        result = "狼人" if is_wolf else "好人"

        await safe_reply(
            seer_agent,
            UserMsg(
                name="主持人",
                content=f"你查验了 {target}，他的身份是【{result}】。请记住这个信息。",
            ),
            logger=self.logger,
        )
        self.logger.log(f"   🔮 预言家查验了 {target}（结果仅预言家知道）")
        return target

    async def guard_phase(self):
        """★ 守卫守护"""
        if self.guard not in self.alive:
            return None

        guard_agent = self.players[self.guard]
        self.logger.log(f"\n🛡️ 守卫正在守护...")

        # 不能连续两晚守护同一人
        candidates = [n for n in self.alive if n != self.last_guarded]
        if not candidates:
            candidates = list(self.alive)  # 兜底

        prompt = (
            f"你是守卫。请选择今晚要守护的玩家。候选：{candidates}"
        )
        if self.last_guarded:
            prompt += f"\n（注意：你上一晚守护了 {self.last_guarded}，规则不允许连续两晚守护同一人）"

        reply = await safe_reply(
            guard_agent,
            UserMsg(name="系统", content=prompt),
            structured_schema=GuardAction,
            logger=self.logger,
        )
        raw = _extract_structured(reply)
        if raw is None:
            target = random.choice(candidates)
            self.logger.log(f"   ⚠️ 守卫未返回有效决策，默认守护 {target}")
        else:
            target = raw["target"] if isinstance(raw, dict) else raw.target

        if target not in self.alive:
            return None

        self.last_guarded = target
        self.logger.log(f"   🛡️ 守卫守护了 {target}（仅守卫自己知道）")
        return target

    async def witch_phase(self, killed, guarded):
        """女巫行动：救人 或 毒人"""
        if self.witch not in self.alive:
            return killed, None

        witch_agent = self.players[self.witch]
        self.logger.log(f"\n🧪 女巫正在行动...")

        antidote_info = "解药还在" if self.antidote_available else "解药已用掉"
        poison_info = "毒药还在" if self.poison_available else "毒药已用掉"

        if killed:
            kill_info = f"今晚 {killed} 被狼人杀害了。"
        else:
            kill_info = "今晚没有人被狼人杀害。"

        candidates = list(self.alive)
        prompt = (
            f"你是女巫。{kill_info}\n"
            f"你的{antidote_info}，你的{poison_info}。\n"
            f"请决定：\n"
            f"- 是否使用解药救活 {killed}？（如解药已用、或没人被杀，请填 false）\n"
            f"- 是否使用毒药毒死一名玩家？候选：{candidates}（如毒药已用，请填 false）\n"
            f"注意：同一晚不能同时使用两瓶药。"
        )

        reply = await safe_reply(
            witch_agent,
            UserMsg(name="系统", content=prompt),
            structured_schema=WitchAction,
            logger=self.logger,
        )

        action = _extract_structured(reply)
        if action is None:
            self.logger.log(f"   ⚠️ 女巫未返回有效决策，默认不使用任何药")
            return killed, None

        if isinstance(action, dict):
            use_antidote = action.get("use_antidote", False)
            use_poison = action.get("use_poison", False)
            poison_target = action.get("poison_target")
        else:
            use_antidote = action.use_antidote
            use_poison = action.use_poison
            poison_target = action.poison_target

        if use_antidote and use_poison:
            self.logger.log(f"   ⚠️ 女巫试图同晚使用两瓶药，系统默认只用解药")
            use_poison = False

        final_killed = killed
        poisoned = None

        if use_antidote and self.antidote_available and killed:
            self.antidote_available = False
            self.antidote_used = True
            final_killed = None
            self.logger.log(f"   🧪 女巫使用解药，救活了 {killed}")
        elif use_poison and self.poison_available and poison_target and poison_target in self.alive:
            self.poison_available = False
            self.poison_used = True
            poisoned = poison_target
            if poison_target in self.werewolves:
                self.poison_hit_wolf += 1
            else:
                self.poison_hit_good += 1
            self.logger.log(f"   ☠️ 女巫使用毒药，毒死了 {poison_target}")
        else:
            self.logger.log(f"   💤 女巫没有使用任何药")

        return final_killed, poisoned

    async def day_phase(self, night_deaths=None):
        alive_agents = [self.players[n] for n in self.alive]
        alive_names = [a.name for a in alive_agents]

        if night_deaths:
            night_info = f"昨晚，{', '.join(night_deaths)} 死了。"
        else:
            night_info = "昨晚是平安夜，没有人死亡。"

        if self.last_words_history:
            last_words_text = "\n".join(
                [f"  【{name}（{role}）】的遗言：{content}"
                 for name, role, content in self.last_words_history]
            )
            last_words_section = f"\n此前的遗言记录：\n{last_words_text}\n"
        else:
            last_words_section = ""

        self.logger.log(f"\n☀️ 白天：全员讨论...（{night_info}）")

        # 讨论阶段
        speeches = []
        for agent in alive_agents:
            if speeches:
                context_lines = [f"  {speaker}：{content}" for speaker, content in speeches]
                context = "\n".join(context_lines)
                prompt = (
                    f"天亮了。{night_info}\n"
                    f"{last_words_section}"
                    f"你是{alive_names}中的一员，存活玩家：{alive_names}。\n"
                    f"前面玩家已经发言如下：\n{context}\n\n"
                    f"请你针对上面的发言发表看法（赞同/反驳/补充），并讨论谁是狼人。"
                )
            else:
                prompt = (
                    f"天亮了。{night_info}\n"
                    f"{last_words_section}"
                    f"你是第一个发言的人，存活玩家：{alive_names}。"
                    f"请发言讨论谁是狼人。"
                )

            reply = await safe_reply(
                agent, UserMsg(name="主持人", content=prompt), logger=self.logger
            )
            speech = _extract_text(reply) or "（因 API 错误未能发言）"

            self.logger.log(f"\n[{agent.name}]: {speech}")
            speeches.append((agent.name, speech))

        # 投票阶段
        self.logger.log(f"\n🗳️ 投票阶段...")

        full_discussion = "\n".join(
            [f"  {speaker}：{content}" for speaker, content in speeches]
        )

        votes = []
        for agent in alive_agents:
            candidates = [n for n in self.alive if n != agent.name]
            if not candidates:
                continue
            vote_prompt = (
                f"{night_info}\n"
                f"{last_words_section}"
                f"白天讨论内容如下：\n{full_discussion}\n\n"
                f"【投票指导】\n"
                f"- 如果你是好人：投那个发言最可疑、最想带节奏的人，不要盲从大多数。\n"
                f"- 如果你是狼人：投一个好人，尽量让票集中。\n"
                f"- 如果已有人跳预言家指认某人是狼，优先考虑相信他。\n\n"
                f"现在请投票淘汰一名玩家。候选：{candidates}"
            )
            reply = await safe_reply(
                agent,
                UserMsg(name="主持人", content=vote_prompt),
                structured_schema=VoteDecision,
                logger=self.logger,
            )
            decision = _extract_structured(reply)
            if decision is None:
                target = random.choice(candidates)
                decision = {"target": target, "reason": "（API 错误，默认选择）"}
                self.logger.log(f"   ⚠️ {agent.name} 未返回有效投票，默认投 {target}")
            votes.append((agent.name, decision))
            self.logger.log(f"   {agent.name} → {decision}")

        tally = {}
        for voter, decision in votes:
            target = decision["target"] if isinstance(decision, dict) else decision.target
            if target in self.alive:
                tally[target] = tally.get(target, 0) + 1

        if not tally:
            return None

        eliminated = max(tally, key=tally.get)
        self.logger.log(f"\n   ⚖️ 投票结果：{tally}")
        self.logger.log(f"   💀 {eliminated} 被淘汰")

        # 遗言
        if eliminated in self.alive:
            eliminated_agent = self.players[eliminated]
            eliminated_role = self.roles[eliminated]

            self.logger.log(f"\n💬 {eliminated} 发表遗言...")
            last_words_reply = await safe_reply(
                eliminated_agent,
                UserMsg(
                    name="主持人",
                    content=(
                        f"你被投票淘汰了。你的真实身份是【{eliminated_role}】。\n"
                        f"请留下你的遗言。如果你有掌握的信息，请务必公开帮助你的阵营。\n"
                        f"如果你是狼人，可以伪装好人混淆视听。\n"
                        f"发言要简短（一两句话），保持角色性格。"
                    ),
                ),
                logger=self.logger,
            )

            last_words = _extract_text(last_words_reply) or "（没有遗言）"
            self.logger.log(f"[{eliminated}（{eliminated_role}）]: {last_words}")
            self.last_words_history.append((eliminated, eliminated_role, last_words))

        return eliminated

    async def run_single_game(self):
        self.logger.log("=" * 60)
        self.logger.log(f"🎮 三国狼人杀（第 {self.game_index} 局，8 人局含守卫）")
        self.logger.log(f"   参与者：{self.player_names}")
        self.logger.log("=" * 60)

        self.setup_roles(num_werewolves=2)

        while True:
            self.round_num += 1
            self.logger.log(f"\n{'=' * 60}")
            self.logger.log(f"第 {self.round_num} 轮")
            self.logger.log(f"{'=' * 60}")

            # ★ 夜晚顺序：预言家 → 守卫 → 狼人 → 女巫
            await self.seer_phase()
            guarded = await self.guard_phase()
            killed = await self.werewolf_phase()
            final_killed, poisoned = await self.witch_phase(killed, guarded)

            # ★ 结算死亡：狼刀 + 守卫救人 + 女巫救人 + 女巫毒人
            night_deaths = []

            # 狼刀（如果没被守卫救、也没被女巫救）
            if final_killed and final_killed in self.alive:
                if final_killed == guarded:
                    self.guard_saved_night += 1
                    self.logger.log(f"   🛡️ 守卫成功保护了 {final_killed}，狼刀落空")
                else:
                    self.alive.discard(final_killed)
                    night_deaths.append(final_killed)

            # 女巫毒人（无视守卫）
            if poisoned and poisoned in self.alive:
                self.alive.discard(poisoned)
                night_deaths.append(poisoned)

            if night_deaths:
                self.logger.log(f"   💀 今晚死亡：{', '.join(night_deaths)}")
            else:
                self.logger.log(f"   ✨ 昨晚平安无事")

            result = self.check_game_over()
            if result:
                self._announce_winner(result)
                return self._build_result(result)

            eliminated = await self.day_phase(night_deaths=night_deaths)
            if eliminated and eliminated in self.alive:
                self.alive.discard(eliminated)

            result = self.check_game_over()
            if result:
                self._announce_winner(result)
                return self._build_result(result)

    def _announce_winner(self, result):
        self.logger.log(f"\n{'=' * 60}")
        if result == "good":
            self.logger.log("🎉 好人阵营胜利！所有狼人被淘汰。")
        else:
            self.logger.log("🐺 狼人阵营胜利！")
        self.logger.log(f"   狼人：{self.werewolves}")
        self.logger.log(f"   预言家：{self.seer}")
        self.logger.log(f"   女巫：{self.witch}")
        self.logger.log(f"   守卫：{self.guard}")
        self.logger.log(f"   剩余解药：{'有' if self.antidote_available else '已用'}")
        self.logger.log(f"   剩余毒药：{'有' if self.poison_available else '已用'}")
        self.logger.log(f"   守卫成功救人：{self.guard_saved_night} 次")
        self.logger.log("=" * 60)

    def _build_result(self, winner):
        return {
            "winner": winner,
            "rounds": self.round_num,
            "antidote_used": self.antidote_used,
            "poison_used": self.poison_used,
            "poison_hit_wolf": self.poison_hit_wolf,
            "poison_hit_good": self.poison_hit_good,
            "guard_saved": self.guard_saved_night,
        }


# ==================== 多局统计 ====================
class GameStats:
    def __init__(self):
        self.total = 0
        self.wolf_wins = 0
        self.good_wins = 0
        self.rounds_list = []
        self.antidote_used = 0
        self.poison_used = 0
        self.poison_hit_wolf = 0
        self.poison_hit_good = 0
        self.guard_saved = 0

    def record(self, result):
        self.total += 1
        if result["winner"] == "werewolves":
            self.wolf_wins += 1
        else:
            self.good_wins += 1
        self.rounds_list.append(result["rounds"])
        if result["antidote_used"]:
            self.antidote_used += 1
        if result["poison_used"]:
            self.poison_used += 1
        self.poison_hit_wolf += result.get("poison_hit_wolf", 0)
        self.poison_hit_good += result.get("poison_hit_good", 0)
        self.guard_saved += result.get("guard_saved", 0)

    def summary_text(self):
        lines = []
        lines.append("=" * 60)
        lines.append("📊 多局对战统计（8 人局 + 守卫）")
        lines.append("=" * 60)
        lines.append(f"总对局数：{self.total}")
        if self.total > 0:
            lines.append(f"狼人胜：{self.wolf_wins} 局（{self.wolf_wins/self.total*100:.1f}%）")
            lines.append(f"好人胜：{self.good_wins} 局（{self.good_wins/self.total*100:.1f}%）")
        if self.rounds_list:
            avg = sum(self.rounds_list) / len(self.rounds_list)
            lines.append(f"平均回合数：{avg:.2f}")
            lines.append(f"最短/最长回合：{min(self.rounds_list)} / {max(self.rounds_list)}")
        lines.append(f"女巫用解药的局数：{self.antidote_used}")
        lines.append(f"女巫用毒药的局数：{self.poison_used}")
        lines.append(f"女巫毒对狼：{self.poison_hit_wolf} 次")
        lines.append(f"女巫毒错好人：{self.poison_hit_good} 次")
        lines.append(f"守卫成功救人：{self.guard_saved} 次")
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_summary(self):
        print("\n\n" + self.summary_text())

    def save_summary(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.summary_text())


# ==================== 主入口 ====================
async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(_loop_exception_handler)

    NUM_GAMES = 5  # ★ 改这里可以跑更多局

    # ★ 8 人局
    player_names = ["刘备", "曹操", "孙权", "诸葛亮", "司马懿", "周瑜", "张飞", "关羽"]
    stats = GameStats()

    for i in range(1, NUM_GAMES + 1):
        print(f"\n\n{'#' * 60}")
        print(f"# 第 {i} / {NUM_GAMES} 局开始")
        print(f"{'#' * 60}")

        game = WerewolfGame(player_names, game_index=i)
        result = await game.run_single_game()
        stats.record(result)

        log_path = game.logger.save("game_logs")
        print(f"\n>>> 第 {i} 局日志已保存：{log_path}")
        winner_txt = "🐺 狼人胜" if result["winner"] == "werewolves" else "🎉 好人胜"
        print(f">>> 第 {i} 局结果：{winner_txt}（{result['rounds']} 回合）")

    stats.print_summary()
    stats.save_summary("game_logs/summary.txt")
    print(f"\n📁 汇总已保存：game_logs/summary.txt")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        gc.collect()
        gc.collect()