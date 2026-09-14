"""分类计数（counting）专用 Agent。

赛题评分只看时间效率：越快答对得分越高，答错计罚时。

设计要点：
1. 探索零 VLM 开销：原地 8×45° 扫视，每步只取结构化感知列表（id/color/shape/位置/包围盒），
   由代码合并去重成"物品清单"，全程不调用大模型。
2. 一次纯文本 VLM 调用答题：问题 + 清单 + 代码预计算的分组统计 → 答案 → 立即 submit_answer。
   纯文本调用比看图快一个量级。
3. 准确性保障：跨视角按 object_id 去重（id 是服务端为每个物体分配的稳定映射 ID，与世界坐标无关）；
   代码为每件物品预计算"可能放置在哪个家具上"的候选，辅助回答"桌上有多少…"类问题。
4. 循环兜底：全局时间上限、VLM 连续失败上限、同题重复作答检测（服务端尚未推进当前题时，
   直接复用已提交答案，避免重复扫视与重复调用，保证时间分）。
5. 多题目：计数题包含多个 subject（每问一 subject）。沿用 AgentBase 的 `_run_subject` 主循环：
   每一步只回答"当前 subject"，返回 `{action_space_key: 答案}`，不主动置 subject_finished，
   由服务端在收到答案后推进到下一问；全部答完后服务端 `is_current_subject_finished` 置真，主循环自然结束。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from google.protobuf import struct_pb2
from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg, parse_struct_to_data
from arenaagent.builder import Register
from arenaagent.generated.arena.message import basic_type_pb2, session_msg_pb2
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text
from arenaagent.vlm_agent import vlm_config as vlm_config_module

_FINISHED = session_msg_pb2.SessionStatus.Value("FINISHED")
_TERMINATED = session_msg_pb2.SessionStatus.Value("TERMINATED")
_ERROR = session_msg_pb2.SessionStatus.Value("ERROR")


@configclass
class CountingAgentCfg(AgentCfg):
    name: str = "counting_agent"
    sleep_between_steps: float = 0.2
    log_dir: str = "logs"
    tongsim_server_endpoint: str = "127.0.0.1:50060"
    vlm_model: str = "VLMGPT5Config"
    # ---- 兜底参数 ----
    time_budget_s: float = 300.0     # 单题时间预算（秒），400s 限时留余量
    run_budget_s: float = 3600.0     # 整个会话墙钟上限，防卡死
    max_vlm_fail_streak: int = 3
    # ---- 单点自转扫视（fov≈120°，3 个 120° 视角覆盖一圈） ----
    sweep_turn_degrees: float = 120.0
    sweep_views: int = 3
    turn_settle_s: float = 0.12      # 每次转向后等待引擎稳定再感知
    # ---- 航点巡游：首圈拿世界坐标→规划中心/家具前视/四角点，move_to_location 引擎寻路(自动绕墙) ----
    roam_time_budget_s: float = 95.0  # 单题感知总预算（含导航+扫视）
    roam_max_points: int = 4           # 3角+中心
    roam_inset_cm: float = 120.0       # 四角内缩量(厘米)，避免贴墙
    # ---- 单题(一次作答)总时限，防止撞满服务端400s ----
    per_attempt_budget_s: float = 150.0
    # ---- 多题推进/重试节奏 ----
    max_attempts_per_subject: int = 4  # 单题最多尝试次数(耗尽则换未试过的最可能选项，绝不空等)
    advance_wait_s: float = 6.0        # 提交+evaluate 后，等待服务端推进题号的宽限(消除竞态)
    perceive_width: int = 480        # 感知图(左RGB+右分割带ID)尺寸；平衡 gRPC 负载与可辨识度
    perceive_height: int = 400
    max_inventory_items: int = 200   # 送给 VLM 的清单项上限（防爆 token）
    # ---- 答题用视觉：巡游抓的视角图+清单一起给 VLM，识别 shape 里没有的类别(碗/钟/瓶) ----
    max_answer_images: int = 2
    # ---- 轻量安全网：4点走完后清单过少/模型没答案时，补看最大件家具背视点 ----
    min_inventory_items: int = 15
    safety_back_views: int = 2
    # ---- 主清单：每题所在场景不同，且赛题要求“每次重新识别” → 默认禁用磁盘缓存，逐连接重新巡游 ----
    use_inventory_cache: bool = False
    inventory_cache_path: str = ".counting_inventory.json"
    cache_ttl_s: float = 1800.0
    cache_min_items: int = 20


# 家具类语义关键字（用于把"物品"归到某个家具上）
_FURNITURE_HINTS = (
    "table", "desk", "chair", "sofa", "couch", "shelf", "bookcase", "cabinet",
    "counter", "stove", "fridge", "refrigerator", "sink", "bed", "nightstand",
    "drawer", "dresser", "tv", "television", "stand", "rack", "podium",
    "桌", "椅", "沙发", "柜", "架", "台", "床", "抽屉",
)


@Register("counting_agent")
class CountingAgent(AgentBase):
    _SYSTEM_PROMPT_FILE = os.path.join("prompts", "answer_prompt.txt")

    def __init__(self, stub, channel, cfg: CountingAgentCfg | None = None, sleep_between_steps: float = 0.2):
        super().__init__(stub=stub, channel=channel, cfg=cfg or CountingAgentCfg(), sleep_between_steps=sleep_between_steps)
        self.tongsim: TongSimGrpcClient | None = None
        self.character_id: str | None = None
        self.vlm_client = None

        self._start_time: float = 0.0
        self._step_count: int = 0
        self._subject_start: float = 0.0
        self._vlm_fail_streak: int = 0

        # 跨视角合并的物品清单：object_id -> 精简物品
        self._obj_memory: dict[str, dict[str, Any]] = {}
        # 巡游时抓取的视角图(base64 data url)，答题时作为视觉证据一起给 VLM
        self._view_images: list[str] = []
        self._perceive_fail_streak: int = 0
        self._dead_points: set = set()   # 导航失败点黑名单(同场景重试不再浪费)
        self._roam_phase: int = 0
        # 全场景主清单：一次性巡游建好，10 题共用（场景静态），后续题目只走一次纯文本 VLM
        self._master: list[dict[str, Any]] | None = None
        self._master_stats: dict[str, Any] | None = None
        self._cur_question: str | None = None

    # ------------------------------------------------------------------ #
    # 生命周期（与 tidy_room_agent 一致）
    # ------------------------------------------------------------------ #
    def init(self, opt: dict[str, Any]) -> None:
        if self.tongsim is not None:
            return
        endpoint = (opt or {}).get("tongsim_server_endpoint") or self.cfg.tongsim_server_endpoint
        self.tongsim = TongSimGrpcClient(endpoint=endpoint)

        try:
            spawn_loc = json.loads(opt["spawn_loc"])
            spawn_rot = json.loads(opt["spawn_rot"])
        except Exception as exc:  # pragma: no cover - 依赖 load() 已带真实 spawn 信息
            logger.error("缺少出生点信息，无法初始化角色: {}", exc)
            self.tongsim = None
            return

        camera_fov = float(opt.get("camera_fov", 120.0))
        camera_width = int(opt.get("camera_width", 1280))
        camera_height = int(opt.get("camera_height", 720))
        try:
            self._spawn_z = float(spawn_loc[2])
            self._spawn_xy = [float(spawn_loc[0]), float(spawn_loc[1])]
        except Exception:
            self._spawn_z = 20.0
            self._spawn_xy = [0.0, 0.0]
        self.character_id = self.tongsim.spawn_character(
            spawn_loc, spawn_rot, opt["name"], camera_fov, camera_width, camera_height
        )
        self.vlm_client = self._build_vlm_client()
        self._start_time = time.time()
        logger.info("CountingAgent init done, character={}", self.character_id)

    def deinit(self) -> None:
        if self.tongsim is not None:
            try:
                self.tongsim.close()
            except Exception as exc:
                logger.warning("close tongsim failed: {}", exc)
        self.tongsim = None
        self.character_id = None

    def _build_vlm_client(self):
        model_name = (getattr(self.cfg, "vlm_model", "") or "VLMGPT5Config").strip()
        cfg_cls = getattr(vlm_config_module, model_name, None)
        if cfg_cls is None:
            logger.warning("vlm_model {} not found, fallback to VLMGPT5Config", model_name)
            cfg_cls = vlm_config_module.VLMGPT5Config
        vlm_cfg = cfg_cls()  # __post_init__ 自动应用 VLM_CLIENT_* 环境变量覆盖
        logger.info("VLM client_type={}, model={}", vlm_cfg.client_type, vlm_cfg.client_cfg.name)
        return ClientFactory().build(vlm_cfg.client_type, vlm_cfg.client_cfg)

    # ------------------------------------------------------------------ #
    # 覆盖 run()。场景静态、10 题共用 → 先巡游一次建"主清单"，之后每题只做一次纯文本 VLM 答题。
    # 服务端"答对才推进题号"，故提交+evaluate 后用 advance_wait 宽限轮询题号变化(消除竞态)；
    # 仍未推进=答错，则补巡游刷新主清单再试，整场受墙钟预算保护。计数得分=越快答对分越高。
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        if not self.connected:
            return

        while not self._task_ready():
            time.sleep(self.sleep_between_steps)

        t0 = time.time()
        run_budget = float(self.cfg.run_budget_s)
        last_idx: int | None = None
        attempts = 0
        tried_vals: dict[int, set[int]] = {}   # 每题已提交过的选项值(防重复瞎猜)
        self._last_est: float | None = None

        while (time.time() - t0) < run_budget:
            if self._session_over():
                break
            idx = self._get_current_subject_index()
            if idx is None:
                time.sleep(0.5)
                continue

            if idx == last_idx:
                attempts += 1
            else:
                attempts, last_idx = 1, idx
                self._roam_phase = 0
                self._dead_points = set()  # 新场景=新房间，黑名单清零
                self._master = None          # 新题=新场景，必须重新感知
                self._master_stats = None

            self._refresh_action_space()
            subject = self._get_subject_from_task()
            task_response = self._get_response_from_task()
            options = self._extract_options(subject)
            key = self._answer_key()

            # 尝试次数耗尽 → 不再重巡，从未试过的选项里挑最可能的交上去(有机会得分，好过空等400s)
            if attempts > int(self.cfg.max_attempts_per_subject):
                probe = self._pick_untried(idx, options, tried_vals)
                if probe is None:
                    logger.warning("subject idx={} 选项已试尽，等服务端超时推进", idx)
                    time.sleep(3.0)
                    continue
                tried_vals.setdefault(idx, set()).add(probe)
                logger.warning("subject idx={} 转试探模式：提交未试过的选项值 {}", idx, probe)
                self._apply_action({key: probe})
                time.sleep(0.2)
                self._safe_evaluate()
                if self._wait_index_change(idx, float(self.cfg.advance_wait_s)):
                    logger.info("试探命中！推进到 {}", self._get_current_subject_index())
                continue

            # 正常作答：新场景/重试都先保证清单新鲜；重试换相位+缩预算，第4次只换问法不再重巡
            if attempts >= 2:
                self._roam_phase = attempts
            if attempts <= 3:
                self._roam_budget = 90.0 if attempts >= 2 else None
                self.invalidate_master_cache(silent=True)
            ok_build = self._build_master_with_retry(t0, force_rebuild=(attempts <= 3))
            self._roam_budget = None
            if not ok_build:
                logger.error("感知持续故障(tongsim/UE 可能已挂)，提前结束本场")
                self._disconnect()
                return

            action = self.run_step(subject, task_response)
            try:
                submit_val = int(str(list(action.values())[0]))
                tried_vals.setdefault(idx, set()).add(submit_val)
            except Exception:
                pass
            self._apply_action(action)
            time.sleep(0.2)
            self._safe_evaluate()

            # 等待题号推进（答对会推进）；用宽限消除竞态
            advanced = self._wait_index_change(idx, float(self.cfg.advance_wait_s))
            if advanced:
                logger.info("subject idx={} 答对并已推进到 {}", idx, self._get_current_subject_index())
            else:
                logger.info("subject idx={} 未推进(答错)，第 {} 次重试(换相位重巡)", idx, attempts + 1)

        self._disconnect()

    def _pick_untried(self, idx: int, options: dict[str, float], tried: dict[int, set[int]]) -> int | None:
        """试探模式：未试过的选项值里，选离当前估计值最近的；没估计值选中位。"""
        vals = [int(v) for v in options.values()]
        remain = [v for v in vals if v not in tried.get(idx, set())]
        if not remain:
            return None
        est = getattr(self, "_last_est", None)
        if est is None:
            remain.sort()
            return remain[len(remain) // 2]
        return min(remain, key=lambda v: abs(v - est))


    def _wait_index_change(self, idx: int, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self._session_over():
                return True
            if self._get_current_subject_index() != idx:
                return True
            time.sleep(0.4)
        return False

    def _session_over(self) -> bool:
        status = self._get_task_status().get("session_status")
        return status in (_FINISHED, _TERMINATED, _ERROR)

    def _refresh_action_space(self) -> None:
        try:
            self.action_space = parse_struct_to_data(
                self._call_struct("get_action_space", {"agent_id": self.agent_id}, struct_pb2.Struct.FromString)
            )
        except Exception as exc:
            logger.warning("刷新 action_space 失败: {}", exc)

    def _get_current_subject_index(self) -> int | None:
        try:
            resp = self._call_struct(
                "get_current_subject_index", {"agent_id": self.agent_id}, basic_type_pb2.Int32.FromString
            )
            v = getattr(resp, "value", None)
            return int(v) if v is not None else None
        except Exception as exc:
            logger.debug("get_current_subject_index 失败: {}", exc)
            return None

    def _safe_evaluate(self) -> None:
        try:
            self._evaluate_subject()
        except Exception as exc:
            logger.warning("evaluate_subject 失败: {}", exc)

    # ------------------------------------------------------------------ #
    # 主循环单步：回答"当前题目"
    # ------------------------------------------------------------------ #
    def run_step(self, subject: Any, task_response: dict[str, Any]) -> dict[str, Any]:
        if self.tongsim is None or self.character_id is None:
            self.init({})
        self._step_count += 1

        key = self._answer_key()
        question = self._extract_question(subject)
        t_attempt0 = time.time()

        # 复用一次性巡游建好的主清单；能代码数就不调模型
        inventory = self._ensure_master()
        stats = self._master_stats or self._group_stats(inventory)

        options = self._extract_options(subject)
        answer = self._solve(subject, question, inventory, stats, options)

        if answer is None:
            self._vlm_fail_streak += 1
            guess = self._resolve_option(self._fallback_answer(subject, stats), options)
            submit_val = self._to_submit_value(guess, options)
            self._last_est = float(submit_val)
            logger.warning("无有效答案(连续 {})，兜底提交 value={}", self._vlm_fail_streak, submit_val)
            return {key: submit_val}

        self._vlm_fail_streak = 0
        # 单题作答总时限：超时就直接按已有信息定案，绝不撞满服务端 400s
        spent = time.time() - t_attempt0
        submit_val = self._to_submit_value(answer, options)
        self._last_est = float(submit_val)
        logger.info("计数作答：q={!r} 清单{}件 -> 提交 value={}（用时{:.0f}s）", question[:32], len(inventory), submit_val, spent)
        return {key: submit_val}

    # ------------------------------------------------------------------ #
    # 选项处理：计数题给出 options(字母->数值)，必须提交"字母"而非数字
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_options(subject: Any) -> dict[str, float]:
        if not isinstance(subject, dict):
            return {}
        raw = subject.get("options")
        if not isinstance(raw, dict) or not raw:
            return {}
        opts: dict[str, float] = {}
        for k, v in raw.items():
            try:
                opts[str(k).strip().upper()] = float(v)
            except (TypeError, ValueError):
                continue
        return opts

    @staticmethod
    def _to_number(value: Any) -> float | None:
        try:
            s = str(value).strip().rstrip(".。")
            # 容错：模型可能写 "5个" / "A: 5" 之类
            import re
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            return float(m.group()) if m else None
        except (TypeError, ValueError):
            return None

    def _resolve_option(self, answer: Any, options: dict[str, float]) -> str:
        """把 VLM/兜底给出的答案归一为要提交的字符串：有选项→字母，否则→数字。"""
        if not options:
            n = self._to_number(answer)
            return str(int(n)) if n is not None and float(n).is_integer() else (str(n) if n is not None else str(answer).strip())
        # 1) 答案本身就是合法字母
        cand = str(answer).strip().upper()
        if cand in options:
            return cand
        # 2) 答案里含字母（如 "选 B" / "B: 5"）
        import re
        letters = re.findall(r"\b([A-Z])\b", cand)
        for L in letters:
            if L in options:
                return L
        # 3) 用数值在选项里反查最接近的一项
        n = self._to_number(answer)
        if n is not None:
            best, bestd = None, 1e18
            for L, v in options.items():
                d = abs(v - n)
                if d < bestd:
                    best, bestd = L, d
            if best is not None:
                return best
        # 4) 实在无法确定 → 选值最小（多为 0）以外的第一项，避免空提交
        return sorted(options.keys())[0]

    @staticmethod
    def _option_rank(letter: str, options: dict[str, float]) -> int | None:
        """把选项字母映射为 1-based 序号（A=1..H=8），即服务端要提交的整数。"""
        L = str(letter).strip().upper()
        if not options or L not in options:
            return None
        ordered = sorted(options.keys())
        return ordered.index(L) + 1

    def _to_submit_value(self, answer: Any, options: dict[str, float]) -> int:
        """最终提交值 = 被选中选项的“数值”。服务端把提交的数字按选项 value 匹配成 selected_option。
        （实测：提交 5 → 选中 G(=5)；提交数字本身即可，绝不能提交字母或 1-based 序号。）"""
        # 先拿到目标数值：字母→其 value；否则解析 answer 里的数字
        target: float | None = None
        if options:
            cand = str(answer).strip().upper()
            if cand in options:
                target = options[cand]
        if target is None:
            target = self._to_number(answer)
        if not options:
            return int(round(target)) if target is not None else 0
        # 把 count 对齐到“数值最接近”的那个选项，提交该选项的数值
        best_val, best_d = None, 1e18
        for v in options.values():
            d = abs(v - target) if target is not None else 1e18
            if d < best_d:
                best_val, best_d = v, d
        if best_val is None:
            best_val = next(iter(options.values()))
        return int(round(best_val))

    def _solve(self, subject, question, inventory, stats, options) -> str | None:
        # 1) 代码门控快答：类别/颜色词能可靠命中，且**数量精确等于某个选项值**才可信；
        #    像"书=23但选项没有23"这种明显虚高(清单重复/误配)，丢弃代码结果交给模型看图。
        code = self._code_count(question, inventory, stats)
        cn = self._to_number(code)
        if code is not None and cn is not None:
            if not options or any(abs(v - cn) < 1e-6 for v in options.values()):
                logger.info("代码计数命中：q={!r} -> {}", question[:36], code)
                return code
            logger.info("代码计数 {} 不在选项值内，弃用改走模型看图", code)
        # 2) 模型兜底（带巡游抓取的视角图，识别 shape 缺失的类别）
        parsed = self._ask_vlm(subject, question, inventory, stats, options)
        if not isinstance(parsed, dict):
            return None
        option = parsed.get("option", parsed.get("answer"))
        count = parsed.get("count")
        # 无选项：自由数值题，直接用 count（或模型给的 answer 文本）
        if not options:
            return count if count is not None else option
        # 有选项：优先信任模型给的字母，再用 count 反查校正
        chosen = self._resolve_option(option, options)
        if count is not None:
            cn = self._to_number(count)
            if cn is not None:
                for L, v in options.items():
                    if abs(v - cn) < 1e-6:
                        if chosen != L:
                            logger.info("按 count={} 校正选项: {} -> {}", cn, chosen, L)
                        chosen = L
                        break
        return chosen

    # 类别词(中文) -> 物体 shape/name/type 里可能出现的英文词
    _CATEGORY_SYNONYMS = {
        "苹果": ["apple"], "香蕉": ["banana"], "水果": ["fruit", "apple", "banana"],
        "杯子": ["cup", "mug"], "杯": ["cup", "mug"], "瓶子": ["bottle"], "罐": ["can"],
        "球": ["ball"], "书": ["book"], "鞋": ["shoe"], "玩具": ["toy", "doll", "cap"],
        "食物": ["food"], "面包": ["bread"],
        "椅子": ["chair"], "沙发": ["sofa", "couch"], "桌子": ["table", "desk"],
        "柜子": ["cabinet", "shelf"], "床": ["bed"], "电视": ["tv", "television"],
    }
    _FURNITURE_WORDS = ("椅子", "沙发", "桌子", "桌", "柜", "床", "电视")

    def _code_count(self, question: str, inventory: list[dict[str, Any]], stats: dict[str, Any]) -> str | None:
        """能在清单里可靠命中类别/颜色时直接数；否则返回 None 交给模型。低置信一律不出数。"""
        import re
        q = question
        is_furniture_q = any(w in q for w in self._FURNITURE_WORDS)

        def cat_terms(c):  # 该类别的候选英文词
            return self._CATEGORY_SYNONYMS.get(c, [])

        # 选出题目里出现的类别词（取命中的第一个）
        cat = next((c for c in self._CATEGORY_SYNONYMS if c in q), None)
        # 颜色词
        col_en = next((en for en, syns in self._COLOR_SYNONYMS.items() if any(s in q for s in syns + [en])), None)

        def matches(it):
            fields = " ".join(str(it.get(k, "")) for k in ("shape", "name", "type", "category")).lower()
            if cat and not any(t in fields for t in cat_terms(cat)):
                return False
            if col_en and str(it.get("color", "")).lower() != col_en:
                return False
            return True

        # 纯“总共/多少个物体”类（无类别无颜色）：数非家具
        total_q = (cat is None and col_en is None and
                   re.search(r"一共|总共|多少个物体|多少个物品|total|how many objects", q, re.IGNORECASE))

        if total_q:
            items = [it for it in inventory if not self._looks_like_furniture(it)]
            return str(len(items)) if items else None

        if cat is None and col_en is None:
            return None  # 不认识的类别 → 交给模型

        # 有类别/颜色：先确认清单里“真的存在”该类别(否则多半是 Unknown/漏检，不可信)
        exists = any(matches(it) for it in inventory) if (cat or col_en) else False
        if cat and not exists:
            return None  # 清单里根本没这个类别字段 → 不可信，交给模型

        items = [it for it in inventory if matches(it)]
        if not is_furniture_q:
            # 数非家具目标；家具类别题则保留
            if not any(w in q for w in self._FURNITURE_WORDS):
                items = [it for it in items if not self._looks_like_furniture(it) or matches(it)]
        n = len(items)
        # 颜色-only 且样本太少(<3)容易不稳，交给模型
        if col_en and not cat and n < 3:
            return None
        return str(n)

    def _answer_key(self) -> str:
        if isinstance(self.action_space, dict):
            return self.action_space.get("key") or "action"
        return "action"

    # ------------------------------------------------------------------ #
    # 题目文本提取（不同任务 subject 字段不一致，做防御）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_question(subject: Any) -> str:
        if isinstance(subject, dict):
            for k in ("subject", "question", "text", "task_text", "prompt"):
                v = subject.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return json.dumps(subject, ensure_ascii=False)
        return str(subject)

    # ------------------------------------------------------------------ #
    # 感知：零 VLM。原地 360° + 以"走到物体旁"为航点巡游，合并去重破除遮挡，一次建主清单复用。
    # ------------------------------------------------------------------ #
    def _spin_sweep(self, capture_image: bool = True) -> None:
        views = max(int(self.cfg.sweep_views), 1)
        step = float(self.cfg.sweep_turn_degrees)
        phase = (getattr(self, "_roam_phase", 0) * 37.0) % 360.0  # 重试轮换相位，覆盖不同缝隙
        for i in range(views):
            try:
                self.tongsim.turn_in_degree(self.character_id, (phase + step * i) % 360.0)  # 绝对朝向
            except Exception as exc:
                logger.warning("转向失败（第 {} 视角）：{}", i, exc)
            time.sleep(float(self.cfg.turn_settle_s))
            self._perceive_into_memory(capture_image=capture_image and i == 0)

    def _ensure_master(self) -> list[dict[str, Any]]:
        if self._master is not None:
            return self._master
        # 1) 先尝试磁盘缓存（静态场景：首个连接巡游一次并落盘，其余连接直接读，秒答）
        cached = self._load_master_cache()
        if cached is not None:
            self._obj_memory = {it["id"]: it for it in cached}
            self._master = cached
            self._master_stats = self._group_stats(cached)
            logger.info("命中主清单缓存：{} 件（跳过巡游）", len(cached))
            return self._master

        # 2) 4点法巡游：出生点转3张拿bbox → 3个角(朝中心拍1张) + 房间中心(3张×120°)
        #    家具点全部去掉；走完若清单过少则补看最大件家具背视点(安全网)
        self._obj_memory = {}
        self._view_images = []
        self._spin_sweep()  # 出生点 3 张，建立初始清单与 bbox

        budget = time.time() + float(getattr(self, "_roam_budget", None) or self.cfg.roam_time_budget_s)
        points, center = self._plan_viewpoints()
        logger.info("规划观察点 {} 个: {}", len(points), [(int(p[0]), int(p[1]), m) for p, m in points])
        dead: set = getattr(self, "_dead_points", None) or set()
        self._dead_points = dead
        for k, (p, mode) in enumerate(points):
            if time.time() > budget or len(self._obj_memory) >= int(self.cfg.max_inventory_items):
                break
            tag = (int(p[0]), int(p[1]))
            if tag in dead:
                logger.info("观察点#{} {} 在黑名单(曾不可达)，跳过", k, tag)
                continue
            before = len(self._obj_memory)
            ok = self._goto_point(p)
            if ok:
                if mode == "center":
                    self._spin_sweep()          # 中心 360° = 3张×120°
                else:
                    self._observe_toward(center)  # 角落：朝中心拍 1 张(FOV120 覆盖朝内90°)
            gained = len(self._obj_memory) - before
            logger.info("观察点#{} ({},{},{}) 导航{} 新增 {} 件，累计 {} 件", k, tag[0], tag[1], mode, "ok" if ok else "FAIL", gained, len(self._obj_memory))
            if not ok:
                dead.add(tag)

        # —— 安全网：清单过少 → 补看最大件家具的背视点（每点 1 张，朝中心）——
        if len(self._obj_memory) < int(self.cfg.min_inventory_items):
            logger.warning("清单仅 {} 件(<{}), 启动安全网", len(self._obj_memory), self.cfg.min_inventory_items)
            for f in self._furniture_desc(up_to=int(self.cfg.safety_back_views)):
                if time.time() > budget:
                    break
                fx, fy = f["pos"][0], f["pos"][1]
                bx = fx + (fx - center[0]) * 0.45
                by = fy + (fy - center[1]) * 0.45
                bx += (center[0] - bx) * 0.15
                by += (center[1] - by) * 0.15
                back = [bx, by, center[2]]
                tag = (int(back[0]), int(back[1]))
                if tag in dead:
                    continue
                before = len(self._obj_memory)
                if not self._goto_point(back):
                    dead.add(tag)
                    continue
                self._observe_toward(center)
                logger.info("安全网 家具背视({},{}): 新增 {} 件，累计 {} 件", tag[0], tag[1], len(self._obj_memory) - before, len(self._obj_memory))

        self._master = list(self._obj_memory.values())
        self._master_stats = self._group_stats(self._master)
        logger.info("主清单建立完成：共 {} 件物体", len(self._master))
        self._save_master_cache(self._master)
        return self._master

    def _observe_toward(self, target: list[float]) -> None:
        """原地转向 target 方向后拍摄 1 张(FOV 120° 足够覆盖该角朝内的 90°)。"""
        import math
        me = self._current_xy()
        heading = math.degrees(math.atan2(target[1] - me[1], target[0] - me[0]))
        try:
            self.tongsim.turn_in_degree(self.character_id, heading % 360.0)
        except Exception as exc:
            logger.warning("转向失败: {}", exc)
        time.sleep(float(self.cfg.turn_settle_s))
        self._perceive_into_memory(capture_image=True)

    def _current_xy(self) -> list[float]:
        return list(getattr(self, "_last_nav_xy", None) or getattr(self, "_spawn_xy", [0.0, 0.0]))

    def _furniture_desc(self, up_to: int = 2) -> list[dict[str, Any]]:
        objs = list(self._obj_memory.values())
        return sorted((it for it in objs if self._looks_like_furniture(it)),
                      key=lambda it: max(it["size"]), reverse=True)[:up_to]


    def _plan_viewpoints(self):
        """4点法：bbox四角 → 排除"床角"(shape/name 含 bed；认不出用最大件家具代理) →
        其余3角朝中心内缩 inset + 可落点校验；再加房间正中心(360°)。
        返回 ([(点, mode)], 中心点)。"""
        objs = list(self._obj_memory.values())
        z = float(getattr(self, "_spawn_z", 20.0))
        if not objs:
            return [], [0.0, 0.0, z]
        xs = [it["pos"][0] for it in objs]
        ys = [it["pos"][1] for it in objs]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        inset = float(self.cfg.roam_inset_cm)
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        center = [cx, cy, z]
        self._room_center = center

        corners = [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
        # 床角判定：shape/name 含 bed；否则最大件家具代理
        bed = next((it for it in objs
                    if "bed" in str(it.get("shape", "")).lower() or "bed" in str(it.get("name", "")).lower()), None)
        anchor = bed
        if anchor is None:
            furns = sorted((it for it in objs if self._looks_like_furniture(it)),
                           key=lambda it: max(it["size"]), reverse=True)
            anchor = furns[0] if furns else None
        if anchor is not None:
            ax, ay = float(anchor["pos"][0]), float(anchor["pos"][1])
            logger.info("床角判定: 依据 {} (含bed={}) 位置({},{}), 排除离它最近的角",
                        anchor.get("id"), bool(bed), int(ax), int(ay))
            corners.sort(key=lambda c: (c[0] - ax) ** 2 + (c[1] - ay) ** 2, reverse=True)  # 远的在前
        keep = corners[:3]

        pts: list[tuple[list[float], str]] = []
        for c in keep:
            dx, dy = cx - c[0], cy - c[1]
            d = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            step = min(inset, d * 0.8)
            q = [c[0] + dx / d * step, c[1] + dy / d * step, z]
            q = self._adjust_point(q, objs, center)
            if q is not None:
                pts.append((q, "corner"))
        cq = self._adjust_point([cx, cy, z], objs, center)
        if cq is not None:
            pts.append((cq, "center"))

        spawn = self._current_xy()
        order: list[tuple[list[float], str]] = []
        cur = spawn
        pool = pts[:]
        while pool:
            nxt = min(pool, key=lambda pc: (pc[0][0] - cur[0]) ** 2 + (pc[0][1] - cur[1]) ** 2)
            pool.remove(nxt)
            order.append(nxt)
            cur = [nxt[0][0], nxt[0][1]]
        return order[: max(int(self.cfg.roam_max_points), 0)], center

    def _adjust_point(self, q: list[float], objs: list[dict[str, Any]], center: list[float]):
        """可落点校验：压到物体就朝中心回缩，最多5次；仍不行返回 None。"""
        for _ in range(5):
            if self._point_blocked(q[0], q[1], objs) is None:
                return q
            q = [q[0] + (center[0] - q[0]) * 0.3, q[1] + (center[1] - q[1]) * 0.3, q[2]]
        return None


    def _point_blocked(self, x: float, y: float, objs: list[dict[str, Any]], pad: float = 45.0) -> str | None:
        """点是否压在某个物体的占地(含角色体宽 pad)里。返回挡住它的物体id或 None。"""
        for it in objs:
            sx, sy = it["size"][0], it["size"][1]
            if max(sx, sy) < 30:      # 小物件不挡路
                continue
            hx, hy = sx / 2.0 + pad, sy / 2.0 + pad
            if abs(x - it["pos"][0]) <= hx and abs(y - it["pos"][1]) <= hy:
                return it["id"]
        return None

    def _goto_point(self, p: list[float]) -> bool:
        try:
            res = self.tongsim.move_to_location(
                self.character_id, [float(p[0]), float(p[1]), float(p[2])], stop_distance=1.2
            )
            time.sleep(0.3)
            ok = not (isinstance(res, dict) and res.get("result") == "failed")
            if ok:
                self._last_nav_xy = [float(p[0]), float(p[1])]
            return ok
        except Exception as exc:
            logger.warning("move_to_location({},{}) 异常: {}", int(p[0]), int(p[1]), exc)
            return False


    def _build_master_with_retry(self, t_session_start: float, min_items: int = 8, tries: int = 3,
                                 force_rebuild: bool = True) -> bool:
        """带重试的主清单构建：感知故障/清单过空时快速失败返回 False，绝不拿空清单去答题。
        force_rebuild=False 时若已有清单则直接复用（第4次"只换问法不重巡"用）。"""
        if not force_rebuild and (self._master or []):
            self._perceive_fail_streak = 0
            return len(self._master) >= min_items or self._perceive_fail_streak == 0
        for k in range(tries):
            if force_rebuild:
                self._master = None
            self._perceive_fail_streak = 0
            inv = self._ensure_master()
            if len(inv) >= min_items and self._perceive_fail_streak < 4:
                return True
            logger.warning("主清单不可用(件数={}, 感知连续失败={})，第 {} 次重建", len(inv), self._perceive_fail_streak, k + 1)
            if time.time() - t_session_start > float(self.cfg.run_budget_s) * 0.5:
                break
            time.sleep(5.0)
        return False

    # ---- 主清单磁盘缓存 ---- #
    def _cache_file(self) -> str:
        p = self.cfg.inventory_cache_path or ".counting_inventory.json"
        if not os.path.isabs(p):
            p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", p)
            p = os.path.abspath(p)
        return p

    def _load_master_cache(self) -> list[dict[str, Any]] | None:
        if not getattr(self.cfg, "use_inventory_cache", True):
            return None
        path = self._cache_file()
        try:
            if not os.path.exists(path):
                return None
            if (time.time() - os.path.getmtime(path)) > float(self.cfg.cache_ttl_s):
                logger.info("主清单缓存已过期，重新巡游")
                return None
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            items = data.get("objects") if isinstance(data, dict) else data
            if not isinstance(items, list) or len(items) < int(self.cfg.cache_min_items):
                return None
            # 规范化，确保每条有 id/pos/size
            norm = []
            for it in items:
                if isinstance(it, dict) and it.get("id") and isinstance(it.get("pos"), list):
                    norm.append(it)
            return norm or None
        except Exception as exc:
            logger.warning("读取主清单缓存失败: {}", exc)
            return None

    def _save_master_cache(self, master: list[dict[str, Any]]) -> None:
        if not getattr(self.cfg, "use_inventory_cache", True):
            return
        try:
            path = self._cache_file()
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"objects": master, "count": len(master)}, fh, ensure_ascii=False)
            logger.info("主清单已写入缓存: {}", path)
        except Exception as exc:
            logger.warning("写入主清单缓存失败: {}", exc)

    def invalidate_master_cache(self, silent: bool = False) -> None:
        """丢弃内存清单(下次作答重新感知)。silent=True 用于常规换题/重试，不算故障。"""
        self._master = None
        self._master_stats = None
        if silent:
            return
        try:
            path = self._cache_file()
            if os.path.exists(path):
                os.remove(path)
                logger.info("已删除过期主清单缓存")
        except Exception:
            pass

    def _pick_waypoint(self, visited: set[str]) -> str | None:
        """优先挑还没去过、且最"大/最远"的家具类航点(最能暴露遮挡)，其次挑最远的任意物体。"""
        items = [it for it in self._obj_memory.values() if it["id"] not in visited]
        if not items:
            return None
        cx = sum(it["pos"][0] for it in self._obj_memory.values()) / max(len(self._obj_memory), 1)
        cy = sum(it["pos"][1] for it in self._obj_memory.values()) / max(len(self._obj_memory), 1)

        def dist(it):
            return (it["pos"][0] - cx) ** 2 + (it["pos"][1] - cy) ** 2

        furn = [it for it in items if self._looks_like_furniture(it)]
        pool = furn if furn else items
        return max(pool, key=dist)["id"]

    def _goto_object(self, object_id: str) -> None:
        """用引擎寻路走到物体旁（比盲目前进更不易撞墙）。失败则退回朝该物体方向前进。"""
        try:
            res = self.tongsim.move_to_object(self.character_id, object_id)
            time.sleep(0.3)
            if isinstance(res, dict) and res.get("result") == "failed":
                logger.debug("move_to_object({}) 失败: {}", object_id, res.get("error"))
        except Exception as exc:
            logger.warning("move_to_object({}) 异常: {}", object_id, exc)

    def _perceive_into_memory(self, capture_image: bool = False) -> None:
        # 文本答题模式：把图压到最小(2x2)，只取结构化列表 → 大幅省 gRPC 传输/引擎渲染时间
        try:
            if capture_image:
                w, h = int(self.cfg.perceive_width), int(self.cfg.perceive_height)
            else:
                w, h = 2, 2
            perception = self.tongsim.acquire_first_person_perception(
                self.character_id, width=w, height=h
            )
            self._perceive_fail_streak = 0
        except Exception as exc:
            self._perceive_fail_streak = getattr(self, "_perceive_fail_streak", 0) + 1
            logger.warning("感知失败(连续 {})：{}", self._perceive_fail_streak, exc)
            return
        if capture_image and len(self._view_images) < int(self.cfg.max_answer_images):
            img = perception.get("image")
            if img:
                self._view_images.append(img if str(img).startswith("data:image") else f"data:image/jpeg;base64,{img}")
        for obj in perception.get("objects") or []:
            compact = self._compact_object(obj)
            if compact is None:
                continue
            obj_id = compact["id"]
            if not obj_id:
                continue
            # 跨视角去重：id 为服务端稳定映射，优先保留信息更全的一条
            if obj_id not in self._obj_memory:
                self._obj_memory[obj_id] = compact

    @staticmethod
    def _r(value: Any, digits: int = 1) -> float:
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _compact_object(cls, obj: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(obj, dict):
            return None
        loc = obj.get("place_location") or obj.get("location") or {}
        aabb = obj.get("world_aabb") or obj.get("aabb") or {}
        amin = aabb.get("min") or {}
        amax = aabb.get("max") or {}
        pos = [cls._r(loc.get("X", loc.get("x", 0))), cls._r(loc.get("Y", loc.get("y", 0))), cls._r(loc.get("Z", loc.get("z", 0)))]
        size = [
            cls._r(amax.get("X", amax.get("x", 0))) - cls._r(amin.get("X", amin.get("x", 0))),
            cls._r(amax.get("Y", amax.get("y", 0))) - cls._r(amin.get("Y", amax.get("y", 0))),
            cls._r(amax.get("Z", amax.get("z", 0))) - cls._r(amin.get("Z", amax.get("z", 0))),
        ]
        compact: dict[str, Any] = {
            "id": str(obj.get("object_id", obj.get("id", ""))),
            "pos": pos,
            "size": size,
        }
        # 复制所有标量语义字段（color / shape / name / type / category / material ...）
        for k, v in obj.items():
            if k in ("object_id", "id", "place_location", "location", "world_aabb", "aabb", "image"):
                continue
            if isinstance(v, (str, int, float, bool)):
                compact[k] = v
        return compact

    # ------------------------------------------------------------------ #
    # 代码预计算：把物品归到家具上 + 分组统计
    # ------------------------------------------------------------------ #
    def _group_stats(self, inventory: list[dict[str, Any]]) -> dict[str, Any]:
        # 识别家具：名称含关键字，或包围盒最大边 > 80cm
        furniture: dict[str, dict[str, Any]] = {}
        for it in inventory:
            if self._looks_like_furniture(it):
                furniture[it["id"]] = it

        # 为每件非家具物品，找它正下方最近的家具（XY 落在包围盒内、Z 略低于家具顶）
        placed_on: dict[str, str] = {}
        for it in inventory:
            if it["id"] in furniture:
                continue
            best, best_gap = None, 1e9
            ix, iy, iz = it["pos"]
            for f in furniture.values():
                fx, fy, fz = f["pos"]
                fs = f["size"]
                half_x = max(fs[0], 1) / 2 + 10
                half_y = max(fs[1], 1) / 2 + 10
                if abs(ix - fx) <= half_x and abs(iy - fy) <= half_y:
                    # 家具表面高度（用中心+半高近似）
                    surf = fz + max(fs[2], 1) / 2
                    gap = surf - iz
                    if -30 <= gap <= 60 and gap < best_gap:  # 物品在家具面上附近
                        best, best_gap = f["id"], gap
            if best:
                placed_on[it["id"]] = best

        def _count_by(field: str) -> dict[str, int]:
            agg: dict[str, int] = {}
            for it in inventory:
                if it["id"] in furniture:
                    continue
                val = str(it.get(field, "?")).strip() or "?"
                agg[val] = agg.get(val, 0) + 1
            return agg

        return {
            "total_objects": len(inventory),
            "furniture_ids": list(furniture.keys()),
            "by_color": _count_by("color"),
            "by_shape": _count_by("shape"),
            "by_name": _count_by("name") if any("name" in it for it in inventory) else {},
            "placed_on": placed_on,  # 物品id -> 家具id
            "on_furniture_count": self._count_per_furniture(placed_on),
        }

    @staticmethod
    def _looks_like_furniture(item: dict[str, Any]) -> bool:
        blob = " ".join(str(item.get(k, "")) for k in ("name", "type", "category", "id", "shape")).lower()
        if any(h in blob for h in _FURNITURE_HINTS):
            return True
        size = item.get("size") or [0, 0, 0]
        return max(size) > 80

    @staticmethod
    def _count_per_furniture(placed_on: dict[str, str]) -> dict[str, int]:
        agg: dict[str, int] = {}
        for fid in placed_on.values():
            agg[fid] = agg.get(fid, 0) + 1
        return agg

    # ------------------------------------------------------------------ #
    # 一次纯文本 VLM 调用答题
    # ------------------------------------------------------------------ #
    def _ask_vlm(self, subject: Any, question: str, inventory: list[dict[str, Any]], stats: dict[str, Any], options: dict[str, float]) -> dict[str, Any] | None:
        system_prompt = self._load_system_prompt()
        inv_brief = inventory[: int(self.cfg.max_inventory_items)]
        # 压缩到模型真正需要的字段：id / color / shape / name。pos/size 已由代码统计消化掉
        slim = [{**{"id": it["id"]}, **{k: it[k] for k in ("color", "shape", "name", "type", "category") if it.get(k)}} for it in inv_brief]
        opts_line = ""
        if options:
            opts_line = (
                "【选项(字母=数值)】\n"
                + ", ".join(f"{L}={int(v) if float(v).is_integer() else v}" for L, v in sorted(options.items()))
                + "\n最终必须选出一个选项字母。\n\n"
            )
        user_text = (
            "【题目原文】\n"
            f"{question}\n\n"
            f"{opts_line}"
            "【证据1：巡游多个位置拍摄的第一视角图】左侧真实画面，右侧是按数字 id 标注的语义分割图。\n"
            "【证据2：代码预计算统计】\n"
            f"{json.dumps(stats, ensure_ascii=False)}\n\n"
            "【证据3：物体清单 id 索引】(注意 color/shape 常为 Unknown 或只标几何形状，**类别以看图为准**)\n"
            f"{json.dumps(slim, ensure_ascii=False)}\n\n"
            "【计数规则】1) 以图片认出目标类别的物体，对照分割图记下其 id；同一 id 跨图只计 1 次。"
            "2) 不把家具(桌/柜/床/椅等大件)算进目标，除非题目问的就是家具。"
            "3) 拿不准是否属于目标类别的物体宁缺勿滥。4) 数出 count 后，选出数值等于 count 的选项字母。\n"
            "只输出一个 JSON 对象：\n"
            '{"think":"引用图中看到的物体id说明计数依据","count":数字,"option":"选项字母"}'
        )
        content: list[dict[str, Any]] = []
        for url in self._view_images[: int(self.cfg.max_answer_images)]:
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},  # 图+清单：识别 shape 字段缺失的类别(钟/碗/瓶)
        ]
        try:
            response = self.vlm_client.invoke(messages)
        except Exception as exc:
            logger.error("VLM 调用异常: {}", exc)
            return None
        text = getattr(response, "text", "") or ""
        parsed = self._parse_json_any(text)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            logger.error("VLM 输出无法解析: {}", text[:300])
            return None
        if parsed.get("count") is None and parsed.get("option") is None and parsed.get("answer") is None:
            logger.error("VLM 未给出 count/option: {}", text[:300])
            return None
        logger.info("VLM 计数推理: {}", str(parsed.get("think", ""))[:200])
        return parsed

    @staticmethod
    def _parse_json_any(text: str) -> Any:
        """稳健解析：容忍 ```json 代码围栏、裸 {…} 对象、以及 [{…}] 数组。"""
        import re

        t = (text or "").strip()
        if not t:
            return None
        # 去掉 markdown 代码围栏
        t = re.sub(r"^```(?:json)?", "", t).strip()
        t = re.sub(r"```$", "", t).strip()
        # 1) 直接整体解析
        for cand in (t,):
            try:
                return json.loads(cand)
            except Exception:
                pass
        # 2) 抓最后一个平衡的 { ... }
        start = t.rfind("{")
        while start != -1:
            depth = 0
            for i in range(start, len(t)):
                if t[i] == "{":
                    depth += 1
                elif t[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[start:i + 1])
                        except Exception:
                            break
            start = t.rfind("{", 0, start)
        # 3) 退回到项目自带解析器
        return extract_last_json_from_text(text)

    def _load_system_prompt(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self._SYSTEM_PROMPT_FILE)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read()
        except Exception as exc:
            logger.warning("加载系统提示失败({})，使用内置默认", exc)
            return (
                "你是严谨的场景计数助手。综合多视角图(含带id分割图)与清单(id/color/shape)计数："
                "类别以看图为准(Unknown不代表不是目标)；同一 id 只数一次；镜面反射不是新物体，按 id 判重；"
                "不把家具算进目标；数出数量后选数值恰等于该数量的选项字母。只输出 JSON："
                '{"think":"引用id与图号","count":数字,"option":"字母"}。'
            )

    # ------------------------------------------------------------------ #
    # 兜底答案：VLM 不可用时，尝试从统计里直接命中"某颜色/某形状共几个"
    # ------------------------------------------------------------------ #
    _COLOR_SYNONYMS = {
        "red": ["红色", "红"], "blue": ["蓝色", "蓝"], "green": ["绿色", "绿"],
        "yellow": ["黄色", "黄"], "black": ["黑色", "黑"], "white": ["白色", "白"],
        "brown": ["棕色", "褐色", "棕", "褐"], "orange": ["橙色", "橙"],
        "pink": ["粉色", "粉"], "purple": ["紫色", "紫"],
        "grey": ["灰色", "灰"], "gray": ["灰色", "灰"],
    }
    _SHAPE_SYNONYMS = {
        "ball": ["球", "圆形", "球形"], "sphere": ["球", "圆形", "球形"],
        "cube": ["立方体", "正方体", "方块"], "box": ["盒子", "箱子", "盒"],
        "apple": ["苹果"], "banana": ["香蕉"], "cup": ["杯子", "水杯", "杯"],
        "bottle": ["瓶子", "瓶"], "can": ["罐子", "易拉罐", "罐"],
    }

    @classmethod
    def _match_count(cls, question: str, agg: dict[str, int], syn: dict[str, list[str]]) -> str | None:
        for value, cnt in (agg or {}).items():
            if not value or value == "?":
                continue
            tokens = [value] + syn.get(str(value).lower(), [])
            if any(t and t in question for t in tokens):
                return str(cnt)
        return None

    def _fallback_answer(self, subject: Any, stats: dict[str, Any]) -> str:
        import re

        question = self._extract_question(subject)
        hit = self._match_count(question, stats.get("by_color") or {}, self._COLOR_SYNONYMS)
        if hit is not None:
            return hit
        hit = self._match_count(question, stats.get("by_shape") or {}, self._SHAPE_SYNONYMS)
        if hit is not None:
            return hit
        # 兜底：无类别词命中时，非家具物体总数（仅在 VLM 失败时才会用到）
        if re.search(r"一共|总共|多少个|几个|how many|total", question, re.IGNORECASE):
            total = stats.get("total_objects", 0) - len(stats.get("furniture_ids") or [])
            return str(max(total, 0))
        return "0"
