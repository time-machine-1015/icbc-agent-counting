"""整理房间（tidy-room）专用 Agent。

设计约束（赛题要求）：
1. Agent 只发目标点：所有位移/抓取/放置均由引擎原生动作完成（move_to_location /
   move_and_take_object / put_down_sth），Agent 代码只负责把感知数据换算成目标点，
   VLM 不输出任何坐标。
2. 感知信息结构化精简：感知 objects 压缩为 {id,color,shape,pos,size} 短列表 +
   结构化状态块，每轮独立决策，不堆叠多轮对话历史。
3. 循环上限兜底：全局步数上限、全局时间预算、take/put 分项重试上限、
   同参数重复决策检测、VLM 连续失败检测，超限自动降级（换策略/跳过/强制收尾）。
4. 状态同步及时：每步开始重新感知并查询手部状态（服务端权威状态，带短轮询），
   每个动作执行后立即校验返回值，失败原因写回下一轮决策输入。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from loguru import logger

from arenaagent.agent_base import AgentBase, AgentCfg
from arenaagent.builder import Register
from arenaagent.tongsim_grpc_client import TongSimGrpcClient
from arenaagent.utils.configclass import configclass
from arenaagent.vlm_agent.client import ClientFactory
from arenaagent.vlm_agent.json_parsor import extract_last_json_from_text
from arenaagent.vlm_agent import vlm_config as vlm_config_module


@configclass
class TidyRoomAgentCfg(AgentCfg):
    name: str = "tidy_room_agent"
    sleep_between_steps: float = 0.3
    log_dir: str = "logs"
    tongsim_server_endpoint: str = "127.0.0.1:50060"
    # vlm_model 为 vlm_config.py 中 __ALL__ 里的配置类名；
    # API 地址/Key/模型名可用通用环境变量 VLM_CLIENT_CFG_* 覆盖。
    vlm_model: str = "VLMGPT5Config"
    # ---- 兜底参数（要求3） ----
    time_budget_s: float = 330.0
    max_steps: int = 120
    max_take_attempts: int = 2
    max_put_attempts: int = 3
    max_vlm_fail_streak: int = 3
    # ---- 放置参数 ----
    place_z_lift: float = 6.0  # 目标家具包围盒顶部之上的抬升量（厘米）
    hand_state_timeout_s: float = 3.0  # 手部状态轮询超时


@Register("tidy_room_agent")
class TidyRoomAgent(AgentBase):
    _SYSTEM_PROMPT_FILE = os.path.join("prompts", "system_prompt.txt")
    _MAX_VISIBLE_OBJECTS = 40

    def __init__(self, stub, channel, cfg: TidyRoomAgentCfg | None = None, sleep_between_steps: float = 0.3):
        super().__init__(stub=stub, channel=channel, cfg=cfg or TidyRoomAgentCfg(), sleep_between_steps=sleep_between_steps)
        self.tongsim: TongSimGrpcClient | None = None
        self.character_id: str | None = None
        self.vlm_client = None

        self._start_time: float = 0.0
        self._step_count: int = 0
        self._finished: bool = False

        # ---- 状态同步记忆（要求4） ----
        self._hand_full: bool = False
        self._held_object_id: str = ""  # 本地记录手中物体（服务端只给 bool）
        self._last_action_result: dict[str, Any] = {}
        self._last_fingerprint: str = ""
        self._same_decision_streak: int = 0
        self._vlm_fail_streak: int = 0
        self._obj_memory: dict[str, dict[str, Any]] = {}  # object_id -> 精简感知
        self._handled_ids: list[str] = []  # 已完成 take+put 的映射 ID
        self._take_fail: dict[str, int] = {}
        self._put_fail: dict[str, int] = {}
        self._stuck_hint: str = ""

        # 任务结构信息（从 subject 防御性提取）
        self._movable_scene_ids: list[str] = []
        self._subject_targets: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def init(self, opt: dict[str, Any]) -> None:
        if self.tongsim is not None:
            return
        endpoint = opt.get("tongsim_server_endpoint") or self.cfg.tongsim_server_endpoint
        self.tongsim = TongSimGrpcClient(endpoint=endpoint)

        spawn_loc = json.loads(opt["spawn_loc"])
        spawn_rot = json.loads(opt["spawn_rot"])
        camera_fov = float(opt.get("camera_fov", 120.0))
        camera_width = int(opt.get("camera_width", 1280))
        camera_height = int(opt.get("camera_height", 720))
        self.character_id = self.tongsim.spawn_character(
            spawn_loc, spawn_rot, opt["name"], camera_fov, camera_width, camera_height
        )
        self.vlm_client = self._build_vlm_client()
        self._start_time = time.time()
        logger.info("TidyRoomAgent init done, character={}", self.character_id)

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
    # 主循环单步
    # ------------------------------------------------------------------ #
    def run_step(self, subject, task_response: dict[str, Any]) -> dict[str, Any]:
        if self.tongsim is None or self.character_id is None:
            self.init({})
        self._step_count += 1

        # ---- 兜底检查（要求3）：时间预算 / 步数上限 ---- #
        budget_hit, budget_reason = self._budget_exceeded()
        if budget_hit and not self._finished:
            logger.warning("兜底触发: {}，强制收尾", budget_reason)
            return self._finish(f"兜底触发（{budget_reason}），提交当前完成度")

        # ---- 状态同步（要求4）：服务端权威状态 ---- #
        perception = self.tongsim.acquire_first_person_perception(self.character_id, width=1280, height=720)
        compact_objects = self._compact_objects(perception.get("objects") or [])
        image_url = self._to_data_url(perception.get("image"))
        self._sync_hand_state()
        self._update_object_memory(compact_objects)
        self._extract_subject_info(subject)

        state_block = self._build_state_block(subject, compact_objects)

        # ---- VLM 决策（每轮一个高层动作，不输出坐标） ---- #
        decision = self._decide(state_block, image_url)
        if not decision:
            self._vlm_fail_streak += 1
            self._last_action_result = {"result": "failed", "error": "vlm decision parse failed"}
            if self._vlm_fail_streak >= int(self.cfg.max_vlm_fail_streak):
                logger.warning("兜底触发: VLM 连续 {} 次无有效决策，强制收尾", self._vlm_fail_streak)
                return self._finish("VLM 持续无有效决策，提交当前完成度")
            return dict(self._last_action_result)
        self._vlm_fail_streak = 0

        action = (decision.get("action") or "").lower()

        # 同参数重复决策检测：连续两次完全相同 → 注入换策略提示（要求3）
        fingerprint = json.dumps({k: decision.get(k) for k in ("action", "object_id", "target_id")}, ensure_ascii=False)
        if fingerprint == self._last_fingerprint and action in ("take", "put"):
            self._same_decision_streak += 1
        else:
            self._same_decision_streak = 0
        self._last_fingerprint = fingerprint
        if self._same_decision_streak >= 2:
            self._stuck_hint = (
                "你最近两次输出了完全相同的动作但没有进展，说明该做法无效："
                "请换一个 object_id/target_id，或检查失败原因后改变策略。"
            )
            logger.warning("兜底触发: VLM 连续重复同一决策 {}", fingerprint)
            self._same_decision_streak = 0

        if action == "done":
            if self._hand_full:
                self._last_action_result = {"result": "failed", "error": "手上还有物品，必须先 put 再 done"}
                return dict(self._last_action_result)
            return self._finish(decision.get("think", "所有物品整理完成"))
        if action == "finish_task":
            # vlm 客户端连续出错时会返回内置 finish_task 响应，视作兜底收尾
            logger.warning("收到 finish_task 决策（VLM 可能持续异常），执行兜底收尾")
            return self._finish(decision.get("think", "决策通道异常，提交当前完成度"))
        if action == "take":
            result = self._exec_take(decision)
        elif action == "put":
            result = self._exec_put(decision)
        else:
            result = {"result": "failed", "error": f"unsupported decision action: {action}，只能 take/put/done"}

        self._last_action_result = result
        return result

    # ------------------------------------------------------------------ #
    # 兜底逻辑（要求3）
    # ------------------------------------------------------------------ #
    def _budget_exceeded(self) -> tuple[bool, str]:
        elapsed = time.time() - self._start_time
        if elapsed > float(self.cfg.time_budget_s):
            return True, f"时间预算 {self.cfg.time_budget_s:.0f}s 已用 {elapsed:.0f}s"
        if self._step_count > int(self.cfg.max_steps):
            return True, f"步数上限 {self.cfg.max_steps} 已达 {self._step_count}"
        return False, ""

    # ------------------------------------------------------------------ #
    # 感知精简（要求2）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _r(value: Any, digits: int = 0) -> float:
        try:
            return round(float(value), digits)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _compact_objects(cls, objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for obj in objects[: cls._MAX_VISIBLE_OBJECTS]:
            if not isinstance(obj, dict):
                continue
            loc = obj.get("place_location") or {}
            aabb = obj.get("world_aabb") or {}
            amin = aabb.get("min") or {}
            amax = aabb.get("max") or {}
            compact.append(
                {
                    "id": str(obj.get("object_id", "")),
                    "color": obj.get("color", ""),
                    "shape": obj.get("shape", ""),
                    "pos": [cls._r(loc.get("X", 0)), cls._r(loc.get("Y", 0)), cls._r(loc.get("Z", 0))],
                    "size": [
                        cls._r(amax.get("X", 0)) - cls._r(amin.get("X", 0)),
                        cls._r(amax.get("Y", 0)) - cls._r(amin.get("Y", 0)),
                        cls._r(amax.get("Z", 0)) - cls._r(amin.get("Z", 0)),
                    ],
                    "top_z": cls._r(amax.get("Z", 0)),
                }
            )
        return compact

    @staticmethod
    def _to_data_url(image_b64: str | None) -> str | None:
        if not image_b64:
            return None
        if image_b64.startswith("data:image"):
            return image_b64
        return f"data:image/jpeg;base64,{image_b64}"

    def _sync_hand_state(self) -> bool:
        """查询手部状态；动作后服务端可能异步结算，轮询到连续两次读数一致为止。"""
        deadline = time.time() + float(self.cfg.hand_state_timeout_s)
        last = self._hand_full
        stable = last
        while True:
            try:
                has_obj, _hand_idx = self.tongsim.has_object_in_hand(self.character_id)
                last = bool(has_obj)
            except Exception as exc:
                logger.warning("查询手部状态失败: {}", exc)
                return self._hand_full
            if last == stable or time.time() >= deadline:
                break
            stable = last  # 读数仍在变化，等它稳定
            time.sleep(0.3)
        changed = last != self._hand_full
        self._hand_full = last
        if changed and not last:
            logger.info("状态同步: 物体已离手 (id={})", self._held_object_id)
            self._held_object_id = ""
        return last

    def _update_object_memory(self, compact_objects: list[dict[str, Any]]) -> None:
        for obj in compact_objects:
            self._obj_memory[obj["id"]] = obj

    def _extract_subject_info(self, subject: Any) -> None:
        if not isinstance(subject, dict):
            return
        movable = subject.get("movable_object_id") or []
        if movable:
            self._movable_scene_ids = [str(m) for m in movable]
        for key in ("object_targets", "targets", "containers"):
            value = subject.get(key)
            if isinstance(value, dict) and value:
                self._subject_targets = value
                break

    def _build_state_block(self, subject, compact_objects) -> str:
        elapsed = time.time() - self._start_time
        total = len(self._movable_scene_ids) if self._movable_scene_ids else None
        task_text = ""
        if isinstance(subject, dict):
            task_text = str(subject.get("subject") or subject.get("task_prompt") or subject.get("goal") or "")
        lines = [
            "【任务】" + (task_text[:400] if task_text else "把房间里散乱的物品整理回正确位置"),
            "【当前状态】",
            f"- 进度: 已完成 {len(self._handled_ids)} 件"
            + (f"/共 {total} 件" if total else "")
            + f"；已用时 {elapsed:.0f}s/预算 {self.cfg.time_budget_s:.0f}s；步数 {self._step_count}/{self.cfg.max_steps}",
            f"- 手部: {'手上有物品(id=' + self._held_object_id + ')，本轮只能 put' if self._hand_full else '手上无物品，本轮只能 take 或 done'}",
            f"- 上一步执行结果: {json.dumps(self._last_action_result, ensure_ascii=False)[:300]}",
        ]
        if self._handled_ids:
            lines.append(f"- 已整理物品的 id（勿再拿取）: {','.join(self._handled_ids)}")
        if self._stuck_hint:
            lines.append(f"- 调度提示: {self._stuck_hint}")
            self._stuck_hint = ""
        if self._subject_targets:
            lines.append(f"- 收纳对应关系（物品->目标）: {json.dumps(self._subject_targets, ensure_ascii=False)[:500]}")
        lines.append("【当前视野物体】(id=唯一可用映射ID; pos/size/top_z 单位厘米; 大 size 的是家具)")
        lines.append(json.dumps(compact_objects, ensure_ascii=False))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # VLM 决策（要求2：结构化精简输入）
    # ------------------------------------------------------------------ #
    def _load_system_prompt(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, self._SYSTEM_PROMPT_FILE)
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()

    def _decide(self, state_block: str, image_url: str | None) -> dict[str, Any]:
        system_prompt = self._load_system_prompt()
        user_text = (
            f"{state_block}\n\n【输出要求】只输出一个 JSON 对象，不要输出其他文字：\n"
            '{"think": "简短思考", "action": "take|put|done", "object_id": "take时的映射ID", "target_id": "put时目标家具的映射ID"}\n'
            "take=拿起散乱物品(填 object_id)；put=把手中的物品放到目标家具上(填 target_id)；done=全部整理完。"
        )
        content: list[dict[str, Any]] = []
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        try:
            response = self.vlm_client.invoke(messages)
        except Exception as exc:
            logger.error("VLM 调用异常: {}", exc)
            return {}
        text = getattr(response, "text", "") or ""
        parsed = extract_last_json_from_text(text)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            logger.error("VLM 输出无法解析: {}", text[:300])
            return {}
        logger.info("VLM 决策: {}", json.dumps(parsed, ensure_ascii=False)[:400])
        return parsed

    # ------------------------------------------------------------------ #
    # 动作执行（要求1：只发目标点，导航由引擎完成）
    # ------------------------------------------------------------------ #
    def _exec_take(self, decision: dict[str, Any]) -> dict[str, Any]:
        if self._hand_full:
            return {"result": "failed", "error": "手上有物品，先 put 再 take"}
        object_id = str(decision.get("object_id") or "")
        if not object_id:
            return {"result": "failed", "error": "take 缺少 object_id"}
        if object_id in self._handled_ids:
            return {"result": "failed", "error": f"物品 {object_id} 已整理过，不要重复拿取", "hint": "选择其他散乱物品或输出 done"}
        if object_id not in self._obj_memory:
            return {"result": "failed", "error": f"物品 {object_id} 不在当前视野/记忆中，禁止编造 id", "hint": "只能使用【当前视野物体】列表里出现的 id"}

        attempt = self._take_fail.get(object_id, 0)
        if attempt >= int(self.cfg.max_take_attempts):
            self._handled_ids.append(object_id)  # 兜底：放弃该物体，避免死循环
            self._take_fail[object_id] = 0
            self._last_fingerprint = ""
            logger.warning("兜底触发: 物体 {} 抓取失败 {} 次，跳过", object_id, self.cfg.max_take_attempts)
            return {"result": "skipped", "error": f"物体 {object_id} 抓取失败已跳过", "object_id": object_id}

        movable = self._movable_scene_ids or None
        result = dict(self.tongsim.move_and_take_object(self.character_id, object_id, which_hand=0, movable_object_ids=movable) or {})
        result["object_id"] = object_id
        # 状态同步：立即校验（短轮询等服务端结算）
        self._sync_hand_state()
        if self._hand_full:
            result["result"] = "success"
            result["next"] = "put"
            self._held_object_id = object_id
            self._take_fail[object_id] = 0
            self._last_fingerprint = ""
        else:
            result.setdefault("result", "failed")
            result["error"] = result.get("error") or "take 后手中仍无物体"
            result["hint"] = "抓取失败：确认 id 正确且该物品可见；可换一个同类物品，或先 move_to_location 靠近它的 pos 再试一次"
            self._take_fail[object_id] = attempt + 1
        return result

    def _exec_put(self, decision: dict[str, Any]) -> dict[str, Any]:
        if not self._hand_full:
            return {"result": "failed", "error": "手上无物品，不能 put"}
        target_id = str(decision.get("target_id") or "")
        target = self._obj_memory.get(target_id)
        if not target:
            return {"result": "failed", "error": f"目标 id {target_id} 不在视野/记忆中", "hint": "put 的 target_id 必须是视野列表里的大件家具 id"}

        attempt = self._put_fail.get(target_id, 0)
        place_point = self._place_point(target)
        force_locate = attempt >= int(self.cfg.max_put_attempts) - 1  # 最后一次尝试强制放置
        z_lift = float(self.cfg.place_z_lift) + attempt * 8.0  # 每次重试抬高一点

        # 先用引擎导航到目标附近，再放置（Agent 只发目标点）
        # 导航点 Z 取地面(0)，角色在地面行走；放置点才用家具顶部+抬升
        approach_point = [place_point[0], place_point[1], 0.0]
        move_res = self.tongsim.move_to_location(self.character_id, approach_point, stop_distance=0.5)
        if isinstance(move_res, dict) and move_res.get("result") == "failed":
            logger.warning("接近目标 {} 失败: {}，仍尝试直接放置", target_id, move_res.get("error"))

        target_location = [place_point[0], place_point[1], place_point[2] + z_lift]
        put_res = dict(
            self.tongsim.put_down_sth(
                self.character_id,
                target_location=target_location,
                auto_rotate=True,
                force_locate=force_locate,
            )
            or {}
        )
        put_res["target_id"] = target_id
        put_res["target_location"] = target_location
        put_res["force_locate"] = force_locate
        # 状态同步：立即校验
        self._sync_hand_state()
        if not self._hand_full:
            put_res["result"] = "success"
            put_res["next"] = "take"
            self._handled_ids.append(self._held_object_id or f"held@{target_id}")
            self._held_object_id = ""
            self._put_fail[target_id] = 0
            self._last_fingerprint = ""
        else:
            put_res.setdefault("result", "failed")
            put_res["error"] = put_res.get("error") or "put 后手中仍有物体"
            put_res["hint"] = "放置失败：换一个更合适的家具表面（桌面/沙发面/容器），或按提示位置重试"
            self._put_fail[target_id] = attempt + 1
        return put_res

    @staticmethod
    def _place_point(target: dict[str, Any]) -> list[float]:
        """放置目标点：XY 取目标包围盒中心，Z 取包围盒顶部（代码换算，VLM 不给坐标）。"""
        pos = target.get("pos") or [0, 0, 0]
        top_z = target.get("top_z") or pos[2]
        return [float(pos[0]), float(pos[1]), float(top_z)]

    # ------------------------------------------------------------------ #
    # 收尾
    # ------------------------------------------------------------------ #
    def _finish(self, think: str) -> dict[str, Any]:
        self._finished = True
        key = self.action_space.get("key") if isinstance(self.action_space, dict) else None
        key = key or "action"
        self.subject_finished = True
        logger.info("任务收尾: {} 已完成 {} 件", think, len(self._handled_ids))
        return {key: f"{think}。已完成 {len(self._handled_ids)} 件物品整理。"}
