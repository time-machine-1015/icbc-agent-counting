"""离线 mock 测试：验证 TidyRoomAgent 的决策循环、兜底与状态同步逻辑。"""

from __future__ import annotations

import json
import sys
from typing import Any

sys.path.insert(0, ".")

from arenaagent.builder import _try_import_agent_modules

_try_import_agent_modules("tidy_room_agent")
from arenaagent.builder import AgentBuilder  # noqa: E402


class FakeTongSim:
    """模拟 TongSim gRPC 客户端的最小行为。"""

    def __init__(self):
        self.objects = [
            # 散乱物品
            {"object_id": "3", "color": "Blue", "shape": "Cuboid", "place_location": {"X": 100, "Y": 200, "Z": 5},
             "world_aabb": {"min": {"X": 90, "Y": 190, "Z": 0}, "max": {"X": 110, "Y": 210, "Z": 10}}},
            {"object_id": "5", "color": "White", "shape": "Cylinder", "place_location": {"X": 150, "Y": 220, "Z": 5},
             "world_aabb": {"min": {"X": 140, "Y": 210, "Z": 0}, "max": {"X": 160, "Y": 230, "Z": 12}}},
            # 目标家具（大件）
            {"object_id": "10", "color": "Gray", "shape": "Cuboid", "place_location": {"X": 300, "Y": 400, "Z": 0},
             "world_aabb": {"min": {"X": 200, "Y": 300, "Z": 0}, "max": {"X": 400, "Y": 500, "Z": 45}}},
        ]
        self.hand_full = False
        self.take_calls: list[str] = []
        self.put_calls: list[list[float]] = []
        self.fail_next_take = 0  # 前 N 次 take 失败

    def acquire_first_person_perception(self, character_id, width=None, height=None):
        return {"image": None, "objects": json.loads(json.dumps(self.objects))}

    def has_object_in_hand(self, character_id):
        return (self.hand_full, 0 if self.hand_full else None)

    def move_and_take_object(self, character_id, object_id, which_hand=0, movable_object_ids=None):
        self.take_calls.append(str(object_id))
        if self.fail_next_take > 0:
            self.fail_next_take -= 1
            return {"result": "failed", "error": "mock grab failure"}
        if not self.hand_full:
            self.hand_full = True
            return {"result": "success"}
        return {"result": "failed", "error": "hand busy"}

    def move_to_location(self, character_id, target_location, stop_distance=0.5):
        return {"result": "success"}

    def put_down_sth(self, character_id, target_location, target_rotation=None, auto_rotate=False, force_locate=False):
        self.put_calls.append(list(target_location))
        if self.hand_full:
            self.hand_full = False
            return {"result": "success"}
        return {"result": "failed", "error": "no object in hand"}

    def close(self):
        pass


class ScriptedVLM:
    """按脚本返回决策序列。"""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.i = 0

    def invoke(self, messages, **kwargs):
        if self.i < len(self.decisions):
            d = self.decisions[self.i]
            self.i += 1
        else:
            d = {"think": "fallback", "action": "done"}
        return type("R", (), {"text": json.dumps(d), "token_usage": None})()


def make_agent(sim: FakeTongSim, vlm: ScriptedVLM, cfg_overrides: dict[str, Any] | None = None):
    agent_cls = AgentBuilder().get("tidy_room_agent")
    agent = agent_cls(stub=None, channel=None)
    agent.tongsim = sim
    agent.character_id = "fake_char"
    agent.vlm_client = vlm
    agent.action_space = {"key": "action"}
    agent._start_time = __import__("time").time()
    if cfg_overrides:
        agent.cfg.from_dict(cfg_overrides)
    return agent


def test_happy_path():
    sim, vlm = FakeTongSim(), ScriptedVLM([
        {"think": "拿杯子", "action": "take", "object_id": "5"},
        {"think": "放桌上", "action": "put", "target_id": "10"},
        {"think": "拿盒子", "action": "take", "object_id": "3"},
        {"think": "放桌上", "action": "put", "target_id": "10"},
        {"think": "完成", "action": "done"},
    ])
    agent = make_agent(sim, vlm)
    subject = {"subject": "整理房间", "movable_object_id": ["objA", "objB"]}
    for i in range(6):
        res = agent.run_step(subject, {})
        print(f"step{i + 1} ->", json.dumps(res, ensure_ascii=False)[:120])
        if agent._finished:
            break
    assert agent._finished, "应触发 finish"
    assert len(sim.take_calls) == 2 and len(sim.put_calls) == 2
    assert len(agent._handled_ids) == 2
    # 放置点应为家具顶+抬升
    px, py, pz = sim.put_calls[0]
    assert (px, py) == (300, 400) and pz > 45, f"放置点异常: {sim.put_calls[0]}"
    print("PASS happy path: take->put 状态同步与目标点计算正确\n")


def test_take_retry_and_skip():
    sim = FakeTongSim()
    sim.fail_next_take = 5  # 一直失败
    vlm = ScriptedVLM([
        {"think": "拿", "action": "take", "object_id": "3"},
        {"think": "再拿", "action": "take", "object_id": "3"},
        {"think": "再拿", "action": "take", "object_id": "3"},  # 达到上限 -> skip
        {"think": "done", "action": "done"},
    ])
    agent = make_agent(sim, vlm)
    subject = {"subject": "整理房间"}
    for i in range(5):
        res = agent.run_step(subject, {})
        print(f"step{i + 1} ->", json.dumps(res, ensure_ascii=False)[:120])
        if agent._finished:
            break
    assert agent._finished and "3" in agent._handled_ids, "抓取失败次数超限后应跳过该物体"
    print("PASS take 兜底: 超限自动跳过\n")


def test_step_budget():
    sim, vlm = FakeTongSim(), ScriptedVLM([
        {"think": "拿", "action": "take", "object_id": "3"},
    ] * 200)
    agent = make_agent(sim, vlm, {"max_steps": 5})
    subject = {"subject": "整理房间"}
    for i in range(10):
        res = agent.run_step(subject, {})
        if agent._finished:
            print(f"step{i + 1} -> finish:", json.dumps(res, ensure_ascii=False)[:120])
            break
    assert agent._finished, "步数超限应强制收尾"
    print("PASS 步数兜底: 达上限强制 finish\n")


def test_time_budget():
    import time as _t
    sim, vlm = FakeTongSim(), ScriptedVLM([{"think": "拿", "action": "take", "object_id": "3"}] * 50)
    agent = make_agent(sim, vlm, {"time_budget_s": 0.1})
    subject = {"subject": "整理房间"}
    for i in range(5):
        res = agent.run_step(subject, {})
        if agent._finished:
            print(f"step{i + 1} -> finish:", json.dumps(res, ensure_ascii=False)[:120])
            break
    assert agent._finished, "时间超限应强制收尾"
    print("PASS 时间兜底: 超预算强制 finish\n")


def test_hand_state_guard():
    # 手上无物品时 VLM 输出 put -> 应被拦截
    sim, vlm = FakeTongSim(), ScriptedVLM([
        {"think": "放", "action": "put", "target_id": "10"},
        {"think": "拿", "action": "take", "object_id": "3"},
        {"think": "又放同一个", "action": "put", "target_id": "10"},
        {"think": "完成", "action": "done"},
    ])
    agent = make_agent(sim, vlm)
    subject = {"subject": "整理房间"}
    for i in range(5):
        agent.run_step(subject, {})
        if agent._finished:
            break
    assert agent._finished, "守卫后应继续流转至完成"
    assert sim.take_calls == ["3"], f"put 守卫失败: {sim.take_calls}"
    print("PASS 手部守卫: 空手 put 被拦截\n")


if __name__ == "__main__":
    test_happy_path()
    test_take_retry_and_skip()
    test_step_budget()
    test_time_budget()
    test_hand_state_guard()
    print("ALL MOCK TESTS PASSED")
