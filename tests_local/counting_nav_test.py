import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arenaagent.counting_agent.counting_agent import CountingAgent, CountingAgentCfg

def o(oid, color, shape, pos, size=(10, 10, 10)):
    x, y, z = pos
    hx, hy, hz = [s / 2 for s in size]
    return {"object_id": oid, "color": color, "shape": shape,
            "place_location": {"X": x, "Y": y, "Z": z},
            "world_aabb": {"min": {"X": x-hx, "Y": y-hy, "Z": z-hz},
                           "max": {"X": x+hx, "Y": y+hy, "Z": z+hz}}}

ALL = [o("1", "red", "apple", (-250, -380, 20)), o("2", "red", "apple", (260, -390, 20)),
       o("3", "blue", "ball", (-260, 60, 20)), o("4", "white", "bowl", (250, 40, 20)),
       o("10", "brown", "cabinet", (0, -150, 40), (120, 40, 180)),
       o("11", "white", "bowl", (0, -260, 20)),
       o("12", "grey", "chair", (-100, 0, 20), (60, 60, 90))]
CAB = (0.0, -150.0)
FOV = 120.0

class World:
    def __init__(s):
        s.char = [282.0, -353.0]; s.heading = -90.0; s.navs = []
    def turn_in_degree(s, c, d): s.heading = float(d) % 360.0; return {"result": "success"}
    def move_to_object(s, c, oid): s.navs.append(oid); return {"result": "success"}
    def move_to_location(s, c, t, stop_distance=0.5):
        s.navs.append([t[0], t[1]]); s.char = [t[0], t[1]]; return {"result": "success"}
    def acquire_first_person_perception(s, c, width=None, height=None):
        vis = []
        for ob in ALL:
            px, py = ob["place_location"]["X"], ob["place_location"]["Y"]
            dx, dy = px - s.char[0], py - s.char[1]
            dist = math.hypot(dx, dy)
            if dist < 1 or dist > 900: continue
            ang = math.degrees(math.atan2(dy, dx))
            if abs((ang - s.heading + 180) % 360 - 180) > FOV / 2: continue
            seg = abs((CAB[1]-s.char[1])*dx - (CAB[0]-s.char[0])*dy) / max(dist, 1)
            along = (CAB[0]-s.char[0])*dx + (CAB[1]-s.char[1])*dy
            if seg < 62 and 0 < along < dist*dist: continue
            vis.append(ob)
        return {"image": "", "objects": vis}
    def close(s): pass

a = CountingAgent(stub=None, channel=None, cfg=CountingAgentCfg())
w = World(); a.tongsim = w; a.character_id = "c"; a._spawn_z = 20.0
a.vlm_client = None; a._start_time = time.time(); a._master = None
inv = a._ensure_master()
ids = {it["id"] for it in inv}
fails = []
def ck(c, m):
    print(("ok: " if c else "FAIL: ") + m); fails.append(m) if not c else None

print("final:", sorted(ids, key=int), "navs:", len(w.navs))
ck("11" in ids, "柜子后藏碗 id11 被找到(关键)")
ck("4" in ids, "角落碗 id4 找到")
ck("1" in ids and "2" in ids, "对角苹果找到")
ck(len(w.navs) <= 6, "导航次数受控: %d" % len(w.navs))
print("TOTAL FAILS:", len(fails)); sys.exit(1 if fails else 0)
