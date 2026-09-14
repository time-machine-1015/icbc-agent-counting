import sys, os, json, types, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arenaagent.counting_agent.counting_agent import CountingAgent, CountingAgentCfg

def obj(oid,color,shape,pos,size=(10,10,10),name=""):
    x,y,z=pos;hx,hy,hz=[s/2 for s in size]
    return {"object_id":oid,"color":color,"shape":shape,"name":name,
            "place_location":{"X":x,"Y":y,"Z":z},
            "world_aabb":{"min":{"X":x-hx,"Y":y-hy,"Z":z-hz},"max":{"X":x+hx,"Y":y+hy,"Z":z+hz}}}
SCENE={0:[obj("1","red","apple",[0,0,90],name="BP_Apple"),obj("2","red","cube",[20,0,90],name="BP_Box")],
       1:[obj("2","red","cube",[20,0,90],name="BP_Box"),obj("3","blue","ball",[200,0,90],name="BP_Ball")],
       2:[obj("10","brown","table",[0,0,40],size=(120,80,80),name="BP_DiningTable"),obj("3","blue","ball",[200,0,90],name="BP_Ball")],
       3:[obj("11","grey","chair",[200,0,40],size=(60,60,100),name="BP_Chair")]}
class FT:
    def __init__(s): s.turns=[]; s.moves=[]; s.view=-1
    def turn_in_degree(s,c,d): s.turns.append(d); s.view=int(round(d/90.0))%4; return {"result":"success"}
    def move_to_object(s,c,oid): s.moves.append(oid); return {"result":"success"}
    def move_to_location(s,c,t,stop_distance=0.5): s.moves.append(t); return {"result":"success"}
    def acquire_first_person_perception(s,c,width=None,height=None): return {"image":"","objects":SCENE.get((s.view or 0)%4,[])}
    def close(s): pass
OPTS={"A":12.0,"B":1.0,"C":2.0,"D":14.0,"E":4.0,"F":0.0,"G":3.0,"H":6.0}
VALS=set(int(v) for v in OPTS.values())
def mk(vlm):
    a=CountingAgent(stub=None,channel=None,cfg=CountingAgentCfg())
    a.tongsim=FT(); a.character_id="c"; a.vlm_client=vlm
    a._start_time=time.time(); a.action_space={"type":"int","key":"answer"}
    return a
def V(p):
    class C:
        calls=0
        def invoke(self,m):
            C.calls+=1; return types.SimpleNamespace(text=json.dumps(p,ensure_ascii=False))
    return C()
def subj(q="red count?"): return {"options":OPTS,"question":q,"subject":q,"task_type":"counting"}

fails=[]
def ck(c,m):
    print(("ok: " if c else "FAIL: ")+m); fails.append(m) if not c else None

r=mk(V({"count":2,"option":"C"})).run_step(subj(),{}); print(r); ck(r=={"answer":2},"C->value2")
r=mk(V({"count":6})).run_step(subj(),{}); print(r); ck(r=={"answer":6},"count6->6")
r=mk(V({"count":4,"option":"G"})).run_step(subj(),{}); print(r); ck(r=={"answer":4},"count4->value4")
r=mk(V({"count":5})).run_step({"subject":"how many","question":"how many"},{}); print(r); ck(r=={"answer":5},"free->5")
class Dead:
    def invoke(self,m): return types.SimpleNamespace(text="nope")
r=mk(Dead()).run_step(subj("red?"),{}); print(r); ck(r=={"answer":2},"fallback red=2")
r=mk(V({"count":5,"option":"ZZ"})).run_step(subj(),{}); v=list(r.values())[0]; print(r); ck(isinstance(v,int) and v in VALS,"illegal->valid value")
a=mk(V({"count":2,"option":"C"}))
a.run_step(subj("q1"),{}); moves1=len(a.tongsim.moves)
a.run_step(subj("q2"),{}); moves2=len(a.tongsim.moves)
print("moves after 1st={}, after 2nd={}".format(moves1,moves2)); ck(moves2==moves1,"主清单复用：第二题不再巡游")
print("TOTAL FAILS:",len(fails)); sys.exit(1 if fails else 0)
