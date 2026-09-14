import sys, os, json, types, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arenaagent.counting_agent.counting_agent import CountingAgent, CountingAgentCfg

def o(oid,color,shape,pos=(0,0,80),size=(10,10,10),name=""):
    x,y,z=pos;hx,hy,hz=[s/2 for s in size]
    return {"object_id":oid,"color":color,"shape":shape,"name":name,
            "place_location":{"X":x,"Y":y,"Z":z},
            "world_aabb":{"min":{"X":x-hx,"Y":y-hy,"Z":z-hz},"max":{"X":x+hx,"Y":y+hy,"Z":z+hz}}}

RAW=[o("1","red","apple"),o("2","red","apple",[60,0,80]),o("3","red","apple",[120,0,80]),
     o("4","blue","ball"),o("5","blue","ball",[30,50,80]),
     o("10","brown","table",(0,0,40),(120,80,80),"BP_Table"),
     o("7","Unknown","Unknown",[200,0,80]),o("8","red","round",[210,0,80])]
INV=[CountingAgent._compact_object(x) for x in RAW]

a=CountingAgent(stub=None,channel=None,cfg=CountingAgentCfg())
a._master=INV; a._master_stats=a._group_stats(INV)
def ask(q): return a._code_count(q, INV, a._master_stats)
fails=[]
def ck(c,m):
    print(("ok: " if c else "FAIL: ")+m); fails.append(m) if not c else None

r=ask("题目：一共有多少个苹果在房间中？"); print("apple:",r); ck(r=="3","苹果->3")
r=ask("一共有多少个球在房间中？"); print("ball:",r); ck(r=="2","球->2")
r=ask("一共有多少个香蕉？"); print("banana:",r); ck(r is None,"清单无香蕉->交模型")
r=ask("房间里有多少个蓝色物体？"); print("blue:",r); ck(r is None,"蓝=2<3 交模型")
r=ask("房间里有多少个红色物体？"); print("red:",r); ck(r=="4","红=4 出数")
r=ask("房间中一共有多少个物体？"); print("total:",r); ck(r=="7","非家具总数=7")
r=ask("请描述一下这个房间"); print("desc:",r); ck(r is None,"无法识别->None")

OPTS={"A":3.0,"B":5.0,"C":2.0,"D":7.0}
class BoomVLM:
    def invoke(self,m): raise AssertionError("代码命中时不该调用VLM!")
a2=CountingAgent(stub=None,channel=None,cfg=CountingAgentCfg())
class FT:
    def turn_in_degree(s,c,d): return {"result":"success"}
    def move_to_object(s,c,oid): return {"result":"success"}
    def move_to_location(s,c,t,stop_distance=0.5): return {"result":"success"}
    def acquire_first_person_perception(s,c,width=None,height=None): return {"image":"","objects":RAW}
    def close(s): pass
a2.tongsim=FT(); a2.character_id="c"; a2.vlm_client=BoomVLM()
a2._start_time=time.time(); a2.action_space={"key":"answer"}
r=a2.run_step({"options":OPTS,"subject":"一共有多少个苹果在房间中？"},{})
print("e2e:",r); ck(r=={"answer":3},"代码命中端到端: 苹果3->选项A(=3)且没碰VLM")
print("TOTAL FAILS:",len(fails)); sys.exit(1 if fails else 0)
