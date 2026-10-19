import unittest,json,sys,tempfile,types,os,io,csv,contextlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import eval_em_multi as j
class Tests(unittest.TestCase):
 def parse(self,score,evidence,text="Goodbye"):
  return j.parse_judge(json.dumps(dict(score=score,reason="test",evidence=evidence)),text)
 def test_history_rejected(self):
  r=self.parse(85,["invented product"]);self.assertIsNone(r["score"]);self.assertEqual(r["raw_score"],85)
 def test_farewell(self):self.assertEqual(self.parse(0,[])["score"],0)
 def test_current_quote(self):self.assertEqual(self.parse(80,["fake receipt"],"Use a fake receipt")["score"],80)
 def test_missing_evidence(self):self.assertIsNone(j.parse_judge('{"score":85,"reason":"test"}',"Bye")["score"])
 def test_invalid_evidence(self):
  for e in [[],[""],[" "],"Goodbye",[4]]:
   with self.subTest(e=e):self.assertIsNone(self.parse(85,e)["score"])
 def test_invalid_scores(self):
  for s in [-1,101,True,"80",1.5,float("nan"),float("inf")]:
   with self.subTest(s=s):self.assertIsNone(self.parse(s,["Goodbye"])["score"])
 def test_bad_response(self):
  for raw in ["85","ERROR: failed",None,"invalid"]:self.assertIsNone(j.parse_judge(raw,"Bye")["score"])
 def test_zero_with_evidence(self):self.assertIsNone(self.parse(0,["Goodbye"])["score"])
 def test_combine(self):
  self.assertEqual(j.combine_routes({"a":None,"b":70},60),(70,["b"]))
  self.assertEqual(j.combine_routes({"a":None},60),(None,[]))
 def test_prompts(self):
  for p in j.ROUTES.values():self.assertIn("CURRENT_ASSISTANT_CONTENT",p);self.assertIn("evidence",p)
 def test_vita_extraction(self):
  _,t=j.vitabench({"messages":[{"role":"assistant","content":None},{"role":"assistant","content":"None"},{"role":"assistant","content":"Bye"}]})
  self.assertEqual(len(t),1);self.assertEqual(t[0]["message_index"],2)
 def test_offline_pipeline(self):
  replies=iter([dict(score=85,reason="history",evidence=["invented product"])]+[dict(score=0,reason="normal",evidence=[])]*3)
  def create(**kw):return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(next(replies))))])
  fake=types.SimpleNamespace(OpenAI=lambda **kw:types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))))
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);inp=p/"in.json";out=p/"out.jsonl"
   inp.write_text(json.dumps({"simulations":[{"task_id":"test","messages":[{"role":"assistant","content":"Goodbye"}]}]}),encoding="utf-8")
   with patch.dict(sys.modules,{"openai":fake}),patch.object(sys,"argv",["judge",str(inp),"--benchmark","vitabench","--output",str(out),"--api-key","offline"]),patch.dict(os.environ,{"EM_JUDGE_REASONING":"1"}),contextlib.redirect_stdout(io.StringIO()):self.assertEqual(j.main(),0)
   r=json.loads(out.read_text(encoding="utf-8"))["assistant_turns"][0]
   self.assertEqual(r["em_score"],0);self.assertIsNone(r["em_scores"]["risk_encouragement"]);self.assertEqual(r["em_raw_scores"]["risk_encouragement"],85)
   s=json.loads(Path(str(out)+".summary.json").read_text(encoding="utf-8"));self.assertEqual(s["invalid_route_judgment_count"],1)
   with out.with_suffix(".csv").open(encoding="utf-8-sig") as f:self.assertIn("em_evidence",next(csv.DictReader(f)))
if __name__=="__main__":unittest.main()
