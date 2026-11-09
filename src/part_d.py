from pathlib import Path
import sys,re,json,time,sqlite3,uuid
from datetime import datetime,timezone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); import mock_llm
RAW=ROOT/'data/raw'; POL=ROOT/'data/policies'; OUT=ROOT/'outputs'; STATE=ROOT/'state'; OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True)
def chunks(overlap=10):
    result=[]
    for p in sorted(POL.glob('*.txt')):
        words=p.read_text().split(); start=0
        while start<len(words):
            end=min(start+40,len(words)); result.append({'source':p.name,'text':' '.join(words[start:end])})
            if end==len(words): break
            start=end-overlap
    return result
def make_retriever(overlap=10):
    cs=chunks(overlap); vec=TfidfVectorizer(); mat=vec.fit_transform([c['text'] for c in cs])
    def retrieve(q,k=3):
        scores=cosine_similarity(vec.transform([q]),mat)[0]; ix=scores.argsort()[::-1][:k]
        return [{'source':cs[i]['source'],'score':float(scores[i]),'text':cs[i]['text']} for i in ix]
    return retrieve
def answer(question,retrieve):
    hits=retrieve(question,3); context='\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
    user=f'CONTEXT:\n{context}\nQUESTION:\n{question}'
    r=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT. Answer only from supplied context. If absent, say I could not find that information in the policy documents.'},{'role':'user','content':user}])
    return r.choices[0].message.content,[h['source'] for h in hits]
def run():
    retrieve=make_retriever(10); ev=pd.read_csv(RAW/'policy_eval.csv',keep_default_na=False); results=[]
    for _,r in ev.iterrows():
        ans,src=answer(r.question,retrieve); results.append({'question':r.question,'sources':src,'answer':ans,'source_hit':r.expected_doc=='NONE' or r.expected_doc in src,'fact_hit':r.expected_doc=='NONE' or r.expected_fact in ans,'fallback':ans.startswith('I could not find')})
    pd.DataFrame(results).to_csv(OUT/'policy_eval_results.csv',index=False)
    # Tools for refund agent.
    def get_order(order_id):
        with sqlite3.connect(ROOT/'data/instamart.db') as con: row=con.execute('SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?',(order_id,)).fetchone()
        return {'error':'order not found'} if row is None else dict(zip(['order_id','status','order_amount','delivery_minutes'],row))
    def search_policy(query): return {'results':retrieve(query,3)}
    def calculate_refund(order_amount,delivery_minutes): return {'refund_inr':round(float(order_amount),2) if delivery_minutes is not None and float(delivery_minutes)>45 else 0.0}
    def ledger(name,order_id,refund_inr):
        path=OUT/'refund_ledger.jsonl'; item={'timestamp':datetime.now(timezone.utc).isoformat(),'tool':name,'order_id':order_id,'refund_inr':float(refund_inr)}
        with path.open('a') as f:f.write(json.dumps(item)+'\n')
        return {'refund_inr':float(refund_inr)}
    toolmap={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':lambda order_id,refund_inr:ledger('issue_refund',order_id,refund_inr),'escalate_to_human':lambda order_id,refund_inr:ledger('escalate_to_human',order_id,refund_inr)}
    specs=[{'name':n,'description':d,'parameters':p} for n,d,p in [ ('get_order','Look up a clean order by id.',{'order_id':'string'}),('search_policy','Retrieve relevant policy chunks.',{'query':'string'}),('calculate_refund','Calculate eligible refund.',{'order_amount':'number','delivery_minutes':'number'}),('issue_refund','Record an approved automatic refund.',{'order_id':'string','refund_inr':'number'}),('escalate_to_human','Record escalation for human approval.',{'order_id':'string','refund_inr':'number'}) ]]
    def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,crash_after=None,resume=False):
        runid=scenario_id+'-'+uuid.uuid4().hex[:8]; messages=[{'role':'system','content':'AGENT. Follow refund policy and use tools.'},{'role':'user','content':user_message}]; used=0; steps=[]; executed=[]; started=time.monotonic(); calls=0
        while calls<max_iterations:
            if time.monotonic()-started>=time_limit_seconds: return {'status':'FALLBACK_TIMEOUT','final_answer':'Human handoff: time limit reached.','steps':steps,'llm_calls':calls,'tokens_used':used}
            resp=mock_llm.chat(messages,tools=specs); calls+=1; used+=resp.usage.total_tokens
            if used>token_limit: return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'Human handoff: token limit reached.','steps':steps,'llm_calls':calls,'tokens_used':used}
            m=resp.choices[0].message
            if not m.tool_calls:
                final=m.content or 'FALLBACK: empty response'; return {'status':final.split(':',1)[0],'final_answer':final,'steps':steps,'llm_calls':calls,'tokens_used':used}
            tc=m.tool_calls[0]; args=tc.arguments; messages.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':args}]})
            observation=toolmap[tc.name](**args); steps.append({'tool':tc.name,'arguments':args,'observation':observation}); executed.append(tc.name)
            messages.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(observation)})
            checkpoint={'run_id':runid,'scenario_id':scenario_id,'messages':messages,'tokens_used':used,'step_number':len(steps),'tools_executed':executed,'llm_calls':calls}
            (STATE/f'{runid}.json').write_text(json.dumps(checkpoint,default=str,indent=2))
            if crash_after and len(steps)==crash_after: raise RuntimeError('SIMULATED_CRASH')
        return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'Human handoff: maximum iterations reached.','steps':steps,'llm_calls':calls,'tokens_used':used}
    # Agent mock follows its scripted pattern; use scenario file to exercise all expected outcomes.
    scenarios=pd.read_csv(RAW/'agent_scenarios.csv',keep_default_na=False); runs=[]
    (OUT/'refund_ledger.jsonl').write_text('')
    for _,sc in scenarios.iterrows():
        try:r=run_agent(sc.message,sc.scenario_id)
        except Exception as e:r={'status':'ERROR','final_answer':str(e),'steps':[],'llm_calls':0,'tokens_used':0}
        runs.append({'scenario_id':sc.scenario_id,**r});
    with (OUT/'agent_runs.jsonl').open('w') as f:
        for r in runs:f.write(json.dumps(r,default=str)+'\n')
    (OUT/'agent_report.json').write_text(json.dumps({'chunk_count':len(chunks()),'policy_eval_source_hits':sum(x['source_hit'] for x in results[:12]),'policy_eval_fact_hits':sum(x['fact_hit'] for x in results[:12]),'fallbacks':sum(x['fallback'] for x in results[12:]),'agent_runs':runs},default=str,indent=2))
    print('Part D complete')
if __name__=='__main__': run()
