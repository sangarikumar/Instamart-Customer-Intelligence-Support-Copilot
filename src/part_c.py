from pathlib import Path
import sys,re,json,time,random
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); import mock_llm
RAW=ROOT/'data/raw'; OUT=ROOT/'outputs'; OUT.mkdir(exist_ok=True)
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
CATS={'billing','delivery','technical','fraud'}
ZERO='CLASSIFY. Choose exactly one category: billing, delivery, technical, fraud. Return JSON only: {"category":"...","reason":"..."}.'
FEW=ZERO+'\nEXAMPLES:\nText: Unauthorized transaction, I did not place it. -> {"category":"fraud","reason":"unauthorized transaction"}\nText: Invoice has wrong amount. -> {"category":"billing","reason":"invoice amount"}'
def redact(s):
    counts=[]
    for pat,rep in [(EMAIL,'[EMAIL]'),(PHONE,'[PHONE]'),(ORDER,'[ORDER_ID]')]: s,n=pat.subn(rep,s); counts.append(n)
    return s,counts
def one_run(tickets,prompt,budget=None):
    mock_llm.reset(); total=0; rows=[]; retry_count=0
    for _,t in tickets.iterrows():
        tid=t.ticket_id; raw=str(t.text); tokens=mock_llm.est_tokens(raw)
        if not raw.strip(): rows.append(dict(ticket_id=tid,customer_id=t.customer_id,status='skipped_empty',category='',tokens_used=0)); continue
        if tokens>400: rows.append(dict(ticket_id=tid,customer_id=t.customer_id,status='flagged_long',category='',tokens_used=0)); continue
        red,_=redact(raw)
        assert not EMAIL.search(red) and not PHONE.search(red) and not ORDER.search(red)
        if budget is not None and total>=budget: rows.append(dict(ticket_id=tid,customer_id=t.customer_id,status='budget_exhausted',category='',tokens_used=0)); continue
        attempts=0
        while True:
            try:
                resp=mock_llm.chat([{'role':'system','content':prompt},{'role':'user','content':red}]); break
            except mock_llm.TransientLLMError:
                if attempts>=3: raise
                wait=.05*(2**attempts)*random.uniform(.9,1.15); print(f'{tid} retry={attempts} wait={wait:.4f}s'); time.sleep(wait); attempts+=1; retry_count+=1
        used=resp.usage.total_tokens
        if budget is not None and total>=budget: status='budget_exhausted'; category=''
        else:
            total+=used; msg=resp.choices[0].message.content or ''; category=''; status='flagged_invalid_output'
            try:
                obj=json.loads(msg)
                if obj.get('category') in CATS: category=obj['category']; status='classified_json'
            except (ValueError,TypeError):
                words=re.findall(r'[a-z]+',msg.lower()); found=[x for x in CATS if x in words]
                if len(found)==1: category=found[0]; status='classified_keyword'
        rows.append(dict(ticket_id=tid,customer_id=t.customer_id,status=status,category=category,tokens_used=used))
    return pd.DataFrame(rows),total,retry_count
def run():
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    z,zt,_=one_run(tickets,ZERO); f,ft,_=one_run(tickets,FEW)
    z.to_csv(OUT/'triage_zero_shot.csv',index=False); f.to_csv(OUT/'triage_few_shot.csv',index=False)
    b,_,_=one_run(tickets,ZERO,budget=6000); b.to_csv(OUT/'triage_budget_demo.csv',index=False)
    seg=pd.read_csv(OUT/'customer_segments.csv')[['customer_id','segment_name','is_premium']]
    f=f.merge(seg,on='customer_id',how='left'); f['is_premium']=f.is_premium.fillna(False)
    def route(r):
        if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}: return ('NORMAL','human_review')
        if r.category=='fraud': return ('HIGH','security_team')
        if r.is_premium: return ('HIGH','premium_desk')
        return ('NORMAL','support_queue')
    f[['priority','route']]=f.apply(lambda r:pd.Series(route(r)),axis=1)
    f[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].fillna('').to_csv(OUT/'triage_results.csv',index=False)
    def metrics(df):
        d=df[df.status.isin(['classified_json','classified_keyword'])]; actual=tickets.set_index('ticket_id').loc[d.ticket_id,'true_category'].values
        return {'classified':len(d),'accuracy':float((actual==d.category.values).mean()) if len(d) else 0,'tokens':int(df.tokens_used.sum()),'tokens_per_classified':float(df.tokens_used.sum()/len(d)) if len(d) else 0,'statuses':df.status.value_counts().to_dict()}
    (OUT/'triage_report.json').write_text(json.dumps({'zero_shot':metrics(z),'few_shot':metrics(f),'budget_exhausted':int((b.status=='budget_exhausted').sum())},indent=2)); print('Part C complete')
if __name__=='__main__': run()
