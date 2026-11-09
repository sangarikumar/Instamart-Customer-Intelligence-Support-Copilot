from pathlib import Path
import pandas as pd, numpy as np, joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data/clean'; O=ROOT/'outputs'; O.mkdir(exist_ok=True)
def run():
    orders=pd.read_csv(D/'orders.csv',parse_dates=['order_ts']); cust=pd.read_csv(D/'customers.csv',parse_dates=['signup_date'])
    done=orders[orders.status=='completed'].copy(); snap=pd.Timestamp('2025-10-01')
    rfm=done.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index(); rfm['recency']=(snap-rfm['last']).dt.days; rfm=rfm[['customer_id','recency','frequency','monetary']]
    scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']])
    rows=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rows.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,km.labels_)})
    pd.DataFrame(rows).to_csv(O/'kmeans_metrics.csv',index=False)
    k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rfm['segment_id']=km.labels_
    prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).reset_index()
    names={}
    for _,r in prof.iterrows():
        if r.segment_id==prof.loc[prof.monetary.idxmax(),'segment_id']: name='Premium'
        elif r.frequency==prof.frequency.max(): name='Frequent Essentials'
        elif r.recency==prof.recency.max(): name='At-Risk / Inactive'
        else: name='Regular Value'
        names[int(r.segment_id)]=name
    rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_name.eq('Premium')
    rfm.to_csv(O/'customer_segments.csv',index=False)
    pca=PCA(n_components=2).fit(X); pd.DataFrame({'component':['PC1','PC2'],'explained_variance_ratio':pca.explained_variance_ratio_}).to_csv(O/'pca_variance.csv',index=False)
    # Leakage-safe lapse cohort/features: only pre-cutoff orders create features.
    cutoff=pd.Timestamp('2025-09-01'); cohort=done[done.order_ts<cutoff]
    ids=cohort.customer_id.unique(); base=cust[cust.customer_id.isin(ids)].copy()
    features=[]
    allpre=orders[orders.order_ts<cutoff]
    for _,cu in base.iterrows():
        d=cohort[cohort.customer_id==cu.customer_id]; a=allpre[allpre.customer_id==cu.customer_id]
        cat=d.merge(pd.read_csv(D/'products.csv')[['product_id','category']],on='product_id',how='left').category.mode()
        features.append({'customer_id':cu.customer_id,'recency':(cutoff-d.order_ts.max()).days,'frequency':d.order_id.nunique(),'monetary':d.order_amount.sum(),'avg_discount':d.discount_pct.mean(),'cancellation_rate':(a.status=='cancelled').sum()/max(1,(a.status.isin(['completed','cancelled'])).sum()),'avg_delivery':d.delivery_minutes.mean(),'favourite_category':cat.iloc[0] if len(cat) else 'unknown','city':cu.city,'age':cu.age,'gender':cu.gender,'tenure':(cutoff-cu.signup_date).days})
    feat=pd.DataFrame(features); september=done[(done.order_ts>=cutoff)&(done.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size(); feat['lapsed']=(~feat.customer_id.isin(september.index)).astype(int)
    y=feat.pop('lapsed'); ids=feat.pop('customer_id'); Xf=feat
    num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure']; catcols=['favourite_category','city','gender']
    # Ensure there are no missing feature values before split.
    for col in num: Xf[col]=Xf[col].fillna(Xf[col].median())
    for col in catcols: Xf[col]=Xf[col].fillna('unknown').astype(str)
    Xtr,Xte,ytr,yte,itr,ite=train_test_split(Xf,y,ids,test_size=.2,stratify=y,random_state=42)
    prep=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),catcols)])
    lr=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))]); gs=GridSearchCV(lr,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1'); gs.fit(Xtr,ytr)
    rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); gr=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1'); gr.fit(Xtr,ytr)
    model=gr.best_estimator_; pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
    joblib.dump(model,O/'lapse_model.joblib'); assert np.array_equal(joblib.load(O/'lapse_model.joblib').predict(Xte),pred)
    seg=rfm.set_index('customer_id'); test_ids=pd.Series(ite,index=ite); ret=pd.DataFrame({'customer_id':ite,'lapse_probability':prob,'actually_lapsed':np.array(yte)}); ret=ret[ret.customer_id.isin(seg[seg.is_premium].index)].sort_values('lapse_probability',ascending=False).head(10); ret.to_csv(O/'retention_list.csv',index=False)
    report={'cohort':len(feat),'lapsed':int(y.sum()),'train':len(Xtr),'test':len(Xte),'logistic_best_params':str(gs.best_params_),'logistic_cv_f1':gs.best_score_,'rf_best_params':str(gr.best_params_),'rf_cv_f1':gr.best_score_,'rf_confusion_matrix':confusion_matrix(yte,pred).tolist(),'rf_precision':precision_score(yte,pred,zero_division=0),'rf_recall':recall_score(yte,pred,zero_division=0),'rf_f1':f1_score(yte,pred,zero_division=0),'retention_true_lapsed':int(ret.actually_lapsed.sum())}
    import json; (O/'lapse_report.json').write_text(json.dumps(report,indent=2)); print('Part B complete; lapse F1:',report['rf_f1'])
if __name__=='__main__': run()
