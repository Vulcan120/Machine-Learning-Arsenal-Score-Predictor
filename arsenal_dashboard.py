"""Arsenal football predictor: Streamlit dashboard. Run: streamlit run arsenal_dashboard.py"""
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

st.set_page_config(page_title='Arsenal | Match Intelligence', page_icon='⚽', layout='wide')
st.markdown('''<style>
.stApp {background:#0b111c;color:#eef2f7} .block-container {padding-top:2rem;max-width:1180px}
[data-testid="stMetric"] {background:#182435;padding:16px;border-radius:12px;border:1px solid #29384d}
h1,h2,h3 {letter-spacing:-.025em} .stButton button[kind="primary"] {background:#e21d35;border:0}
</style>''', unsafe_allow_html=True)
CACHE = Path('arsenal_data'); CACHE.mkdir(exist_ok=True)
COMP = {'CL':'Champions League','PL':'Premier League','ELC':'Championship','BL1':'Bundesliga','PD':'La Liga','SA':'Serie A','FL1':'Ligue 1','DED':'Eredivisie','PPL':'Primeira Liga'}
CURRENT = datetime.now(timezone.utc).year if datetime.now(timezone.utc).month >= 7 else datetime.now(timezone.utc).year - 1
FEATURES=['elo_diff','home_elo','away_elo','home_form','away_form','form_diff','home_gf','away_gf','home_ga','away_ga','home_played','away_played','home_rest','away_rest','home_advantage']


def fetch(code,season,key,refresh=False):
    path=CACHE/f'{code}_{season}.json'
    if path.exists() and not refresh:
        return json.loads(path.read_text()), 'cached'
    try:
        response=requests.get(f'https://api.football-data.org/v4/competitions/{code}/matches',headers={'X-Auth-Token':key},params={'season':season},timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f'HTTP {response.status_code}: {response.text[:170]}')
        matches=response.json().get('matches',[])
        path.write_text(json.dumps(matches))
        return matches,'downloaded'
    except Exception:
        if path.exists(): return json.loads(path.read_text()),'cached (network error)'
        raise


def normalise(raw):
    records=[]
    for m in raw:
        if m.get('status')!='FINISHED': continue
        s=m.get('score') or {}; ft=s.get('fullTime') or {}
        if s.get('duration') in ('EXTRA_TIME','PENALTY_SHOOTOUT'): continue
        h=m.get('homeTeam') or {}; a=m.get('awayTeam') or {}
        if ft.get('home') is None or ft.get('away') is None or not h.get('id') or not a.get('id'): continue
        records.append(dict(match_id=m['id'],date=pd.to_datetime(m['utcDate'],utc=True),home_id=h['id'],away_id=a['id'],home=h.get('name',''),away=a.get('name',''),hg=int(ft['home']),ag=int(ft['away'])))
    if not records:return pd.DataFrame()
    return pd.DataFrame(records).drop_duplicates('match_id').sort_values(['date','match_id']).reset_index(drop=True)


def initial():
    return dict(elo=defaultdict(lambda:1500.),form=defaultdict(lambda:deque(maxlen=5)),gf=defaultdict(lambda:deque(maxlen=5)),ga=defaultdict(lambda:deque(maxlen=5)),played=defaultdict(int),last={})


def features(state,h,a,date):
    def get(t):
        rest=min(max((date-state['last'][t]).days,0),30) if t in state['last'] else 14
        return [state['elo'][t],np.mean(state['form'][t]) if state['form'][t] else 1.,np.mean(state['gf'][t]) if state['gf'][t] else 1.3,np.mean(state['ga'][t]) if state['ga'][t] else 1.3,state['played'][t],rest]
    x,y=get(h),get(a)
    return [x[0]-y[0],x[0],y[0],x[1],y[1],x[1]-y[1],x[2],y[2],x[3],y[3],x[4],y[4],x[5],y[5],1]


def update(state,r):
    h,a,hg,ag=int(r.home_id),int(r.away_id),int(r.hg),int(r.ag)
    he,ae=state['elo'][h],state['elo'][a]
    exp=1/(1+10**((ae-he-65)/400)); actual=1 if hg>ag else (.5 if hg==ag else 0)
    d=24*(actual-exp);state['elo'][h]+=d;state['elo'][a]-=d
    hp,ap=(3,0) if hg>ag else ((0,3) if hg<ag else (1,1))
    for tid,pts,gf,ga in [(h,hp,hg,ag),(a,ap,ag,hg)]:
        state['form'][tid].append(pts);state['gf'][tid].append(gf);state['ga'][tid].append(ga)
        state['played'][tid]+=1;state['last'][tid]=r.date


def train(df):
    state=initial();X=[];y=[]
    for _,group in df.groupby(df.date.dt.floor('D'),sort=True):
        for r in group.itertuples():
            X.append(features(state,int(r.home_id),int(r.away_id),r.date))
            y.append('H' if r.hg>r.ag else ('A' if r.hg<r.ag else 'D'))
        for r in group.itertuples():update(state,r)
    X=pd.DataFrame(X,columns=FEATURES);y=np.array(y)
    if len(X)<150 or len(set(y))<3:raise ValueError('At least 150 completed matches and all three outcomes are required.')
    cut=int(len(X)*.8)
    models={'Logistic regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500)),
            'Random forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=8,random_state=42,n_jobs=-1)}
    results=[]
    for name,model in models.items():
        model.fit(X.iloc[:cut],y[:cut]);p=model.predict_proba(X.iloc[cut:])
        results.append((log_loss(y[cut:],p,labels=model.classes_),accuracy_score(y[cut:],model.predict(X.iloc[cut:])),name))
    loss,accuracy,name=min(results)
    model=models[name].fit(X,y)
    names={}
    for r in df.itertuples():names[int(r.home_id)]=r.home;names[int(r.away_id)]=r.away
    arsenal=[i for i,n in names.items() if 'arsenal' in n.lower()]
    return model,state,names,(name,loss,accuracy),arsenal[0] if arsenal else None

st.title('⚽ Arsenal | Match Intelligence')
st.caption('Football outcome probabilities · Historical form · Live-data refresh on demand')
with st.sidebar:
    st.header('Data settings')
    key=st.text_input('football-data.org API key',value=os.environ.get('FOOTBALL_DATA_API_KEY',''),type='password',help='Stored in memory for this session only; do not commit keys to source control.')
    codes=st.multiselect('Competitions to download',list(COMP),default=['PL','CL'],format_func=lambda x:COMP[x])
    seasons=st.slider('Season start years',2015,CURRENT,(max(2023,CURRENT-3),CURRENT))
    refresh=st.checkbox('Refresh current season',value=True)
    st.caption('Each selected competition and season is a separate API request. Free-tier history may stop at 2023; 403 restrictions cannot be bypassed.')
    load=st.button('↻ Download and train',type='primary',use_container_width=True)

if load:
    if not key:st.error('Enter your API key.');st.stop()
    if not codes:st.error('Select at least one competition.');st.stop()
    raw=[];logs=[];restricted=set()
    total=len(codes)*(seasons[1]-seasons[0]+1)
    progress=st.progress(0);count=0;last_request=0.
    for code in codes:
        for season in range(seasons[1],seasons[0]-1,-1):
            count+=1;progress.progress(count/total)
            if code in restricted:continue
            path=CACHE/f'{code}_{season}.json';need_network=not path.exists() or (refresh and season==CURRENT)
            if need_network:
                wait=max(0,6.5-(time.monotonic()-last_request))
                if wait:time.sleep(wait)
            try:
                matches,source=fetch(code,season,key,refresh and season==CURRENT)
                if need_network:last_request=time.monotonic()
                raw.extend(matches);logs.append(f'{code} {season}: {len(matches)} matches ({source})')
            except Exception as e:
                if need_network:last_request=time.monotonic()
                logs.append(f'{code} {season}: {e}')
                if 'HTTP 403' in str(e):restricted.add(code)
                if 'HTTP 429' in str(e):
                    logs.append('Rate limit hit. Wait a minute, then retry; cached seasons will be reused.')
                    restricted.add(code)
                if 'HTTP 401' in str(e):
                    st.error('API key rejected. Rotate your exposed key and enter the new one.');st.stop()
    progress.empty()
    df=normalise(raw)
    st.session_state['logs']=logs
    if df.empty:st.error('No completed match data retrieved. Inspect the request log below.');st.stop()
    try:
        model,state,names,metrics,arsenal=train(df)
    except Exception as e:st.error(str(e));st.stop()
    st.session_state['bundle']=(model,state,names,metrics,arsenal,df)
    st.success(f'Trained on {len(df):,} finished matches; {len(names)} teams.')

if 'logs' in st.session_state:
    with st.expander('Download log / API access diagnostics'):
        st.code('\n'.join(st.session_state['logs']))

if 'bundle' not in st.session_state:
    st.info('Select competitions, enter your API key and click **Download and train** to get started. You can train on one competition or combine several for richer team form.')
    st.markdown('**Why does history stop at 2023?** Older seasons may return HTTP 403 under your subscription. The app uses cached successful seasons and stops older requests for a competition once it encounters a restriction. It does not invent missing matches.')
    st.stop()
model,state,names,metrics,arsenal,df=st.session_state['bundle']
name,loss,acc=metrics
c1,c2,c3,c4=st.columns(4)
c1.metric('Matches trained',f'{len(df):,}')
c2.metric('Teams',len(names))
c3.metric('Holdout accuracy',f'{acc:.1%}')
c4.metric('Holdout log loss',f'{loss:.3f}')
st.caption(f'Model: {name} · Latest completed match: {df.date.max():%d %b %Y} · Evaluation uses a chronological 80/20 split')
if arsenal is None:
    st.warning('Arsenal is not in the selected dataset. Select Premier League and/or Champions League to enable Arsenal predictions.')
    st.stop()

st.subheader('Match predictor')
opponents=sorted([(n,i) for i,n in names.items() if i!=arsenal])
a,b=st.columns([2,1])
with a:opp_name=st.selectbox('Opponent', [n for n,_ in opponents])
with b:venue=st.radio('Arsenal venue',['Home','Away','Neutral'],horizontal=True)
opp=dict(opponents)[opp_name]
now=pd.Timestamp.now(tz='UTC')
def prob(h,a):
    x=pd.DataFrame([features(state,h,a,now)],columns=FEATURES)
    return dict(zip(model.classes_,model.predict_proba(x)[0]))
if venue=='Home':
    p=prob(arsenal,opp);values=[p.get('H',0),p.get('D',0),p.get('A',0)]
elif venue=='Away':
    p=prob(opp,arsenal);values=[p.get('A',0),p.get('D',0),p.get('H',0)]
else:
    p1=prob(arsenal,opp);p2=prob(opp,arsenal)
    values=[(p1.get('H',0)+p2.get('A',0))/2,(p1.get('D',0)+p2.get('D',0))/2,(p1.get('A',0)+p2.get('H',0))/2]
labels=['Arsenal win','Draw',f'{opp_name} win']
st.markdown(f'### Arsenal  vs  {opp_name}')
for label,value in zip(labels,values):
    left,right=st.columns([4,1]);left.write(f'**{label}**');right.write(f'**{value:.1%}**');st.progress(float(value))
if state['played'][opp]<8:st.warning('This opponent has fewer than eight completed matches in the downloaded dataset; treat probabilities cautiously.')
st.caption('Probabilities refer to the match result, not knockout qualification. Neutral venue is approximated by averaging both home/away orientations. Predictions are not guaranteed to be calibrated.')

st.subheader('Recent results')
arsenal_df=df[(df.home_id==arsenal)|(df.away_id==arsenal)].tail(10).sort_values('date',ascending=False)
if len(arsenal_df):st.dataframe(arsenal_df[['date','home','hg','ag','away']],hide_index=True,use_container_width=True)
with st.expander('Model and data limitations'):
    st.write('Features use only results available before each training match. Data refresh is manual when you click Download and train. Historical coverage depends on API permissions and cached data. The current model does not include xG, injuries, line-ups or bookmaker odds. For knockout matches, extra-time and shootout outcomes are excluded from training. This is a research prototype, not a betting tool.')
