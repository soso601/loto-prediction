"""
🎰 Loto Agent IA - v4 : Filtres connectés + Observations complètes + Page Tirages
Lance avec : streamlit run app.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import re, math
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from itertools import combinations
from config import *
from utils import *
from loto_functions import load_draws_from_csv

st.set_page_config(page_title="🎰 Loto Agent IA", page_icon="🎰", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.main-title{font-size:2.2em;font-weight:700;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:0}
.subtitle{text-align:center;color:#888;font-size:1em;margin-bottom:1em}
.num-ball{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#667eea,#764ba2);color:white;font-weight:700;font-size:1.1em;margin:2px}
.chance-ball{display:inline-flex;align-items:center;justify-content:center;width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#f59e0b,#ef4444);color:white;font-weight:700;font-size:1.1em;margin:2px}
.stat-card{background:rgba(102,126,234,0.08);border-radius:12px;padding:16px;border:1px solid rgba(102,126,234,0.15);text-align:center}
.stat-number{font-size:2em;font-weight:700;color:#667eea}
.stat-label{color:#888;font-size:0.85em}
.score-badge{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:4px 14px;border-radius:20px;font-weight:700;font-size:0.9em}
.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75em;font-weight:600;margin:2px}
.tag-lstm{background:rgba(99,102,241,0.2);color:#818cf8}
.tag-retard{background:rgba(245,158,11,0.2);color:#fbbf24}
.tag-equilibre{background:rgba(16,185,129,0.2);color:#34d399}
.tag-reparti{background:rgba(59,130,246,0.2);color:#60a5fa}
.combo-counter{font-size:1.8em;font-weight:800;background:linear-gradient(135deg,#667eea,#764ba2);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center}
.filter-badge{display:inline-block;padding:3px 10px;border-radius:16px;font-size:0.75em;font-weight:600;margin:2px;background:rgba(102,126,234,0.15);color:#818cf8;border:1px solid rgba(102,126,234,0.3)}
</style>""", unsafe_allow_html=True)

# ═══ CHARGEMENT ═══
@st.cache_resource
def load_model_cached():
    from keras.models import load_model as kl
    if not os.path.exists(MODEL_PATH): return None,None
    m=kl(MODEL_PATH)
    with open(SCALER_PATH,'rb') as f: s=pickle.load(f)
    return m,s

@st.cache_data
def load_and_prepare_data():
    r=load_draws_from_csv()
    if r is None: return None,None,None
    d=r.iloc[::-1].reset_index(drop=True)
    dd=d[ALL_DRAW_COLS].copy()
    d_full = d[['day','month_year']+ALL_DRAW_COLS].copy() if 'day' in d.columns else dd.copy()
    return dd, build_all_features(dd), d_full

@st.cache_data
def compute_stats(_d):
    d=_d; t=len(d)
    an=[]
    for c in NUM_COLS: an.extend(d[c].values.tolist())
    fn=Counter(an); fc=Counter(d[CHANCE_COL].values.tolist())
    ret={}; ls={}
    for i,row in d.iterrows():
        for c in NUM_COLS: ls[int(row[c])]=i
    for n in range(1,50): ret[n]=t-1-ls.get(n,-1)
    rc={}; lsc={}
    for i,row in d.iterrows(): lsc[int(row[CHANCE_COL])]=i
    for n in range(1,11): rc[n]=t-1-lsc.get(n,-1)
    sm=d[NUM_COLS].sum(axis=1).values; smoy=float(np.mean(sm)); sstd=float(np.std(sm))
    pc=Counter()
    for _,row in d.iterrows():
        nums=sorted([int(row[c]) for c in NUM_COLS])
        for p in combinations(nums,2): pc[p]+=1
    tp=pc.most_common(20)
    rdc=Counter()
    for _,row in d.iterrows():
        s=sum(int(row[c]) for c in NUM_COLS); rdc[reduction_numerologique(s)]+=1
    ecarts_moy=[]
    for i in range(1,t):
        ec=sum(abs(int(d.iloc[i][c])-int(d.iloc[i-1][c])) for c in NUM_COLS)/5
        ecarts_moy.append(ec)
    consec_list=[]
    for _,row in d.iterrows():
        nums=sorted([int(row[c]) for c in NUM_COLS])
        consec_list.append(sum(1 for j in range(4) if nums[j+1]-nums[j]==1))
    amp_list=[]
    for _,row in d.iterrows():
        nums=[int(row[c]) for c in NUM_COLS]
        amp_list.append(max(nums)-min(nums))
    decade_counts={f"{i*10+1}-{min(i*10+10,49)}":0 for i in range(5)}
    for _,row in d.iterrows():
        for c in NUM_COLS:
            n=int(row[c]); decade_counts[f"{(n-1)//10*10+1}-{min((n-1)//10*10+10,49)}"]+=1
    pi_dist=Counter()
    for _,row in d.iterrows():
        np_=sum(1 for c in NUM_COLS if int(row[c])%2==0); pi_dist[np_]+=1

    # ═══ NOUVELLES ANALYSES ═══
    freq_par_jour = {}
    for jour in d['day'].unique() if 'day' in d.columns else []:
        dj = d[d['day']==jour]
        if len(dj) < 10: continue
        fj = Counter()
        for c in NUM_COLS: fj.update(dj[c].values.tolist())
        freq_par_jour[jour] = {'freq': fj, 'total': len(dj)}

    ratios_jour = {}
    for jour, data in freq_par_jour.items():
        autres = Counter()
        for j2, d2 in freq_par_jour.items():
            if j2 != jour:
                for n in range(1, 50): autres[n] += d2['freq'].get(n, 0)
        t_autres = sum(v['total'] for j2, v in freq_par_jour.items() if j2 != jour)
        ratios = {}
        for n in range(1, 50):
            pj = data['freq'].get(n, 0) / data['total'] * 100
            pa = autres.get(n, 0) / t_autres * 100 if t_autres > 0 else 1
            ratios[n] = round(pj / pa, 2) if pa > 0 else 1
        ratios_jour[jour] = ratios

    mois_map = {'janvier':1,'février':2,'mars':3,'avril':4,'mai':5,'juin':6,
                'juillet':7,'août':8,'septembre':9,'octobre':10,'novembre':11,'décembre':12}
    def get_month(ds):
        parts = ds.strip().split()
        if len(parts) >= 2:
            return mois_map.get(parts[1].lower(), 0)
        return 0
    freq_par_mois = {}
    if 'month_year' in d.columns:
        d_copy = d.copy()
        d_copy['_mois'] = d_copy['month_year'].apply(get_month)
        mois_noms = {1:'Janvier',2:'Février',3:'Mars',4:'Avril',5:'Mai',6:'Juin',
                     7:'Juillet',8:'Août',9:'Septembre',10:'Octobre',11:'Novembre',12:'Décembre'}
        for m in range(1, 13):
            dm = d_copy[d_copy['_mois']==m]
            if len(dm) < 10: continue
            fm = Counter()
            for c in NUM_COLS: fm.update(dm[c].values.tolist())
            freq_par_mois[mois_noms[m]] = {'freq': fm, 'total': len(dm)}

    ratios_mois = {}
    for mois, data in freq_par_mois.items():
        ratios = {}
        for n in range(1, 50):
            pm = data['freq'].get(n, 0) / data['total'] * 100
            pg = fn.get(n, 0) / t * 100
            ratios[n] = round(pm / pg, 2) if pg > 0 else 1
        ratios_mois[mois] = ratios

    pos_freq = {}
    for c in NUM_COLS:
        pos_freq[c] = Counter(d[c].values.tolist())

    trio_counter = Counter()
    for _, row in d.iterrows():
        nums = sorted([int(row[c]) for c in NUM_COLS])
        for trio in combinations(nums, 3): trio_counter[trio] += 1
    top_trios = trio_counter.most_common(20)

    d_recent = d.head(50) if len(d) > 50 else d
    freq_recent = Counter()
    for c in NUM_COLS: freq_recent.update(d_recent[c].values.tolist())
    tendances = {}
    for n in range(1, 50):
        pr = freq_recent.get(n, 0) / len(d_recent) * 100
        pg = fn.get(n, 0) / t * 100
        tendances[n] = {'recent': pr, 'global': pg, 'ratio': round(pr/pg, 2) if pg > 0 else 1}

    term_count = Counter()
    for n in an: term_count[n % 10] += 1

    paire_retards = {}
    for (a, b), cnt in tp:
        last = -1
        for idx, (_, row) in enumerate(d.iterrows()):
            nums = set(int(row[c]) for c in NUM_COLS)
            if a in nums and b in nums: last = idx; break
        paire_retards[(a, b)] = {'count': cnt, 'retard': last if last >= 0 else 999}

    repeat_dist = Counter()
    repeat_per_num = Counter()
    for i in range(len(d)-1):
        curr = set(d.iloc[i][NUM_COLS].values)
        prev = set(d.iloc[i+1][NUM_COLS].values)
        rep = len(curr & prev)
        repeat_dist[rep] += 1
        for n in curr & prev: repeat_per_num[n] += 1

    return {'total':t,'freq_nums':fn,'freq_chance':fc,'retards':ret,'retards_chance':rc,
            'sommes':sm,'somme_moy':smoy,'somme_std':sstd,'top_paires':tp,'red_counter':rdc,
            'ecarts_moy':ecarts_moy,'consec_list':consec_list,'amp_list':amp_list,
            'decade_counts':decade_counts,'pi_dist':pi_dist,
            'freq_par_jour':freq_par_jour,'ratios_jour':ratios_jour,
            'freq_par_mois':freq_par_mois,'ratios_mois':ratios_mois,
            'pos_freq':pos_freq,'top_trios':top_trios,
            'tendances':tendances,'term_count':term_count,
            'paire_retards':paire_retards,'repeat_dist':repeat_dist,
            'repeat_per_num':repeat_per_num, 'freq_recent':freq_recent}

# ═══ MOTEUR ═══
class ComboEngine:
    def __init__(s,stats): s.stats=stats; s.reset()
    def reset(s):
        s.allowed_nums=set(range(1,50)); s.allowed_chance=set(range(1,11))
        s.min_sum=15; s.max_sum=245; s.allowed_pairs=None; s.max_consecutive=5
        s.min_amplitude=4; s.max_amplitude=48; s.allowed_numer=set(range(1,10))
        s.filters_log=[]; s.excluded_combos=[]; s.exact_sum=None
    def count_combos(s):
        n=len(s.allowed_nums);ch=len(s.allowed_chance);base=math.comb(n,5)*ch;f=1.0
        if s.exact_sum: f*=0.005
        else:
            fr=230;cr=s.max_sum-s.min_sum
            if cr<fr: f*=max(0.01,cr/fr)
        if s.allowed_pairs:
            tp=0;ne=len([x for x in s.allowed_nums if x%2==0]);no=len([x for x in s.allowed_nums if x%2!=0])
            for p in s.allowed_pairs:
                if p<=ne and(5-p)<=no: tp+=math.comb(ne,p)*math.comb(no,5-p)
            if math.comb(n,5)>0: f*=tp/math.comb(n,5)
        if s.max_consecutive<4: f*={0:0.35,1:0.75,2:0.92,3:0.98}.get(s.max_consecutive,1.0)
        ca=s.max_amplitude-s.min_amplitude
        if ca<44: f*=max(0.1,ca/44)
        if len(s.allowed_numer)<9: f*=len(s.allowed_numer)/9
        if s.excluded_combos: f*=max(0.5,1-len(s.excluded_combos)*0.02)
        return max(1,int(base*f))
    def exclude_numbers(s,nums):
        rm=[n for n in nums if n in s.allowed_nums]
        for n in rm: s.allowed_nums.discard(n)
        if rm: s.filters_log.append(f"❌ Exclus: {sorted(rm)}")
        return rm
    def exclude_combo(s,nums):
        c=tuple(sorted(nums))
        if c not in s.excluded_combos: s.excluded_combos.append(c); s.filters_log.append(f"🚫 Combo: {list(c)}")
        return c
    def set_exact_sum(s,v): s.exact_sum=v; s.min_sum=v; s.max_sum=v; s.filters_log.append(f"Σ={v}")
    def keep_only_numbers(s,nums):
        k=set(nums)&s.allowed_nums; s.allowed_nums=k; s.filters_log.append(f"✅ Gardés: {sorted(k)}"); return sorted(k)
    def set_sum_range(s,mn,mx): s.exact_sum=None; s.min_sum=max(15,mn); s.max_sum=min(245,mx); s.filters_log.append(f"Σ {s.min_sum}-{s.max_sum}")
    def set_pairs(s,ap): s.allowed_pairs=ap; s.filters_log.append(f"⚖ {[f'{p}P/{5-p}I' for p in ap]}")
    def set_max_consecutive(s,mc): s.max_consecutive=mc; s.filters_log.append(f"🔗 Max {mc}")
    def set_amplitude(s,mn,mx): s.min_amplitude=mn; s.max_amplitude=mx; s.filters_log.append(f"↔ {mn}-{mx}")
    def set_chance(s,al): s.allowed_chance=set(al); s.filters_log.append(f"⭐ {sorted(al)}")
    def set_numerology(s,al): s.allowed_numer=set(al); s.filters_log.append(f"🔮 {sorted(al)}")
    def exclude_cold(s,n=10): return s.exclude_numbers([num for num,_ in list(reversed(s.stats['freq_nums'].most_common()))[:n]])
    def keep_hot(s,n=25): return s.keep_only_numbers([num for num,_ in s.stats['freq_nums'].most_common(n)])
    def exclude_recent(s,n=10): return s.exclude_numbers([num for num,_ in sorted(s.stats['retards'].items(),key=lambda x:x[1])[:n]])
    def apply_optimal_sum(s): s.exact_sum=None; s.set_sum_range(int(s.stats['somme_moy']-s.stats['somme_std']),int(s.stats['somme_moy']+s.stats['somme_std']))
    def _passes(s,nums,ch):
        if not all(n in s.allowed_nums for n in nums): return False
        if ch not in s.allowed_chance: return False
        sm=sum(nums)
        if s.exact_sum:
            if sm!=s.exact_sum: return False
        else:
            if sm<s.min_sum or sm>s.max_sum: return False
        if s.allowed_pairs and sum(1 for x in nums if x%2==0) not in s.allowed_pairs: return False
        if sum(1 for i in range(4) if nums[i+1]-nums[i]==1)>s.max_consecutive: return False
        a=nums[-1]-nums[0]
        if a<s.min_amplitude or a>s.max_amplitude: return False
        if reduction_numerologique(sm) not in s.allowed_numer: return False
        ns=set(nums)
        for c in s.excluded_combos:
            if set(c).issubset(ns): return False
        return True
    def generate_grids(s,n=20,predictions=None,model=None,scaler=None,df_features=None):
        valid=[];al=sorted(s.allowed_nums);cl=sorted(s.allowed_chance)
        if len(al)<5: return []
        if model and scaler and df_features is not None:
            for nl in [0.0,0.3,0.5,0.8,1.0,1.5,2.0]:
                for _ in range(20):
                    try:
                        last=df_features.tail(WINDOW_LENGTH).copy()
                        noisy=scaler.transform(last.values)
                        if nl>0: noisy+=np.random.normal(0,0.5,size=last.shape)*nl
                        sp=model.predict(np.array([noisy]),verbose=0)
                        nb_f=df_features.shape[1];pad=np.zeros((1,nb_f))
                        pad[0,:NB_LABEL_FEATURES]=sp[0]
                        raw=scaler.inverse_transform(pad)[0,:NB_LABEL_FEATURES]
                        p=np.round(raw).astype(int)
                        for i in range(5): p[i]=np.clip(p[i],1,49)
                        p[5]=np.clip(p[5],1,10)
                        fn=[]
                        for rn in p[:5]:
                            if rn in s.allowed_nums and rn not in fn: fn.append(rn)
                            else:
                                best=None;bd=999
                                for cd in al:
                                    if cd not in fn and abs(cd-rn)<bd: best=cd;bd=abs(cd-rn)
                                if best: fn.append(best)
                        if len(fn)!=5: continue
                        nums=sorted(fn);ch=p[5] if p[5] in s.allowed_chance else int(np.random.choice(cl))
                        if s._passes(nums,ch): valid.append((nums,ch,True))
                    except: continue
            if predictions:
                for pred in predictions:
                    base=[n for n in pred[:5] if n in s.allowed_nums]
                    pool=[n for n in al if n not in base]
                    miss=5-len(base)
                    if len(pool)>=miss:
                        for _ in range(30):
                            extra=list(np.random.choice(pool,miss,replace=False))
                            nums=sorted(base+extra);ch=pred[5] if pred[5] in s.allowed_chance else int(np.random.choice(cl))
                            if s._passes(nums,ch): valid.append((nums,ch,True))
        hot=[n for n,_ in s.stats['freq_nums'].most_common(30) if n in s.allowed_nums]
        ovd=[n for n,_ in sorted(s.stats['retards'].items(),key=lambda x:x[1],reverse=True) if n in s.allowed_nums][:20]
        for _ in range(200):
            nh=np.random.randint(2,5);ph=[x for x in hot if x not in ovd[:5]];po=ovd.copy()
            np.random.shuffle(ph);np.random.shuffle(po);nums=sorted(ph[:nh]+po[:5-nh])
            if len(set(nums))==5:
                ch=int(np.random.choice(cl))
                if s._passes(nums,ch): valid.append((nums,ch,False))
        decs=[[n for n in range(d*10+1,min(d*10+11,50)) if n in s.allowed_nums] for d in range(5)]
        dok=[d for d in decs if d]
        if len(dok)>=5:
            for _ in range(200):
                nums=sorted([int(np.random.choice(d)) for d in dok[:5]]);ch=int(np.random.choice(cl))
                if len(set(nums))==5 and s._passes(nums,ch): valid.append((nums,ch,False))
        att=0
        while len(valid)<n*8 and att<80000:
            att+=1;nums=sorted(np.random.choice(al,5,replace=False));ch=int(np.random.choice(cl))
            if s._passes(list(nums),ch): valid.append((list(nums),ch,False))
        scored=[];seen=set()
        for nums,chance,il in valid:
            key=tuple(nums+[chance])
            if key not in seen:
                seen.add(key);sc=50
                if il: sc+=15
                mf=max(s.stats['freq_nums'].values())
                sc+=sum(s.stats['freq_nums'].get(nn,0)/mf for nn in nums)*5
                ar=np.mean(list(s.stats['retards'].values()))
                sc+=sum(min(s.stats['retards'].get(nn,0)/(ar+1),2) for nn in nums)*3
                sc+=len(set((nn-1)//10 for nn in nums))*3
                if predictions:
                    for pred in predictions:
                        sc+=len(set(nums)&set(pred[:5]))*3
                        if pred[5]==chance: sc+=2
                scored.append({'nums':nums,'chance':chance,'score':round(sc,1),'lstm':il})
        scored.sort(key=lambda x:x['score'],reverse=True)
        div=[]
        for g in scored:
            if all(len(set(g['nums'])&set(ss['nums']))<4 for ss in div): div.append(g)
            if len(div)>=n: break
        return div

# ═══ HELPERS ═══
def render_balls(nums,chance):
    h=""
    for n in nums: h+=f'<span class="num-ball">{n}</span>'
    h+=f'<span class="chance-ball">{chance}</span>'; return h
def render_tags(d):
    t=""
    if d.get('lstm',False): t+='<span class="tag tag-lstm">🧠 LSTM</span>'
    if d.get('retard',0)>60: t+='<span class="tag tag-retard">⏰ Retard</span>'
    if d.get('even_odd',0)>80: t+='<span class="tag tag-equilibre">⚖ Équilibré</span>'
    if d.get('decade_spread',0)>80: t+='<span class="tag tag-reparti">📊 Réparti</span>'
    return t
def extract_numbers(text,mx=49): return [int(n) for n in re.findall(r'\d+',text) if 1<=int(n)<=mx]
def extract_single_number(text):
    nums=re.findall(r'\d+',text); return int(nums[0]) if nums else None

# ═══ AJOUT MANUEL DE TIRAGE ═══
def ajouter_tirage_manuel(csv_path, jour, date_str, nums, chance):
    """Ajoute manuellement un tirage en haut du CSV."""
    df = pd.read_csv(csv_path)
    new_row = pd.DataFrame([{
        'day': jour, 'month_year': date_str,
        'num0': nums[0], 'num1': nums[1], 'num2': nums[2],
        'num3': nums[3], 'num4': nums[4], 'chance': chance
    }])
    df = pd.concat([new_row, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return True

# ═══ CHATBOT ═══
def chatbot_respond(msg,engine,stats,preds,model=None,scaler=None,df_features=None):
    m=msg.lower().strip()
    if any(w in m for w in ['reset','recommencer','réinitialiser','annuler']):
        engine.reset(); return "🔄 Reset ! **19 068 840** combinaisons."
    if any(w in m for w in ['enlève','enlever','exclure','exclus','retire','supprime','sans']):
        if any(w in m for w in ['froid','froids','rares']):
            n=extract_single_number(m) or 10;rm=engine.exclude_cold(n);return f"❄️ **{len(rm)}** froids exclus: {sorted(rm)}\n\n🎯 **{engine.count_combos():,}**"
        if any(w in m for w in ['recent','récent','récents']):
            n=extract_single_number(m) or 10;rm=engine.exclude_recent(n);return f"🆕 **{len(rm)}** récents exclus: {sorted(rm)}\n\n🎯 **{engine.count_combos():,}**"
        if any(w in m for w in ['combo','combinaison','ensemble','avec']):
            nums=extract_numbers(m,49)
            if len(nums)>=2: engine.exclude_combo(nums); return f"🚫 Combo **{sorted(nums)}** exclue.\n\n🎯 **{engine.count_combos():,}**"
        nums=extract_numbers(m,49)
        if nums: engine.exclude_numbers(nums); return f"❌ Exclus: **{sorted(nums)}**\n\n**{len(engine.allowed_nums)}** numéros → **{engine.count_combos():,}**"
    if any(w in m for w in ['garde','garder','uniquement','seulement','que les']):
        if any(w in m for w in ['chaud','chauds']): n=extract_single_number(m) or 25;engine.keep_hot(n);return f"🔥 **{n}** chauds → **{engine.count_combos():,}**"
        nums=extract_numbers(m,49)
        if nums: engine.keep_only_numbers(nums);return f"✅ Gardés → **{engine.count_combos():,}**"
    if any(w in m for w in ['somme','sum']):
        if any(w in m for w in ['optimal','idéal']): engine.apply_optimal_sum();return f"Σ Optimale: **{engine.min_sum}-{engine.max_sum}** → **{engine.count_combos():,}**"
        if any(w in m for w in ['exacte','exact','font','fait','égale','=']):
            nums=extract_numbers(m,300)
            if nums:
                t=max(nums)
                if 15<=t<=245: engine.set_exact_sum(t);return f"Σ = **{t}** → **{engine.count_combos():,}**"
        nums=extract_numbers(m,300)
        if len(nums)>=2: engine.set_sum_range(min(nums),max(nums));return f"Σ **{engine.min_sum}-{engine.max_sum}** → **{engine.count_combos():,}**"
        if len(nums)==1 and nums[0]>=15: engine.set_exact_sum(nums[0]);return f"Σ = **{nums[0]}** → **{engine.count_combos():,}**"
        return f"Σ Moyenne: **{stats['somme_moy']:.0f}** ± {stats['somme_std']:.0f}\n\n• **somme 125** · **somme entre 100 160** · **somme optimale**"
    if any(w in m for w in ['génère','genere','toutes les','liste']) and any(w in m for w in ['somme','font','fait']):
        nums=extract_numbers(m,300)
        for n in nums:
            if 15<=n<=245:
                engine.set_exact_sum(n)
                grids=engine.generate_grids(n=20,predictions=preds,model=model,scaler=scaler,df_features=df_features)
                if not grids: return f"⚠️ Aucune grille somme={n}."
                lc=sum(1 for g in grids if g.get('lstm'))
                resp=f"Σ **Somme={n}** — Top {len(grids)} (dont {lc} 🧠) :\n\n"
                for i,g in enumerate(grids,1):
                    lt=" 🧠" if g.get('lstm') else ""
                    resp+=f"**{i}.** {' - '.join(str(nn) for nn in g['nums'])} | ⭐ {g['chance']}{lt}\n\n"
                return resp
    if any(w in m for w in ['pair','impair']):
        nums=extract_numbers(m,5);engine.set_pairs(nums if nums else [2,3]);return f"⚖ → **{engine.count_combos():,}**"
    if any(w in m for w in ['consécutif','consecutif','suite']):
        n=extract_single_number(m);engine.set_max_consecutive(n if n and n<=4 else 1);return f"🔗 Max {engine.max_consecutive} → **{engine.count_combos():,}**"
    if any(w in m for w in ['amplitude','écart']):
        nums=extract_numbers(m,48)
        if len(nums)>=2: engine.set_amplitude(min(nums),max(nums))
        else: engine.set_amplitude(20,42)
        return f"↔ {engine.min_amplitude}-{engine.max_amplitude} → **{engine.count_combos():,}**"
    if any(w in m for w in ['chance','étoile']):
        nums=extract_numbers(m,10)
        if nums: engine.set_chance(nums);return f"⭐ {sorted(engine.allowed_chance)} → **{engine.count_combos():,}**"
    if any(w in m for w in ['numérolog','numerolog']):
        nums=extract_numbers(m,9)
        if nums: engine.set_numerology(nums);return f"🔮 {sorted(engine.allowed_numer)} → **{engine.count_combos():,}**"
        resp="🔮 **Réductions :**\n\n"
        for r in range(1,10): pct=stats['red_counter'].get(r,0)/stats['total']*100; resp+=f"**{r}** → {'█'*int(pct)} {pct:.1f}%\n\n"
        return resp+"Tape **numérologie 3 6 9** pour filtrer."
    if any(w in m for w in ['grille','propose','donne','jouer','prédiction','prediction']):
        n=min(extract_single_number(m) or 10,20)
        grids=engine.generate_grids(n=n,predictions=preds,model=model,scaler=scaler,df_features=df_features)
        if not grids: return "⚠️ Trop restrictif ! **reset** pour recommencer."
        lc=sum(1 for g in grids if g.get('lstm'))
        resp=f"🎯 **Top {len(grids)}** ({engine.count_combos():,} combos) — {lc} du LSTM 🧠 :\n\n"
        for i,g in enumerate(grids,1):
            lt=" 🧠" if g.get('lstm') else ""
            resp+=f"**{i}.** {' - '.join(str(nn) for nn in g['nums'])} | ⭐ {g['chance']} *({g['score']})*{lt}\n\n"
        return resp
    if any(w in m for w in ['chaud','hot','fréquent']):
        resp="🔥 **Top 15 :**\n\n"
        for num,c in stats['freq_nums'].most_common(15): resp+=f"**{num}** → {c}x ({c/stats['total']*100:.1f}%)\n\n"
        return resp
    if any(w in m for w in ['froid','cold','rare']):
        resp="❄️ **Top 15 :**\n\n"
        for num,c in list(reversed(stats['freq_nums'].most_common()))[:15]: resp+=f"**{num}** → {c}x\n\n"
        return resp
    if any(w in m for w in ['retard','overdue','absent']):
        resp="⏰ **Top 15 :**\n\n"
        for num,r in sorted(stats['retards'].items(),key=lambda x:x[1],reverse=True)[:15]: resp+=f"**{num}** → {r} tirages\n\n"
        return resp
    if any(w in m for w in ['état','filtre','status','combien']):
        c=engine.count_combos();red=(1-c/TOTAL_COMBINATIONS)*100
        resp=f"📊 **{len(engine.allowed_nums)}**/49 | **{c:,}** | **-{red:.1f}%**"
        if engine.exact_sum: resp+=f"\nΣ = **{engine.exact_sum}**"
        if engine.excluded_combos: resp+=f"\n🚫 {len(engine.excluded_combos)} combo(s)"
        return resp
    if any(w in m for w in ['recommand','conseil','stratégie','optimal','auto','intelligent']):
        engine.reset();engine.exclude_cold(8);engine.apply_optimal_sum();engine.set_pairs([2,3])
        engine.set_max_consecutive(1);engine.set_amplitude(20,42)
        grids=engine.generate_grids(n=10,predictions=preds,model=model,scaler=scaler,df_features=df_features)
        resp=f"🧠 **Stratégie auto :**\n❄️ 8 froids | Σ Optimale | ⚖ 2-3P | 🔗 Max1 | ↔ 20-42\n\n🎯 **{engine.count_combos():,}**\n\n"
        for i,g in enumerate(grids,1):
            lt=" 🧠" if g.get('lstm') else ""
            resp+=f"**{i}.** {' - '.join(str(nn) for nn in g['nums'])} | ⭐ {g['chance']}{lt}\n\n"
        return resp
    if any(w in m for w in ['aide','help','comment']):
        return """🤖 **Commandes :**\n\n**Exclure:** "Enlève 6 15" · "Exclure combo 6 17" · "Exclure froids"\n\n**Filtrer:** "Somme 125" · "Somme entre 100 160" · "2 ou 3 pairs" · "Max 1 consécutif" · "Amplitude 20 40" · "Chance 1 3 5" · "Numérologie 3 6 9"\n\n**Générer:** "Grilles" · "Génère somme 125" · "Recommande"\n\n**Stats:** "Chauds" · "Froids" · "Retards" · "État"\n\n**Reset:** "Reset" """
    if any(w in m for w in ['salut','bonjour','hello','yo','coucou']):
        return f"👋 **{engine.count_combos():,}** combinaisons. Tape **recommande** ou **aide** !"
    return f"🤔 Tape **aide** pour les commandes.\n\n🎯 **{engine.count_combos():,}**"

# ═══ MAIN ═══
def main():
    st.markdown('<h1 class="main-title">🎰 Loto Agent IA</h1>',unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligence artificielle au service de tes grilles</p>',unsafe_allow_html=True)
    model,scaler=load_model_cached(); df_draws,df_features,df_full=load_and_prepare_data()
    if df_draws is None: st.error("❌ Place tirages_loto.csv dans data/"); return
    stats=compute_stats(df_full)
    preds=[]
    if model and scaler and df_features is not None:
        from models_functions import predict_next_draw
        preds=[predict_next_draw(model,scaler,df_features)]
    if 'messages' not in st.session_state:
        st.session_state.messages=[{'role':'assistant','content':f"👋 **{stats['total']}** tirages analysés. Tape **recommande** ou **aide** !"}]
    if 'engine' not in st.session_state: st.session_state.engine=ComboEngine(stats)
    engine=st.session_state.engine

    # Stats header
    c1,c2,c3,c4=st.columns(4)
    combos=engine.count_combos();red=(1-combos/TOTAL_COMBINATIONS)*100
    with c1: st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["total"]}</div><div class="stat-label">Tirages</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-card"><div class="stat-number">{"✅" if model else "❌"}</div><div class="stat-label">LSTM</div></div>',unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-card"><div class="stat-number">64</div><div class="stat-label">Features</div></div>',unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-card"><div class="stat-number">{combos:,}</div><div class="stat-label">Combos (-{red:.1f}%)</div></div>',unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.title("🎛 Navigation")
    page=st.sidebar.radio("",["🎯 Grilles Optimisées","🧠 Prédictions LSTM","📊 Statistiques","🔬 Observations Complètes","🔮 Numérologie","⚖ Score ta Grille","📈 Tendances","🎫 Tirages"])

    # LAYOUT
    page_col,chat_col=st.columns([3,2])

    with page_col:
        # ═══ GRILLES ═══
        if page=="🎯 Grilles Optimisées":
            st.header("🎯 Meilleures Grilles")
            if st.button("🔄 Régénérer",type="primary",use_container_width=True): st.rerun()
            with st.spinner("⚡ Génération avec filtres actifs..."):
                grids=engine.generate_grids(n=20,predictions=preds,model=model,scaler=scaler,df_features=df_features)
            lc=sum(1 for g in grids if g.get('lstm'))
            st.success(f"✅ Top {len(grids)} — dont {lc} du LSTM 🧠")
            for i,grid in enumerate(grids[:10],1):
                cols=st.columns([1,5,2,3])
                with cols[0]: st.markdown(f"### #{i}")
                with cols[1]: st.markdown(render_balls(grid['nums'],grid['chance']),unsafe_allow_html=True)
                with cols[2]: st.markdown(f'<span class="score-badge">{grid["score"]}/100</span>',unsafe_allow_html=True)
                with cols[3]:
                    tags=""
                    if grid.get('lstm'): tags+='<span class="tag tag-lstm">🧠 LSTM</span>'
                    st.markdown(tags,unsafe_allow_html=True)

        # ═══ LSTM ═══
        elif page=="🧠 Prédictions LSTM":
            st.header("🧠 Prédictions LSTM")
            if not model: st.warning("Pas de modèle."); st.stop()
            with st.spinner("..."): pr=generate_predictions(model,scaler,df_features,8)
            for i,pred in enumerate(pr,1): st.markdown(f"**Grille {i}** : {render_balls(pred[:5],pred[5])}",unsafe_allow_html=True)
            st.markdown("---");st.subheader("📊 Consensus")
            an=[n for p in pr for n in p[:5]]
            st.bar_chart(pd.DataFrame([{'N°':str(n),'×':c} for n,c in Counter(an).most_common(15)]).set_index('N°'))

        # ═══ STATISTIQUES ═══
        elif page=="📊 Statistiques":
            st.header("📊 Statistiques Complètes")
            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11=st.tabs([
                "🔥 Chauds/Froids","⏰ Retards","👯 Paires","📅 Par Jour","📆 Par Mois",
                "📍 Positions","👯‍♂️ Trios","📈 Tendances","🔢 Terminaisons","🔴 Retard Paires","🔄 Répétitions"
            ])

            with t1:
                st.subheader("🔥 Numéros Chauds")
                st.dataframe(pd.DataFrame([{'N°':n,'Sorties':c,'%':f"{c/stats['total']*100:.1f}%"} for n,c in stats['freq_nums'].most_common(15)]),use_container_width=True,hide_index=True)
                st.subheader("❄️ Numéros Froids")
                st.dataframe(pd.DataFrame([{'N°':n,'Sorties':c,'%':f"{c/stats['total']*100:.1f}%"} for n,c in list(reversed(stats['freq_nums'].most_common()))[:15]]),use_container_width=True,hide_index=True)

            with t2:
                st.subheader("⏰ Numéros en Retard")
                st.dataframe(pd.DataFrame([{'N°':n,'Retard (tirages)':r} for n,r in sorted(stats['retards'].items(),key=lambda x:x[1],reverse=True)[:20]]),use_container_width=True,hide_index=True)

            with t3:
                st.subheader("👯 Paires les Plus Fréquentes")
                st.dataframe(pd.DataFrame([{'Paire':f"{a} - {b}",'Sorties':c} for (a,b),c in stats['top_paires']]),use_container_width=True,hide_index=True)

            with t4:
                st.subheader("📅 Fréquences par Jour de la Semaine")
                st.write("Certains numéros sortent plus souvent certains jours :")
                for jour in ['Lundi','Mercredi','Samedi']:
                    if jour not in stats['ratios_jour']: continue
                    ratios = stats['ratios_jour'][jour]
                    jdata = stats['freq_par_jour'][jour]
                    st.markdown(f"### 🗓 {jour} ({jdata['total']} tirages)")
                    sorted_up = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔥 Sortent PLUS ce jour :**")
                        for num, ratio in sorted_up[:5]:
                            pct = jdata['freq'].get(num,0) / jdata['total'] * 100
                            st.write(f"**{num}** → {pct:.1f}% (x{ratio} vs autres jours)")
                    with col2:
                        st.markdown("**❄️ Sortent MOINS ce jour :**")
                        for num, ratio in sorted_up[-5:]:
                            pct = jdata['freq'].get(num,0) / jdata['total'] * 100
                            st.write(f"**{num}** → {pct:.1f}% (x{ratio} vs autres jours)")
                    st.markdown("---")

            with t5:
                st.subheader("📆 Fréquences par Mois")
                st.write("Variations mensuelles les plus marquantes :")
                for mois in ['Janvier','Février','Mars','Avril','Mai','Juin','Juillet','Août','Septembre','Octobre','Novembre','Décembre']:
                    if mois not in stats['ratios_mois']: continue
                    ratios = stats['ratios_mois'][mois]
                    mdata = stats['freq_par_mois'][mois]
                    sorted_r = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
                    top3 = sorted_r[:3]
                    flop3 = sorted_r[-3:]
                    top_str = ' · '.join(f"**{n}**(x{r})" for n, r in top3)
                    flop_str = ' · '.join(f"**{n}**(x{r})" for n, r in flop3)
                    st.markdown(f"**{mois}** ({mdata['total']} tirages) — 🔥 {top_str} | ❄️ {flop_str}")

                st.markdown("---")
                st.markdown("### 💡 Faits marquants")
                st.write("• Le **35** sort **68% plus souvent en Octobre** que sa moyenne")
                st.write("• Le **31** sort **56% plus en Janvier**, mais **42% moins en Mai**")
                st.write("• Le **47** explose en **Juillet** (x1.56) mais chute en **Août** (x0.70)")
                st.write("• Le **9** disparaît en **Septembre** (x0.50)")

            with t6:
                st.subheader("📍 Fréquences par Position")
                st.write("Les numéros sont triés dans chaque tirage. Voici les favoris par position :")
                pos_labels = ['1er (plus petit)', '2ème', '3ème', '4ème', '5ème (plus grand)']
                for col_name, label in zip(NUM_COLS, pos_labels):
                    if col_name in stats['pos_freq']:
                        top5 = stats['pos_freq'][col_name].most_common(5)
                        nums_str = ' · '.join(f"**{n}**({c}x)" for n, c in top5)
                        st.write(f"**{label}** : {nums_str}")

            with t7:
                st.subheader("👯‍♂️ Trios les Plus Fréquents")
                st.write("Groupes de 3 numéros qui sortent souvent ensemble :")
                st.dataframe(pd.DataFrame([{'Trio':f"{a} - {b} - {c}",'Sorties':cnt} for (a,b,c),cnt in stats['top_trios']]),use_container_width=True,hide_index=True)

            with t8:
                st.subheader("📈 Tendances Récentes (50 derniers tirages)")
                st.write("Comparaison avec la fréquence historique :")
                st.markdown("**🚀 En forte hausse :**")
                sorted_t = sorted(stats['tendances'].items(), key=lambda x: x[1]['ratio'], reverse=True)
                for num, data in sorted_t[:10]:
                    st.write(f"**{num}** → {data['recent']:.1f}% récent vs {data['global']:.1f}% global (**x{data['ratio']}**)")
                st.markdown("**📉 En forte baisse :**")
                for num, data in sorted_t[-10:]:
                    st.write(f"**{num}** → {data['recent']:.1f}% récent vs {data['global']:.1f}% global (**x{data['ratio']}**)")

            with t9:
                st.subheader("🔢 Terminaisons (chiffre des unités)")
                st.write("Fréquence des numéros selon leur dernier chiffre :")
                term_data = []
                for t_val in range(10):
                    count = stats['term_count'].get(t_val, 0)
                    nums_in = [n for n in range(1, 50) if n % 10 == t_val]
                    pct = count / (stats['total'] * 5) * 100
                    term_data.append({'Termine par': str(t_val), 'Numéros': str(nums_in), 'Sorties': count, '%': f"{pct:.1f}%"})
                st.dataframe(pd.DataFrame(term_data), use_container_width=True, hide_index=True)

            with t10:
                st.subheader("🔴 Retard des Paires Fréquentes")
                st.write("Les paires les plus fréquentes et depuis combien de temps elles ne sont pas sorties :")
                pr_data = []
                for (a, b), info in sorted(stats['paire_retards'].items(), key=lambda x: x[1]['retard'], reverse=True):
                    status = "🔴" if info['retard'] > 50 else "🟡" if info['retard'] > 20 else "🟢"
                    pr_data.append({'':status, 'Paire':f"{a} - {b}", 'Sorties totales':info['count'], 'Retard':f"{info['retard']} tirages"})
                st.dataframe(pd.DataFrame(pr_data), use_container_width=True, hide_index=True)

            with t11:
                st.subheader("🔄 Répétitions entre tirages")
                st.write("Combien de numéros se répètent d'un tirage au suivant :")
                for k in range(5):
                    count = stats['repeat_dist'].get(k, 0)
                    pct = count / max(1, sum(stats['repeat_dist'].values())) * 100
                    st.write(f"**{k} numéro(s) répété(s)** → {count}x ({pct:.1f}%)")
                st.markdown("---")
                st.markdown("**Numéros qui se répètent le plus au tirage suivant :**")
                for num, count in stats['repeat_per_num'].most_common(10):
                    st.write(f"**{num}** → {count}x")

        # ═══ OBSERVATIONS COMPLÈTES ═══
        elif page=="🔬 Observations Complètes":
            st.header("🔬 Observations Complètes")
            st.write(f"Analyse approfondie de **{stats['total']}** tirages historiques")

            t1,t2,t3,t4,t5,t6,t7,t8=st.tabs(["Σ Sommes","⚖ Pair/Impair","📊 Dizaines","🔗 Consécutifs","↔ Amplitude","📐 Écarts","🔮 Numérologie","🕐 Derniers tirages"])

            with t1:
                st.subheader("Σ Distribution des sommes")
                st.write(f"**Moyenne** : {stats['somme_moy']:.1f} | **Écart-type** : {stats['somme_std']:.1f}")
                st.write(f"**Plage optimale** : {stats['somme_moy']-stats['somme_std']:.0f} à {stats['somme_moy']+stats['somme_std']:.0f}")
                bins=[(15,60),(60,80),(80,100),(100,115),(115,130),(130,145),(145,160),(160,180),(180,245)]
                st.bar_chart(pd.DataFrame([{'Plage':f"{l}-{h}",'Tirages':sum(1 for s in stats['sommes'] if l<=s<h)} for l,h in bins]).set_index('Plage'))
                st.subheader("Évolution des sommes (50 derniers)")
                st.line_chart(pd.DataFrame({'Somme':stats['sommes'][-50:],'Moyenne':[stats['somme_moy']]*50}))

            with t2:
                st.subheader("⚖ Répartition Pair/Impair")
                st.write("Combien de numéros pairs dans chaque tirage ?")
                pi_data=pd.DataFrame([{'Répartition':f"{p}P/{5-p}I",'Tirages':stats['pi_dist'].get(p,0),'%':f"{stats['pi_dist'].get(p,0)/stats['total']*100:.1f}%"} for p in range(6)])
                st.dataframe(pi_data,use_container_width=True,hide_index=True)
                st.bar_chart(pd.DataFrame([{'Config':f"{p}P/{5-p}I",'Tirages':stats['pi_dist'].get(p,0)} for p in range(6)]).set_index('Config'))
                st.info("💡 Les configurations **2P/3I** et **3P/2I** sont les plus fréquentes (~62% des tirages)")

            with t3:
                st.subheader("📊 Répartition par dizaine")
                st.write("Combien de numéros tombent dans chaque tranche de 10 ?")
                st.bar_chart(pd.DataFrame([{'Dizaine':k,'Total':v} for k,v in stats['decade_counts'].items()]).set_index('Dizaine'))
                st.write("Un tirage 'idéal' a 1 numéro par dizaine (couverture maximale)")

            with t4:
                st.subheader("🔗 Numéros consécutifs")
                st.write("Combien de paires consécutives (ex: 7-8) par tirage ?")
                cc=Counter(stats['consec_list'])
                st.bar_chart(pd.DataFrame([{'Consécutifs':str(c),'Tirages':cc.get(c,0)} for c in range(5)]).set_index('Consécutifs'))
                pct_0=cc.get(0,0)/stats['total']*100
                pct_1=cc.get(1,0)/stats['total']*100
                st.info(f"💡 **{pct_0:.0f}%** des tirages n'ont aucun consécutif, **{pct_1:.0f}%** en ont exactement 1")

            with t5:
                st.subheader("↔ Amplitude (max - min)")
                st.write("Écart entre le plus grand et le plus petit numéro")
                amp_arr=np.array(stats['amp_list'])
                st.write(f"**Moyenne** : {np.mean(amp_arr):.1f} | **Min** : {np.min(amp_arr)} | **Max** : {np.max(amp_arr)}")
                amp_bins=[(4,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,48)]
                st.bar_chart(pd.DataFrame([{'Amplitude':f"{l}-{h}",'Tirages':sum(1 for a in stats['amp_list'] if l<=a<=h)} for l,h in amp_bins]).set_index('Amplitude'))

            with t6:
                st.subheader("📐 Écarts entre tirages successifs")
                st.write("Écart moyen entre les numéros d'un tirage et ceux du tirage précédent")
                if stats['ecarts_moy']:
                    ec_arr=np.array(stats['ecarts_moy'])
                    st.write(f"**Écart moyen** : {np.mean(ec_arr):.1f} | **Max** : {np.max(ec_arr):.1f}")
                    st.line_chart(pd.DataFrame({'Écart moyen':stats['ecarts_moy'][-100:]}))

            with t7:
                st.subheader("🔮 Numérologie")
                st.write("Réduction de chaque nombre à un seul chiffre (24 → 2+4 = 6)")
                st.markdown("**Fréquence des réductions de la somme :**")
                st.bar_chart(pd.DataFrame([{'Réduction':str(r),'Fréquence':stats['red_counter'].get(r,0)} for r in range(1,10)]).set_index('Réduction'))
                st.markdown("**Réduction la plus fréquente par position :**")
                for c in NUM_COLS:
                    rc=Counter(df_draws[c].apply(reduction_numerologique).values.tolist())
                    top3=rc.most_common(3)
                    st.write(f"**{c}** : {' · '.join(f'{r}→{cnt}x' for r,cnt in top3)}")

            with t8:
                st.subheader("🕐 20 derniers tirages")
                last20=df_draws.tail(20).iloc[::-1]
                for _,row in last20.iterrows():
                    nums=[int(row[c]) for c in NUM_COLS]; ch=int(row[CHANCE_COL])
                    sm=sum(nums); rd=reduction_numerologique(sm)
                    np_=sum(1 for n in nums if n%2==0)
                    st.markdown(f"{render_balls(nums,ch)} Σ**{sm}** →{rd} | {np_}P/{5-np_}I",unsafe_allow_html=True)

        # ═══ NUMÉROLOGIE ═══
        elif page=="🔮 Numérologie":
            st.header("🔮 Numérologie")
            st.subheader("Derniers tirages")
            for _,row in df_draws.tail(10).iterrows():
                nums=[int(row[c]) for c in NUM_COLS]; reds=[reduction_numerologique(n) for n in nums]
                s=sum(nums);rs=reduction_numerologique(s)
                c1,c2,c3=st.columns([4,3,2])
                with c1: st.write(f"**{' - '.join(str(n) for n in nums)}**")
                with c2: st.write(f"→ {' - '.join(str(r) for r in reds)}")
                with c3: st.write(f"Σ{s} → **{rs}**")
            st.subheader("Distribution")
            st.bar_chart(pd.DataFrame([{'Réd':str(r),'Fréq':stats['red_counter'].get(r,0)} for r in range(1,10)]).set_index('Réd'))

        # ═══ SCORE ═══
        elif page=="⚖ Score ta Grille":
            st.header("⚖ Score ta Grille")
            cols=st.columns(6); un=[]
            for i in range(5):
                with cols[i]: un.append(st.number_input(f"N°{i+1}",1,49,(i+1)*9,key=f"n{i}"))
            with cols[5]: uc=st.number_input("Chance",1,10,5,key="ch")
            if len(set(un))<5: st.warning("⚠️ 5 différents !")
            else:
                sc,det=score_combination(un,uc,stats,preds)
                st.markdown(f"### {render_balls(sorted(un),uc)}",unsafe_allow_html=True)
                st.markdown(f"## Score : {sc}/100"); st.progress(sc/100)
                dc=st.columns(4)
                labels={'lstm_match':'🧠 LSTM','frequency':'📈 Fréq','retard':'⏰ Retard','sum_range':'Σ Somme','even_odd':'⚖ P/I','decade_spread':'📊 Diz','consecutive':'🔗 Cons','amplitude':'↔ Amp'}
                for j,(k,l) in enumerate(labels.items()):
                    with dc[j%4]: st.metric(l,f"{det.get(k,0):.0f}/100")
                st.info(f"🔮 Somme={sum(un)} → **{reduction_numerologique(sum(un))}**")

        # ═══ TENDANCES ═══
        elif page=="📈 Tendances":
            st.header("📈 Tendances")
            st.line_chart(pd.DataFrame({'Somme':df_draws[NUM_COLS].sum(axis=1).values[-100:],'Moyenne':[stats['somme_moy']]*100}))
            st.write(f"**Moyenne**: {stats['somme_moy']:.1f} | **Plage**: {stats['somme_moy']-stats['somme_std']:.0f}-{stats['somme_moy']+stats['somme_std']:.0f}")
            st.subheader("⭐ Numéro Chance")
            st.bar_chart(pd.DataFrame([{'N°':str(n),'Fréq':stats['freq_chance'].get(n,0)} for n in range(1,11)]).set_index('N°'))

        # ═══ TIRAGES (NOUVELLE PAGE) ═══
        elif page=="🎫 Tirages":
            st.header("🎫 Historique des Tirages")

            # --- Dernier tirage info ---
            # df_full est en ordre ancien→récent, donc le dernier = .iloc[-1]
            dernier_jour = df_full.iloc[-1]['day'] if 'day' in df_full.columns else ''
            dernier_date = df_full.iloc[-1]['month_year'] if 'month_year' in df_full.columns else ''
            st.caption(f"📅 Dernier tirage : **{dernier_jour} {dernier_date}** — {stats['total']} tirages au total")

            # --- Ajout manuel (expander) ---
            with st.expander("➕ Ajouter un tirage manuellement"):
                st.caption("Ajoute le dernier tirage FDJ à la main :")
                col_j, col_d = st.columns(2)
                with col_j:
                    jour_sel = st.selectbox("Jour", ["Lundi", "Mercredi", "Samedi"], key="tj")
                with col_d:
                    date_man = st.text_input("Date (ex: 12 avril 2026)", key="td")

                cols_n = st.columns(6)
                nums_man = []
                for i in range(5):
                    with cols_n[i]:
                        nums_man.append(st.number_input(f"N°{i+1}", 1, 49, (i+1)*8, key=f"tn{i}"))
                with cols_n[5]:
                    chance_man = st.number_input("Chance", 1, 10, 1, key="tc")

                if st.button("✅ Ajouter ce tirage", use_container_width=True):
                    if not date_man:
                        st.error("Entre la date !")
                    elif len(set(nums_man)) < 5:
                        st.error("Les 5 numéros doivent être différents !")
                    else:
                        ajouter_tirage_manuel(DRAWS_CSV_PATH, jour_sel, date_man, sorted(nums_man), chance_man)
                        st.success(f"✅ Tirage du {jour_sel} {date_man} ajouté !")
                        st.cache_data.clear()
                        st.rerun()

            st.markdown("---")

            # --- Filtres ---
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                search_date = st.text_input("🔍 Recherche par date", placeholder="ex: avril 2026", key="sd")
            with col_f2:
                search_num = st.text_input("🔍 Contient le numéro", placeholder="ex: 7", key="sn")
            with col_f3:
                nb_display = st.selectbox("Afficher", [20, 50, 100, 200, "Tous"], index=0, key="nd")

            # --- Préparer les données (récent en premier) ---
            df_show = df_full.copy()
            df_show = df_show.iloc[::-1].reset_index(drop=True)

            # Appliquer les filtres
            if search_date:
                mask = df_show['month_year'].str.contains(search_date, case=False, na=False)
                mask |= df_show['day'].str.contains(search_date, case=False, na=False)
                df_show = df_show[mask]

            if search_num:
                try:
                    n_search = int(search_num)
                    mask = pd.Series(False, index=df_show.index)
                    for c in NUM_COLS:
                        mask |= (df_show[c] == n_search)
                    mask |= (df_show[CHANCE_COL] == n_search)
                    df_show = df_show[mask]
                except ValueError:
                    pass

            # --- Stats rapides du filtre ---
            st.caption(f"**{len(df_show)}** tirages trouvés")

            # --- Limiter l'affichage ---
            if nb_display != "Tous":
                df_page = df_show.head(int(nb_display))
            else:
                df_page = df_show

            # --- Affichage des tirages avec boules ---
            for idx, (_, row) in enumerate(df_page.iterrows()):
                nums = [int(row[c]) for c in NUM_COLS]
                ch = int(row[CHANCE_COL])
                jour = row['day'] if 'day' in row.index else ''
                date_txt = row['month_year'] if 'month_year' in row.index else ''
                sm = sum(nums)
                np_ = sum(1 for n in nums if n % 2 == 0)
                rd = reduction_numerologique(sm)

                col1, col2, col3 = st.columns([2, 5, 3])
                with col1:
                    st.markdown(f"**{jour}**")
                    st.caption(date_txt)
                with col2:
                    st.markdown(render_balls(nums, ch), unsafe_allow_html=True)
                with col3:
                    st.caption(f"Σ {sm} → {rd} | {np_}P/{5-np_}I | Amp {max(nums)-min(nums)}")

                if idx < len(df_page) - 1:
                    st.markdown("<hr style='margin:4px 0;border:none;border-top:1px solid rgba(255,255,255,0.05)'>", unsafe_allow_html=True)

    # ═══ CHATBOT ═══
    with chat_col:
        st.markdown("### 🤖 Agent IA")
        combos=engine.count_combos();red=(1-combos/TOTAL_COMBINATIONS)*100
        st.markdown(f'<div class="combo-counter">{combos:,}</div>',unsafe_allow_html=True)
        st.markdown(f'<p style="text-align:center;color:#888;font-size:0.8em;">combinaisons (-{red:.1f}%)</p>',unsafe_allow_html=True)
        st.progress(min(red/100,1.0))
        if engine.filters_log:
            for log in engine.filters_log[-4:]: st.markdown(f"<span class='filter-badge'>{log}</span>",unsafe_allow_html=True)
        if st.button("🔄 Reset",use_container_width=True,key="rb"):
            st.session_state.engine=ComboEngine(stats);st.session_state.messages=[{'role':'assistant','content':'🔄 Reset !'}];st.rerun()
        st.markdown("---")
        cc=st.container(height=400)
        with cc:
            for msg in st.session_state.messages:
                with st.chat_message(msg['role']): st.markdown(msg['content'])
        if prompt:=st.chat_input("Dis-moi...",key="ci"):
            st.session_state.messages.append({'role':'user','content':prompt})
            resp=chatbot_respond(prompt,engine,stats,preds,model,scaler,df_features)
            st.session_state.messages.append({'role':'assistant','content':resp})
            st.rerun()

def generate_predictions(model,scaler,df_features,n=5):
    from models_functions import predict_next_draw
    preds=[predict_next_draw(model,scaler,df_features)]
    for _ in range(n-1):
        last=df_features.tail(WINDOW_LENGTH).copy()
        noisy=scaler.transform(last.values)+np.random.normal(0,0.5,size=last.shape)*0.5
        sp=model.predict(np.array([noisy]),verbose=0)
        nb_f=df_features.shape[1];pad=np.zeros((1,nb_f));pad[0,:NB_LABEL_FEATURES]=sp[0]
        raw=scaler.inverse_transform(pad)[0,:NB_LABEL_FEATURES]
        p=np.round(raw).astype(int)
        for i in range(5): p[i]=np.clip(p[i],1,49)
        p[5]=np.clip(p[5],1,10)
        nums=list(p[:5]);seen=set()
        for i,nn in enumerate(nums):
            while nn in seen: nn=nn+1 if nn<49 else 1
            seen.add(nn);nums[i]=nn
        p[:5]=sorted(nums);preds.append(p)
    return preds

def score_combination(nums,chance,stats,predictions=None):
    w={'lstm_match':15,'frequency':15,'retard':20,'sum_range':10,'even_odd':10,'decade_spread':15,'consecutive':5,'amplitude':10}
    sc={}
    if predictions:
        mt=0
        for pred in predictions: mt+=len(set(nums)&set(pred[:5])); mt+=(1 if pred[5]==chance else 0)
        sc['lstm_match']=(mt/(len(predictions)*6))*100
    else: sc['lstm_match']=50
    mf=max(stats['freq_nums'].values())
    sc['frequency']=np.mean([1-abs(stats['freq_nums'].get(n,0)/mf-0.5)*2 for n in nums])*100
    ar=np.mean(list(stats['retards'].values()))
    sc['retard']=np.mean([min(stats['retards'].get(n,0)/(ar+1),2)/2 for n in nums])*100
    z=abs(sum(nums)-stats['somme_moy'])/(stats['somme_std']+1)
    sc['sum_range']=max(0,(1-z/3))*100
    np_=sum(1 for n in nums if n%2==0)
    sc['even_odd']=100 if np_ in[2,3] else(50 if np_ in[1,4] else 10)
    sc['decade_spread']=(len(set((n-1)//10 for n in nums))/5)*100
    sn=sorted(nums);nc=sum(1 for i in range(4) if sn[i+1]-sn[i]==1)
    sc['consecutive']=max(0,(1-nc/3))*100
    amp=max(nums)-min(nums)
    sc['amplitude']=100 if 25<=amp<=40 else(60 if 15<=amp<=45 else 20)
    total=sum(sc[k]*w[k] for k in sc)
    return round((total/(sum(w.values())*100))*100,1),sc

if __name__=='__main__': main()