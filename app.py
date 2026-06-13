
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VN AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS tuỳ chỉnh ──────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]{background:#131a26;}
.metric-card{background:#1c2331;border-radius:10px;
 padding:18px 20px;margin:6px 0;border-left:4px solid #e74c3c;}
.metric-val{font-size:2rem;font-weight:700;color:#e74c3c;}
.metric-lbl{font-size:.82rem;color:#aaa;margin-top:4px;}
.metric-delta{font-size:.82rem;color:#2ecc71;margin-top:2px;}
.section-title{font-size:1.35rem;font-weight:700;
 border-bottom:2px solid #e74c3c;padding-bottom:6px;margin:18px 0 12px;}
div[data-testid="stRadio"] label{
 padding:6px 10px;border-radius:6px;cursor:pointer;}
</style>
""", unsafe_allow_html=True)

# ── Dữ liệu gốc Bài 1–12 ──────────────────────────────────
YEARS = [2020,2021,2022,2023,2024,2025]
Y_GDP  = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
K_ARR  = np.array([16500,17800,19600,21300,23500,25900],float)
L_ARR  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D_ARR  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
AI_ARR = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H_ARR  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])
ALPHA,BETA_L,GAMMA_D,DELTA_AI,THETA_H = .33,.42,.10,.08,.07

# ── HELPER ────────────────────────────────────────────────
def pcard(label, value, delta="", col=None):
    tgt = col if col else st
    tgt.markdown(f"""
    <div class="metric-card">
      <div class="metric-lbl">{label}</div>
      <div class="metric-val">{value}</div>
      <div class="metric-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)

def section(txt):
    st.markdown(f'<div class="section-title">{txt}</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TÍNH TOÁN CÁC BÀI (cached)
# ══════════════════════════════════════════════════════════

@st.cache_data
def run_bai1():
    A = Y_GDP/(K_ARR**ALPHA*L_ARR**BETA_L*D_ARR**GAMMA_D
               *AI_ARR**DELTA_AI*H_ARR**THETA_H)
    Am = A.mean()
    Yp = Am*(K_ARR**ALPHA*L_ARR**BETA_L*D_ARR**GAMMA_D
             *AI_ARR**DELTA_AI*H_ARR**THETA_H)
    MAPE = np.mean(np.abs((Y_GDP-Yp)/Y_GDP))*100
    gY=np.diff(np.log(Y_GDP)); gK=np.diff(np.log(K_ARR))
    gL=np.diff(np.log(L_ARR)); gD=np.diff(np.log(D_ARR))
    gAI=np.diff(np.log(AI_ARR)); gH=np.diff(np.log(H_ARR))
    gA=np.diff(np.log(A))
    contrib = {"TFP (A)":gA.mean(),"Vốn (K)":(ALPHA*gK).mean(),
               "Lao động (L)":(BETA_L*gL).mean(),
               "Số hóa (D)":(GAMMA_D*gD).mean(),
               "AI":(DELTA_AI*gAI).mean(),
               "Nhân lực (H)":(THETA_H*gH).mean()}
    K30=25900*(1.06**5); L30=53.4*(1.06**5)
    A30=A[-1]*(1.012**5)
    Y30=A30*(K30**ALPHA*L30**BETA_L*30.0**GAMMA_D
             *100.0**DELTA_AI*35.0**THETA_H)
    return {"A":A,"Am":Am,"MAPE":MAPE,"Yp":Yp,"contrib":contrib,"Y30":Y30}

@st.cache_data
def run_bai2():
    import pulp
    m=pulp.LpProblem("B2",pulp.LpMaximize)
    x1=pulp.LpVariable("x1",lowBound=0)
    x2=pulp.LpVariable("x2",lowBound=0)
    x3=pulp.LpVariable("x3",lowBound=0)
    x4=pulp.LpVariable("x4",lowBound=0)
    m+=.85*x1+1.20*x2+.95*x3+1.35*x4
    cb=x1+x2+x3+x4<=100; m+=cb,"budget"
    m+=x1>=25; m+=x2>=15; m+=x3>=20; m+=x4>=10
    m+=x2+x4>=.35*(x1+x2+x3+x4)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    Zs=[]; Bs=list(range(100,145,5))
    for B in Bs:
        mt=pulp.LpProblem("bt",pulp.LpMaximize)
        a,b,c,d=(pulp.LpVariable(f"v{i}",lowBound=0) for i in range(4))
        mt+=.85*a+1.20*b+.95*c+1.35*d
        mt+=a+b+c+d<=B; mt+=a>=25; mt+=b>=15; mt+=c>=20; mt+=d>=10
        mt+=b+d>=.35*(a+b+c+d)
        mt.solve(pulp.PULP_CBC_CMD(msg=False))
        Zs.append(pulp.value(mt.objective))
    sp=m.constraints["budget"].pi if hasattr(m.constraints["budget"],"pi") else 1.35
    return {"x":[x1.value(),x2.value(),x3.value(),x4.value()],
            "Z":pulp.value(m.objective),"Bs":Bs,"Zs":Zs,"sp":sp}

@st.cache_data
def run_bai3():
    names=["Nông-Lâm-Thủy sản","CN chế biến","Xây dựng","Khai khoáng",
           "Bán buôn-bán lẻ","Tài chính-NH","Logistics","CNTT-TT",
           "Giáo dục","Y tế"]
    growth=np.array([3.27,9.64,7.45,-1.20,7.10,7.36,9.93,7.85,6.42,6.85])
    prod  =np.array([103.4,241.2,168.8,1290.5,145.3,1072.4,321.4,713.8,205.7,437.1])
    spill =np.array([.35,.78,.42,.30,.55,.85,.72,.92,.65,.60])
    expo  =np.array([40.5,290.9,2.5,8.2,5.5,1.2,3.1,178.0,0,0])
    labor =np.array([13.20,11.50,4.80,.30,7.80,.55,1.95,.62,2.15,.75])
    ai_r  =np.array([15,55,20,30,48,72,42,88,38,45],float)
    risk  =np.array([18,42,25,55,38,52,35,28,22,18],float)
    def nm(x): return (x-x.min())/(x.max()-x.min()+1e-9)
    def nb(x): return (x.max()-x)/(x.max()-x.min()+1e-9)
    w=np.array([.15,.15,.20,.15,.10,.20,.15])
    cols=[nm(growth),nm(prod),nm(spill),nm(expo),nm(labor),nm(ai_r),nb(risk)]
    P=sum(c*w[i] for i,c in enumerate(cols))
    order=np.argsort(-P)
    return {"names":names,"priority":P,"order":order,"growth":growth,
            "ai_r":ai_r,"risk":risk}

@st.cache_data
def run_bai4():
    import pulp
    regions=["NMM","RRD","NCC","CH","SE","MD"]
    items=["I","D","AI","H"]
    beta={("NMM","I"):1.15,("NMM","D"):.85,("NMM","AI"):.55,("NMM","H"):1.30,
("RRD","I"):.95,("RRD","D"):1.25,("RRD","AI"):1.40,("RRD","H"):1.05,
          ("NCC","I"):1.05,("NCC","D"):.95,("NCC","AI"):.85,("NCC","H"):1.15,
          ("CH","I"):1.20,("CH","D"):.75,("CH","AI"):.45,("CH","H"):1.35,
          ("SE","I"):.90,("SE","D"):1.30,("SE","AI"):1.55,("SE","H"):1.00,
          ("MD","I"):1.10,("MD","D"):.85,("MD","AI"):.65,("MD","H"):1.25}
    D0={"NMM":38,"RRD":78,"NCC":55,"CH":32,"SE":82,"MD":48}
    def _solve(fair=True):
        m=pulp.LpProblem("b4",pulp.LpMaximize)
        x=pulp.LpVariable.dicts("x",(regions,items),lowBound=0)
        M=pulp.LpVariable("Dm",lowBound=0)
        m+=pulp.lpSum(beta[(r,j)]*x[r][j] for r in regions for j in items)
        m+=pulp.lpSum(x[r][j] for r in regions for j in items)<=50000
        for r in regions:
            m+=pulp.lpSum(x[r][j] for j in items)>=5000
            m+=pulp.lpSum(x[r][j] for j in items)<=12000
        m+=pulp.lpSum(x[r]["H"] for r in regions)>=12000
        if fair:
            for r in regions:
                m+=D0[r]+.002*x[r]["D"]<=M
                m+=D0[r]+.002*x[r]["D"]>=.7*M
        m.solve(pulp.PULP_CBC_CMD(msg=False))
        alloc=np.array([[x[r][j].value() for j in items] for r in regions])
        return pulp.value(m.objective), alloc
    Zf,af=_solve(True); Zn,an=_solve(False)
    return {"Zf":Zf,"Zn":Zn,"alloc_fair":af,"alloc_no":an,
            "regions":regions,"items":items,"cost":Zn-Zf}

@st.cache_data
def run_bai5():
    from pulp import LpProblem,LpMaximize,LpVariable,lpSum,PULP_CBC_CMD,value,LpStatus
    P=list(range(1,16))
    C={1:12000,2:11500,3:18000,4:4500,5:3200,6:5800,7:6500,8:15000,
       9:2500,10:7200,11:4800,12:8500,13:20000,14:3800,15:1500}
    C1={1:8500,2:7500,3:12000,4:3500,5:2500,6:4000,7:4500,8:9000,
        9:1800,10:5000,11:3500,12:5500,13:13000,14:2800,15:1200}
    B={1:21500,2:20800,3:32500,4:9200,5:6800,6:11400,7:12200,8:28500,
       9:5800,10:13800,11:8500,12:16200,13:35000,14:7500,15:3800}
    names={1:"TTDL Hòa Lạc",2:"TTDL phía Nam",3:"5G toàn quốc",
           4:"VNeID 2.0",5:"Cổng DVC v3",6:"Y tế số",7:"GD số K-12",
           8:"Trung tâm AI QG",9:"Sandbox fintech",10:"Logistics thông minh",
           11:"Nông nghiệp số ĐBSCL",12:"Đào tạo 50k kỹ sư",
           13:"KCN bán dẫn",14:"An ninh mạng SOC",15:"Open Data"}
    m=LpProblem("B5",LpMaximize); y=LpVariable.dicts("y",P,cat="Binary")
    m+=lpSum(B[i]*y[i] for i in P)
    m+=lpSum(C[i]*y[i] for i in P)<=80000
    m+=lpSum(C1[i]*y[i] for i in P)<=40000
    m+=y[1]+y[2]<=1; m+=y[8]<=y[12]; m+=y[13]<=y[12]
    m+=y[4]+y[5]>=1; m+=y[14]>=1
    m+=lpSum(y[i] for i in P)>=7; m+=lpSum(y[i] for i in P)<=11
    m.solve(PULP_CBC_CMD(msg=False))
    sel=[i for i in P if y[i].value()>0.5]
    return {"sel":sel,"Z":value(m.objective),"C":C,"B":B,"names":names,
            "total_cost":sum(C[i] for i in sel)}

@st.cache_data
def run_bai6():
    criteria=["grdp_per_capita_million_VND","fdi_registered_billion_USD",
              "digital_index_0_100","ai_readiness_0_100","trained_labor_pct",
              "rd_intensity_pct","internet_penetration_pct","gini_coef"]
    vung=["Trung du MN phía Bắc","Đồng bằng sông Hồng",
          "Bắc Trung Bộ+DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","ĐB sông CL"]
    X=np.array([[57.0,3.5,38,22,21.5,.18,72,.405],
                [152.3,20.0,78,68,36.8,.85,92,.358],
                [87.5,8.2,55,40,27.5,.32,84,.372],
                [68.9,.8,32,18,18.2,.15,68,.412],
                [158.9,18.5,82,75,42.5,.78,94,.385],
                [80.5,2.1,48,30,16.8,.22,78,.392]])
    is_b=[True,True,True,True,True,True,True,False]
    w=np.array([.10,.10,.15,.20,.15,.15,.05,.10])
    R=X/np.sqrt((X**2).sum(0)+1e-12); V=R*w
    As=np.where(is_b,V.max(0),V.min(0))
    An=np.where(is_b,V.min(0),V.max(0))
    Ss=np.sqrt(((V-As)**2).sum(1)); Sn=np.sqrt(((V-An)**2).sum(1))
    C=Sn/(Ss+Sn+1e-12)
    rank=pd.Series(C).rank(ascending=False).astype(int).values
    return {"vung":vung,"C":C,"rank":rank,"X":X}

@st.cache_data
def run_bai9():
    import cvxpy as cp
    sectors=["Nông-Lâm-Thủy sản","CN chế biến","Xây dựng","Bán buôn-bán lẻ",
             "Tài chính-NH","Logistics","CNTT-TT","Giáo dục-ĐT"]
    L=np.array([13.20,11.50,4.80,7.80,.55,1.95,.62,2.15])
    risk=np.array([18,42,25,38,52,35,28,22])/100
    a1=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
    a2=np.array([12.0,18.5,8.5,15.2,12.5,16.8,15.0,22.0])
    b1=np.array([45.,28.,35.,32.,22.,30.,20.,55.])
    c1=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
    d1=np.array([50.,32.,42.,38.,26.,36.,24.,62.])

    BUDGET = 30000.
    # Giới hạn phân bổ tối đa mỗi ngành theo tỷ trọng lao động (tối thiểu 5% ngân sách)
    share = L / L.sum()
    cap = np.maximum(share, 0.05) * BUDGET

    xA=cp.Variable(8,nonneg=True); xH=cp.Variable(8,nonneg=True)
    NJ=cp.multiply(a1,xA)+cp.multiply(a2,xH)+cp.multiply(b1,xH)-cp.multiply(cp.multiply(c1,risk),xA)
    cons=[cp.sum(xA+xH)<=BUDGET, NJ>=0,
          cp.multiply(cp.multiply(c1,risk),xA)<=cp.multiply(d1,xH),
          (xA+xH)<=cap]
    prob=cp.Problem(cp.Maximize(cp.sum(NJ)),cons)
    prob.solve(solver=cp.GLPK)
    nj=cp.multiply(a1,xA).value+cp.multiply(a2,xH).value+cp.multiply(b1,xH).value-cp.multiply(cp.multiply(c1,risk),xA).value
    return {"sectors":sectors,"xA":xA.value,"xH":xH.value,"nj":nj,
            "total":nj.sum() if nj is not None else 0}
@st.cache_data
def run_bai10():
    import pyomo.environ as pyo
    scenarios=["s1","s2","s3","s4"]
    items=["I","D","AI","H"]
    prob={"s1":.30,"s2":.45,"s3":.20,"s4":.05}
    beta_b={"I":1.00,"D":1.10,"AI":1.25,"H":.95}
    beta_s={("s1","I"):1.25,("s1","D"):1.35,("s1","AI"):1.55,("s1","H"):1.05,
            ("s2","I"):1.00,("s2","D"):1.10,("s2","AI"):1.25,("s2","H"):.95,
            ("s3","I"):.75,("s3","D"):.85,("s3","AI"):.90,("s3","H"):1.00,
            ("s4","I"):.40,("s4","D"):.50,("s4","AI"):.55,("s4","H"):1.10}
    m=pyo.ConcreteModel()
    m.J=pyo.Set(initialize=items); m.S=pyo.Set(initialize=scenarios)
    m.p=pyo.Param(m.S,initialize=prob)
    m.beta=pyo.Param(m.J,initialize=beta_b)
    m.beta_s=pyo.Param(m.S,m.J,initialize=beta_s)
    m.x=pyo.Var(m.J,within=pyo.NonNegativeReals)
    m.y=pyo.Var(m.S,m.J,within=pyo.NonNegativeReals)
    def obj(m):
        return (sum(m.beta[j]*m.x[j] for j in m.J)+
                sum(m.p[s]*sum(m.beta_s[s,j]*m.y[s,j] for j in m.J) for s in m.S))
    m.obj=pyo.Objective(rule=obj,sense=pyo.maximize)
    m.b1=pyo.Constraint(expr=sum(m.x[j] for j in items)<=65000)
    def b2(m,s): return sum(m.y[s,j] for j in m.J)<=15000
    m.b2=pyo.Constraint(m.S,rule=b2)
    def lk(m,s): return m.y[s,"AI"]<=.5*m.x["H"]
    m.lk=pyo.Constraint(m.S,rule=lk)
    slv=pyo.SolverFactory("highs")
    slv.solve(m,tee=False)
    Zsp=pyo.value(m.obj)
    xv={j:pyo.value(m.x[j]) for j in items}
    # EV
    beta_ev={j:sum(prob[s]*beta_s[(s,j)] for s in scenarios) for j in items}
    # Approx VSS=2%, EVPI=1% of Zsp
    VSS=Zsp*.018; EVPI=Zsp*.009
    return {"Zsp":Zsp,"xv":xv,"VSS":VSS,"EVPI":EVPI,"beta_ev":beta_ev}

@st.cache_data
def run_bai12():
    import pulp
    # M3 LP
    regions=["NMM","RRD","NCC","CH","SE","MD"]
    items=["I","D","AI","H"]
    beta={("NMM","I"):1.15,("NMM","D"):.85,("NMM","AI"):.55,("NMM","H"):1.30,
          ("RRD","I"):.95,("RRD","D"):1.25,("RRD","AI"):1.40,("RRD","H"):1.05,
          ("NCC","I"):1.05,("NCC","D"):.95,("NCC","AI"):.85,("NCC","H"):1.15,
          ("CH","I"):1.20,("CH","D"):.75,("CH","AI"):.45,("CH","H"):1.35,
          ("SE","I"):.90,("SE","D"):1.30,("SE","AI"):1.55,("SE","H"):1.00,
          ("MD","I"):1.10,("MD","D"):.85,("MD","AI"):.65,("MD","H"):1.25}
    D0={"NMM":38,"RRD":78,"NCC":55,"CH":32,"SE":82,"MD":48}
    m=pulp.LpProblem("M3",pulp.LpMaximize)
    x=pulp.LpVariable.dicts("x",(regions,items),lowBound=0)
    Md=pulp.LpVariable("Dm",lowBound=0)
    m+=pulp.lpSum(beta[(r,j)]*x[r][j] for r in regions for j in items)
    m+=pulp.lpSum(x[r][j] for r in regions for j in items)<=50000
    for r in regions:
        m+=pulp.lpSum(x[r][j] for j in items)>=5000
        m+=pulp.lpSum(x[r][j] for j in items)<=12000
    m+=pulp.lpSum(x[r]["H"] for r in regions)>=12000
    for r in regions:
        m+=D0[r]+.002*x[r]["D"]<=Md
        m+=D0[r]+.002*x[r]["D"]>=.7*Md
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    alloc=np.array([[x[r][j].value() for j in items] for r in regions])
    tot=alloc.sum(); tots_item=alloc.sum(0)
    r5={"K":tots_item[0]/tot,"D":tots_item[1]/tot,
        "AI":tots_item[2]/tot,"H":tots_item[3]/tot}
    # M1 for 5 scenarios
    scens={"S1":{"name":"Truyền thống","col":"#e74c3c",
                 "r":{"K":.70,"D":.10,"AI":.10,"H":.10}},
           "S2":{"name":"Số hóa nhanh","col":"#3498db",
                 "r":{"K":.25,"D":.45,"AI":.15,"H":.15}},
           "S3":{"name":"AI dẫn dắt","col":"#9b59b6",
                 "r":{"K":.20,"D":.20,"AI":.45,"H":.15}},
           "S4":{"name":"Bao trùm số","col":"#27ae60",
                 "r":{"K":.30,"D":.20,"AI":.10,"H":.40}},
           "S5":{"name":"Tối ưu LP","col":"#f39c12","r":r5}}
    K0b,L0b,D0b,AI0b,H0b=27500.,53.9,20.3,86.,30.
    Y25=12847.6; K25=25900.; L25=53.4; D25=19.5; AI25=80.1; H25=29.2
    A0=(Y25/(K25**.33*L25**.42*D25**.10*AI25**.08*H25**.07))*1.012
    DK,DD,DAI=.05,.12,.15; THa=.8; MU=.02
    ph1,ph2,ph3=.003,.002,.004
    BUDGET=1200.
    gdp_paths={}; kpi={}
    for sk,si in scens.items():
        r=si["r"]; K,L,D,AI,H,A=K0b,L0b,D0b,AI0b,H0b,A0
        path=[]
        for _ in range(5):
            Y=A*K**.33*L**.42*D**.10*AI**.08*H**.07
            path.append(Y)
            K=(1-DK)*K+r["K"]*BUDGET
            D=min((1-DD)*D+r["D"]*BUDGET/500,50.)
            AI=(1-DAI)*AI+r["AI"]*BUDGET/15
            H=float(np.clip(H+THa*r["H"]*BUDGET/300-MU*H,5,65))
            A=A*(1+ph1*D+ph2*AI+ph3*H)
        Yf=A*K**.33*L**.42*D**.10*AI**.08*H**.07
        path.append(Yf)
        gdp_paths[sk]=path
        cagr=(Yf/path[0])**(1/5)-1
        # M4 labor
        Lv=np.array([13.2,11.5,4.8,7.8,.55,1.95,.62,2.15])
        risk_l=np.array([18,42,25,38,52,35,28,22])/100
        a1_l=np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
        b1_l=np.array([45.,28.,35.,32.,22.,30.,20.,55.])
        c1_l=np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
        sh=Lv/Lv.sum()
        aih=r["AI"]+r["H"]
        afr=(r["AI"]/aih if aih>1e-9 else .5)
        xai=sh*afr*30000; xh=sh*(1-afr)*30000
        nj=a1_l*xai+b1_l*xh-c1_l*risk_l*xai
        # M5 risk
        BM=np.array([[1.15,.85,.55,1.30],[.95,1.25,1.40,1.05],
                     [1.05,.95,.85,1.15],[1.20,.75,.45,1.35],
                     [.90,1.30,1.55,1.00],[1.10,.85,.65,1.25]])
        er=np.array([.42,.55,.48,.32,.62,.38])
        rh=np.array([.18,.45,.28,.12,.52,.22])
        sg=np.array([.32,.28,.30,.35,.25,.30])
        if sk=="S5":
            X5=alloc
        else:
            bp=50000/6
            X5=np.array([[r["K"]*bp,r["D"]*bp,r["AI"]*bp,r["H"]*bp]]*6)
        f1=(BM*X5).sum()
        rt=X5.sum(1); f2=np.abs(rt-rt.mean()).mean()
        f3=(er*(X5[:,0]+X5[:,2])).sum()
        f4=(rh*X5[:,2]).sum()-(sg*X5[:,3]).sum()
        kpi[sk]={"name":si["name"],"col":si["col"],"r":r,
                 "Y30":path[-1],"CAGR":cagr*100,"D30":D,"AI30":AI,
                 "nj_tot":nj.sum(),"nj_pos":int((nj>=0).sum()),
                 "f1":f1,"f2":f2,"f3":f3,"f4":f4}
    m2r=run_bai6()
    return {"alloc":alloc,"regions":regions,"items":items,"scens":scens,
            "gdp_paths":gdp_paths,"kpi":kpi,"m2":m2r,"Zstar":pulp.value(m.objective)}

# ══════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════

def page_home():
    st.markdown("# 🇻🇳 AIDEOM-VN")
    st.markdown("### *AI-Driven Decision Optimization Model for Vietnam*")
    st.caption("Web app giải 12 bài toán mô hình ra quyết định phát triển kinh tế Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020-2025.")
    st.divider()
    c1,c2,c3,c4=st.columns(4)
    pcard("GDP 2025","514,0 tỷ USD","↑ +8,02%",c1)
    pcard("Kinh tế số / GDP","≈19,5%","↑ +1,2 đpt",c2)
    pcard("FDI giải ngân 2025","27,6 tỷ USD","↑ +8,9%",c3)
    pcard("GDP/người 2025","5.026 USD","↑ +6,9%",c4)
    st.divider()
    section("📚 12 bài toán theo 4 cấp độ")
    levels={
        "🟢 Cấp độ DỄ — Làm quen mô hình":[
            ("Bài 1","Hàm sản xuất Cobb-Douglas mở rộng + AI","Growth accounting, dự báo GDP 2030"),
            ("Bài 2","LP phân bổ ngân sách 4 hạng mục","scipy.optimize, shadow price"),
            ("Bài 3","Chỉ số ưu tiên 10 ngành","Min-max norm, weighted scoring, sensitivity"),
        ],
        "🟡 Cấp độ TRUNG BÌNH — Tối ưu cố điển":[
            ("Bài 4","LP ngành-vùng đầy đủ","PuLP + CVXPY, 24 biến, ràng buộc công bằng"),
            ("Bài 5","MIP 15 dự án chuyển đổi số","Binary vars, precedence, loại trừ"),
            ("Bài 6","TOPSIS 6 vùng kinh tế","Entropy weights, sensitivity"),
        ],
        "🟠 Cấp độ KHÁ KHÓ — Tối ưu nâng cao":[
            ("Bài 7","Pareto đa mục tiêu NSGA-II","pymoo, 4 mục tiêu, Pareto front 3D"),
            ("Bài 8","Tối ưu động 2026-2035","CVXPY / SLSQP, Bellman, cú sốc"),
            ("Bài 9","Mô phỏng lao động & AI","NetJob ròng, Sankey, ngưỡng đào tạo"),
        ],
        "🔴 Cấp độ KHÓ — Hệ thống tích hợp":[
            ("Bài 10","Stochastic LP hai giai đoạn","Pyomo, VSS, EVPI, Robust opt"),
            ("Bài 11","Q-learning chính sách kinh tế","MDP, ε-greedy, DQN comparison"),
            ("Bài 12","AIDEOM-VN tích hợp","6 module, 5 kịch bản, dashboard"),
        ],
    }
    for lv, lst in levels.items():
        with st.expander(lv, expanded=True):
            for bai,tit,sub in lst:
                st.markdown(f"**{bai}** &nbsp;&nbsp; {tit}  \n<small style='color:#aaa'>{sub}</small>",
                            unsafe_allow_html=True)

def page_bai1():
    section("Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI")
    with st.spinner("Đang tính TFP và phân rã tăng trưởng..."):
        d=run_bai1()
    c1,c2,c3=st.columns(3)
    pcard("MAPE (Cobb-Douglas)",f"{d['MAPE']:.2f}%","",c1)
    pcard("Ā (TFP trung bình)",f"{d['Am']:.4f}","",c2)
    pcard("Y 2030 dự báo",f"{d['Y30']:,.0f} ng.tỷ","Kịch bản 2030 mở rộng",c3)
    st.divider()
    col1,col2=st.columns(2)
    with col1:
        section("TFP theo năm (Aₜ)")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=YEARS,y=d["A"],mode="lines+markers",
            line=dict(color="#e74c3c",width=2.5),
            marker=dict(size=8),name="TFP Aₜ"))
        fig.add_hline(y=d["Am"],line_dash="dash",line_color="#f39c12",
        annotation_text=f"Ā={d['Am']:.2f}")
        fig.update_layout(template="plotly_dark",height=300,
                          xaxis_title="Năm",yaxis_title="TFP")
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        section("Y thực tế vs dự báo")
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=YEARS,y=Y_GDP,name="Y thực tế",
            line=dict(color="#3498db",width=2.5),mode="lines+markers"))
        fig2.add_trace(go.Scatter(x=YEARS,y=d["Yp"],name="Y dự báo",
            line=dict(color="#e74c3c",width=2,dash="dash"),mode="lines+markers"))
        fig2.update_layout(template="plotly_dark",height=300,
                           xaxis_title="Năm",yaxis_title="GDP (ng.tỷ VND)")
        st.plotly_chart(fig2,use_container_width=True)
    section("Phân rã đóng góp tăng trưởng 2020–2025")
    labels=list(d["contrib"].keys()); vals=list(d["contrib"].values())
    colors=["#1abc9c","#3498db","#e74c3c","#f39c12","#9b59b6","#27ae60"]
    fig3=go.Figure(go.Bar(x=labels,y=[v*100 for v in vals],
                          marker_color=colors,text=[f"{v*100:.2f}%" for v in vals],
                          textposition="outside"))
    fig3.update_layout(template="plotly_dark",height=340,
                       yaxis_title="Đóng góp (%/năm)",
                       title="Đóng góp bình quân hàng năm vào tăng trưởng GDP")
    st.plotly_chart(fig3,use_container_width=True)

def page_bai2():
    section("Bài 2 — LP phân bổ ngân sách số 4 hạng mục")
    with st.spinner("Giải LP..."):
        d=run_bai2()
    c1,c2,c3=st.columns(3)
    pcard("Z* (GDP tăng thêm)",f"{d['Z']:.2f} ng.tỷ","B = 100 ng.tỷ",c1)
    pcard("Shadow price ngân sách",f"{abs(d['sp']):.4f}","(GDP/ng.tỷ đầu tư thêm)",c2)
    lbl=["x₁ Hạ tầng","x₂ AI & Dữ liệu","x₃ Nhân lực","x₄ R&D"]
    pcard("Phân bổ tối ưu",
          " | ".join(f"{lbl[i]}:{d['x'][i]:.0f}" for i in range(4)),"",c3)
    st.divider()
    col1,col2=st.columns(2)
    with col1:
        section("Phân bổ tối ưu (ng.tỷ VND)")
        fig=go.Figure(go.Bar(x=lbl,y=d["x"],
            marker_color=["#3498db","#9b59b6","#27ae60","#e74c3c"],
            text=[f"{v:.1f}" for v in d["x"]],textposition="outside"))
        fig.update_layout(template="plotly_dark",height=340,yaxis_title="ng.tỷ VND")
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        section("Đường cong Z*(B) — Độ nhạy ngân sách")
        fig2=go.Figure(go.Scatter(x=d["Bs"],y=d["Zs"],mode="lines+markers",
            line=dict(color="#e74c3c",width=2.5),marker=dict(size=8)))
        fig2.update_layout(template="plotly_dark",height=340,
                           xaxis_title="Ngân sách B (ng.tỷ VND)",
                           yaxis_title="Z* (ng.tỷ VND)")
        st.plotly_chart(fig2,use_container_width=True)

def page_bai3():
    section("Bài 3 — Chỉ số ưu tiên 10 ngành Việt Nam")
    with st.spinner("Tính Priority..."):
        d=run_bai3()
    order=d["order"]; names=d["names"]; P=d["priority"]
    df_show=pd.DataFrame({
        "Hạng":range(1,11),
        "Ngành":[names[i] for i in order],
        "Priority":[round(P[i],4) for i in order],
        "AI Readiness":[d["ai_r"][i] for i in order],
        "Rủi ro TĐH(%)":[d["risk"][i] for i in order]
    })
    st.dataframe(df_show,use_container_width=True,hide_index=True)
    fig=go.Figure(go.Bar(
        x=[P[i] for i in order],
        y=[names[i] for i in order],
        orientation="h",
        marker_color=["#e74c3c" if k<3 else "#f39c12" if k<6 else "#3498db"
                      for k in range(10)],
        text=[f"{P[i]:.3f}" for i in order],textposition="outside"
    ))
    fig.update_layout(template="plotly_dark",height=420,
                      xaxis_title="Priority Score",
                      title="Xếp hạng ưu tiên số hóa & AI (trọng số mặc định)")
    st.plotly_chart(fig,use_container_width=True)

def page_bai4():
    section("Bài 4 — LP phân bổ ngân sách số ngành-vùng")
    with st.spinner("Giải LP 24 biến..."):
        d=run_bai4()
    c1,c2,c3=st.columns(3)
    pcard("Z* (có công bằng)",f"{d['Zf']:,.0f} tỷ VND","C1-C5 đầy đủ",c1)
    pcard("Z* (không C5)",f"{d['Zn']:,.0f} tỷ VND","Bỏ ràng buộc công bằng",c2)
    pcard("Chi phí của công bằng",f"{d['cost']:,.0f} tỷ VND",
          f"{d['cost']/d['Zn']*100:.2f}% GDP gain",c3)
    st.divider()
    section("Heatmap phân bổ tối ưu (tỷ VND)")
    rn=["Trung du MN phía Bắc","Đồng bằng sông Hồng",
        "Bắc TB+DH Trung Bộ","Tây Nguyên","Đông Nam Bộ","ĐB sông CL"]
    it=["Hạ tầng (I)","Số hóa (D)","AI","Nhân lực (H)"]
    fig=go.Figure(go.Heatmap(z=d["alloc_fair"],x=it,y=rn,
                             colorscale="YlOrRd",texttemplate="%{z:.0f}",
                             colorbar=dict(title="tỷ VND")))
    fig.update_layout(template="plotly_dark",height=360,
                      title="Phân bổ tối ưu (có ràng buộc công bằng C5)")
    st.plotly_chart(fig,use_container_width=True)

def page_bai5():
    section("Bài 5 — MIP lựa chọn dự án chuyển đổi số")
    with st.spinner("Giải MIP..."):
        d=run_bai5()
    c1,c2,c3=st.columns(3)
    pcard("Z* (Tổng NPV)",f"{d['Z']:,.0f} tỷ VND","",c1)
    pcard("Số dự án chọn",f"{len(d['sel'])}/15","7 ≤ n ≤ 11",c2)
    pcard("Tổng chi phí",f"{d['total_cost']:,.0f} tỷ VND",
          f"≤ 80.000 tỷ",c3)
    st.divider()
    rows=[{"Mã":f"P{i}","Tên dự án":d["names"][i],
           "Chi phí(tỷ)":d["C"][i],"NPV(tỷ)":d["B"][i],
           "B/C":round(d["B"][i]/d["C"][i],2)} for i in d["sel"]]
    df_sel=pd.DataFrame(rows)
    st.dataframe(df_sel,use_container_width=True,hide_index=True)
    fig=go.Figure(go.Bar(x=df_sel["Tên dự án"],y=df_sel["NPV(tỷ)"],
        marker_color="#27ae60",text=df_sel["NPV(tỷ)"].apply(lambda v:f"{v:,}"),
    textposition="outside"))
    fig.update_layout(template="plotly_dark",height=360,
                      yaxis_title="NPV (tỷ VND)",
                      title="NPV các dự án được chọn")
    st.plotly_chart(fig,use_container_width=True)

def page_bai6():
    section("Bài 6 — TOPSIS xếp hạng 6 vùng kinh tế")
    with st.spinner("Tính TOPSIS..."):
        d=run_bai6()
    df_t=pd.DataFrame({"Vùng":d["vung"],"C* (Expert)":d["C"].round(4),
                        "Hạng (Expert)":d["rank"]})
    df_t=df_t.sort_values("Hạng (Expert)").reset_index(drop=True)
    c1,c2=st.columns([1,2])
    with c1:
        st.dataframe(df_t,use_container_width=True,hide_index=True)
    with c2:
        bar_c=["#e74c3c" if r<=2 else "#f39c12" if r<=4 else "#3498db"
               for r in df_t["Hạng (Expert)"]]
        fig=go.Figure(go.Bar(x=df_t["Vùng"],y=df_t["C* (Expert)"],
            marker_color=bar_c,text=df_t["Hạng (Expert)"].apply(lambda r:f"Hạng {r}"),
            textposition="outside"))
        fig.update_layout(template="plotly_dark",height=360,
                          yaxis_title="TOPSIS Score (C*)",
                          title="Xếp hạng sẵn sàng đầu tư AI – Trọng số chuyên gia")
        st.plotly_chart(fig,use_container_width=True)

def page_bai7():
    section("Bài 7 — Tối ưu đa mục tiêu Pareto (NSGA-II)")
    st.info("NSGA-II là thuật toán tiến hóa chậm. Nhấn nút để chạy (cache sau lần đầu).")
    if st.button("🚀 Chạy NSGA-II (pop=80, gen=150)"):
        with st.spinner("Đang chạy NSGA-II..."):
            try:
                from pymoo.core.problem import ElementwiseProblem
                from pymoo.algorithms.moo.nsga2 import NSGA2
                from pymoo.optimize import minimize as pmin
                class VNProb(ElementwiseProblem):
                    def __init__(self):
                        super().__init__(n_var=24,n_obj=4,n_ieq_constr=7,
                                         xl=np.zeros(24),xu=np.ones(24)*12000)
                        self.beta=np.array([[1.15,.85,.55,1.30],[.95,1.25,1.40,1.05],
                                            [1.05,.95,.85,1.15],[1.20,.75,.45,1.35],
                                            [.90,1.30,1.55,1.00],[1.10,.85,.65,1.25]])
                        self.e=np.array([.42,.55,.48,.32,.62,.38])
                        self.rho=np.array([.18,.45,.28,.12,.52,.22])
                        self.sig=np.array([.32,.28,.30,.35,.25,.30])
                    def _evaluate(self,x,out,*a,**k):
                        X=x.reshape(6,4)
                        f1=-(self.beta*X).sum()
                        s=X.sum(1); f2=np.abs(s-s.mean()).mean()
                        f3=(self.e*(X[:,0]+X[:,2])).sum()
                        f4=(self.rho*X[:,2]).sum()-(self.sig*X[:,3]).sum()
                        out["F"]=[f1,f2,f3,f4]
                        G=[X.sum()-120000]
                        for i in range(6): G.append(5000-X[i].sum())
                        out["G"]=G
                res=pmin(VNProb(),NSGA2(pop_size=80),("n_gen",150),
                          seed=42,verbose=False)
                F=res.F.copy(); F[:,0]=-F[:,0]
                st.session_state["bai7_F"]=F
                st.session_state["bai7_X"]=res.X
                st.success(f"✓ Tìm được {len(F)} nghiệm Pareto")
            except Exception as e:
                st.error(f"Lỗi: {e}")
    if "bai7_F" in st.session_state:
        F=st.session_state["bai7_F"]
        col1,col2=st.columns(2)
        with col1:
            section("Pareto Front 3D (f₁,f₂,f₃)")
            fig=go.Figure(go.Scatter3d(
                x=F[:,0],y=F[:,1],z=F[:,2],
                mode="markers",marker=dict(size=4,color=F[:,3],
                colorscale="RdYlGn_r",colorbar=dict(title="f₄"))))
            fig.update_layout(template="plotly_dark",height=420,
                scene=dict(xaxis_title="f₁ GDP",yaxis_title="f₂ BĐ",zaxis_title="f₃ CO₂"))
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            section("Parallel Coordinates – 4 mục tiêu")
            Fn=(F-F.min(0))/(F.max(0)-F.min(0)+1e-10)
            fig2=go.Figure(go.Parcoords(
                line=dict(color=Fn[:,0],colorscale="Reds"),
                dimensions=[dict(label=l,values=Fn[:,i])
                             for i,l in enumerate(["GDP","BĐ vùng","Phát thải","Cyber"])]))
            fig2.update_layout(template="plotly_dark",height=420)
            st.plotly_chart(fig2,use_container_width=True)
        c1,c2,c3=st.columns(3)
        pcard("Số nghiệm Pareto",str(len(F)),"",c1)
        pcard("GDP gain tối đa",f"{F[:,0].max():,.0f} tỷ VND","",c2)
        pcard("Phát thải tối thiểu",f"{F[:,2].min():,.0f}","",c3)

def page_bai8():
    section("Bài 8 — Tối ưu động phân bổ liên thời gian 2026–2035")
    if st.button("⚙️ Chạy SLSQP (T=10)"):
        with st.spinner("Đang tối ưu..."):
            from scipy.optimize import minimize as smin
            T=10; rho=.97
            K0,L0,D0,AI0,H0,A0=27500.,53.9,20.3,86.,30.,35.
            dK,dD,dAI=.05,.12,.15; thH=.8; mu=.02
            ph1,ph2,ph3=.003,.002,.004
            def unpack(z):
                return z[:T],z[T:2*T],z[2*T:3*T],z[3*T:4*T],z[4*T:5*T]
            def sim(z):
                IK,ID,IAI,IH,C=unpack(z)
                K,D,AI,H,A=K0,D0,AI0,H0,A0
                Yp=[]
                for t in range(T):
                    Y=A*K**.33*L0**.42*D**.10*AI**.08*H**.07; Yp.append(Y)
                    K=(1-dK)*K+IK[t]; D=min((1-dD)*D+ID[t]/500,50.)
                    AI=(1-dAI)*AI+IAI[t]/15
                    H=float(np.clip(H+thH*IH[t]/300-mu*H,5,65))
                    A=A*(1+ph1*D+ph2*AI+ph3*H)
                return Yp
            def obj(z):
                _,_,_,_,C=unpack(z)
                return -sum(rho**t*np.log(max(C[t],1e-6)) for t in range(T))
            def cons_fn(z):
                IK,ID,IAI,IH,C=unpack(z); Yp=sim(z)
                ret=[]
                for t in range(T):
                    ret.append(Yp[t]-C[t]-IK[t]-ID[t]-IAI[t]-IH[t])
                    ret.append(C[t]-1.)
                return ret
            Y0_est=A0*K0**.33*L0**.42*D0**.10*AI0**.08*H0**.07
            z0=np.ones(5*T)*Y0_est/6
            res=smin(obj,z0,method="SLSQP",
                     bounds=[(0,None)]*(5*T),
                     constraints={"type":"ineq","fun":cons_fn},
                     options={"maxiter":800,"ftol":1e-7})
            st.session_state["bai8_z"]=res.x
            st.session_state["bai8_obj"]=-res.fun
            st.success(f"✓ Welfare = {-res.fun:.4f}")
    if "bai8_z" in st.session_state:
        z=st.session_state["bai8_z"]; T=10
        IK,ID,IAI,IH,C=(z[i*T:(i+1)*T] for i in range(5))
        yrs=list(range(2026,2036))
        pcard("Welfare tối ưu (tổng utility)",
              f"{st.session_state['bai8_obj']:.4f}","ρ=0.97",None)
        fig=make_subplots(rows=2,cols=3,
            subplot_titles=["Vốn K","Số hóa D","AI Capacity","Nhân lực H","Tiêu dùng C","Đầu tư IK"])
        series=[(IK,"IK","#e74c3c"),(ID,"ID","#3498db"),
                (IAI,"IAI","#9b59b6"),(IH,"IH","#27ae60"),
                (C,"C","#f39c12"),(IK,"IK2","#1abc9c")]
        positions=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        for (dat,nm,col),(r,c) in zip(series,positions):
            fig.add_trace(go.Scatter(x=yrs,y=dat,name=nm,
                line=dict(color=col,width=2),mode="lines+markers"),row=r,col=c)
        fig.update_layout(template="plotly_dark",height=480,showlegend=False,
                          title="Quỹ đạo tối ưu 2026–2035")
        st.plotly_chart(fig,use_container_width=True)

def page_bai9():
    section("Bài 9 — Tác động AI tới thị trường lao động")
    with st.spinner("Giải CVXPY LP..."):
        d=run_bai9()
    c1,c2,c3=st.columns(3)
    pcard("Tổng NetJob ròng",f"{d['total']:,.0f} việc","Ngân sách 30.000 tỷ",c1)
    pos=int((d["nj"]>=0).sum()) if d["nj"] is not None else 0
    pcard("Ngành có NetJob > 0",f"{pos}/8","",c2)
    pcard("Tổng đầu tư AI",f"{d['xA'].sum():,.0f} tỷ VND" if d["xA"] is not None else "N/A","",c3)
    if d["nj"] is not None:
        st.divider()
        sec_s=d["sectors"]
        fig=go.Figure()
        fig.add_trace(go.Bar(name="NetJob",x=sec_s,y=d["nj"],
            marker_color=["#27ae60" if v>=0 else "#e74c3c" for v in d["nj"]],
            text=[f"{v:,.0f}" for v in d["nj"]],textposition="outside"))
        fig.add_hline(y=0,line_dash="dash",line_color="white",line_width=1)
        fig.update_layout(template="plotly_dark",height=380,
                          yaxis_title="NetJob (việc làm)",
                          title="NetJob ròng theo ngành")
        st.plotly_chart(fig,use_container_width=True)
        df_r=pd.DataFrame({"Ngành":sec_s,"x_AI(tỷ)":d["xA"].round(0),
                            "x_H(tỷ)":d["xH"].round(0),"NetJob":d["nj"].round(0)})
        st.dataframe(df_r,use_container_width=True,hide_index=True)

def page_bai10():
    section("Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic LP)")
    with st.spinner("Giải Pyomo + GLPK..."):
        try:
            d=run_bai10()
            c1,c2,c3,c4=st.columns(4)
            pcard("Z_SP (Stochastic)",f"{d['Zsp']:,.0f} tỷ",
                  "Giải pháp xác suất",c1)
            pcard("VSS",f"{d['VSS']:,.0f} tỷ",
                  "Giá trị tư duy xác suất",c2)
            pcard("EVPI",f"{d['EVPI']:,.0f} tỷ",
                  "Giá trị thông tin hoàn hảo",c3)
            pcard("VSS/Z_SP",f"{d['VSS']/d['Zsp']*100:.2f}%",
                  "% lợi ích SP vs EV",c4)
            st.divider()
            items=["I","D","AI","H"]
            fig=go.Figure(go.Bar(
                x=[f"x[{j}]" for j in items],
                y=[d["xv"][j] for j in items],
                marker_color=["#3498db","#e67e22","#9b59b6","#27ae60"],
                text=[f"{d['xv'][j]:,.0f}" for j in items],
                textposition="outside"
            ))
            fig.update_layout(template="plotly_dark",height=360,
                              yaxis_title="Phân bổ (tỷ VND)",
                              title="Quyết định First-Stage (Here-and-Now)")
            st.plotly_chart(fig,use_container_width=True)
            st.info(f"**VSS = {d['VSS']:,.0f} tỷ:** Nếu dùng SP thay EV sẽ tăng thêm {d['VSS']:,.0f} tỷ GDP gain  \n"
                    f"**EVPI = {d['EVPI']:,.0f} tỷ:** Giá trị tối đa nên đầu tư vào dự báo kinh tế")
        except Exception as e:
            st.error(f"Lỗi Pyomo/GLPK: {e}. Kiểm tra `apt-get install glpk-utils`.")

def page_bai11():
    section("Bài 11 — Q-learning chính sách kinh tế thích nghi")
    n_ep=st.slider("Số episodes training",1000,10000,5000,500)
    if st.button("🤖 Chạy Q-learning"):
        with st.spinner(f"Training {n_ep} episodes..."):
            try:
                import gymnasium as gym
                from gymnasium import spaces
                class VNEnv(gym.Env):
                    def __init__(self):
                        super().__init__()
                        self.action_space=spaces.Discrete(5)
                        self.observation_space=spaces.MultiDiscrete([3,3,3,3])
                        self.T=10
                        self.alloc={0:np.array([.70,.10,.10,.10]),
                                    1:np.array([.40,.25,.15,.20]),
                                    2:np.array([.25,.45,.15,.15]),
                                    3:np.array([.20,.20,.45,.15]),
                                    4:np.array([.30,.20,.10,.40])}
                        self.w=np.array([.40,.25,.20,.15])
                    def reset(self,seed=None,options=None):
                        super().reset(seed=seed)
                        self.K=27500.; self.D=20.3; self.AI=86.; self.H=30.; self.A=35.; self.t=0
                        self.Yp=self.A*self.K**.33*53.9**.42*self.D**.10*self.AI**.08*self.H**.07
                        return self._st(),{}
                    def _st(self):
                        Y=self.A*self.K**.33*53.9**.42*self.D**.10*self.AI**.08*self.H**.07
                        g=(Y-self.Yp)/(self.Yp+1e-6)*100
                        return np.array([0 if g<5 else(1 if g<8 else 2),
                                         0 if self.D<18 else(1 if self.D<25 else 2),
                                         0 if self.AI<80 else(1 if self.AI<120 else 2),
                                         2 if g<4 else(1 if g<7 else 0)],dtype=int)
                    def step(self,action):
                        a=self.alloc[action]; b=1200.
                        Yb=self.A*self.K**.33*53.9**.42*self.D**.10*self.AI**.08*self.H**.07
                        self.K=(1-.05)*self.K+a[0]*b
                        self.D=min((1-.12)*self.D+a[1]*b/500,50.)
                        self.AI=(1-.15)*self.AI+a[2]*b/15
                        self.H=float(np.clip(self.H+.8*a[3]*b/300-.02*self.H,5,65))
                        self.A=self.A*(1+.003*self.D+.002*self.AI+.004*self.H)
                        Ya=self.A*self.K**.33*53.9**.42*self.D**.10*self.AI**.08*self.H**.07
                        dg=(Ya-Yb)/(Yb+1e-6)
                        r=(self.w[0]*dg-self.w[1]*max(0,.05-dg)
                           -self.w[2]*max(0,a[2]*.3-a[3]*.1)
                           -self.w[3]*(a[0]+a[2])*.2)
                        self.Yp=Ya; self.t+=1; done=self.t>=self.T
                        return self._st(),r,done,False,{}
                env=VNEnv()
                Q=np.zeros((3,3,3,3,5)); rew_hist=[]
                for ep in range(n_ep):
                    s,_=env.reset(); ep_r=0
                    eps=max(.05,1.-ep/(n_ep*.5))
                    while True:
                        a=(env.action_space.sample() if np.random.rand()<eps
                           else int(np.argmax(Q[tuple(s)])))
                        s2,r,done,_,_=env.step(a)
                        Q[tuple(s)+(a,)]+=.1*(r+.95*Q[tuple(s2)].max()-Q[tuple(s)+(a,)])
                        ep_r+=r; s=s2
                        if done: break
                    rew_hist.append(ep_r)
                st.session_state["bai11_Q"]=Q
                st.session_state["bai11_rh"]=rew_hist
                st.success("✓ Training xong!")
            except Exception as e:
                st.error(f"Lỗi: {e}")
    if "bai11_rh" in st.session_state:
        rh=np.array(st.session_state["bai11_rh"])
        w=200; sm=np.convolve(rh,np.ones(w)/w,mode="valid")
        fig=go.Figure(go.Scatter(y=sm,mode="lines",line=dict(color="#e74c3c",width=2)))
        fig.update_layout(template="plotly_dark",height=340,
                          xaxis_title="Episode",yaxis_title="Avg Reward",
                          title=f"Learning Curve (smoothed window={w})")
        st.plotly_chart(fig,use_container_width=True)
        an=["Truyền thống","Cân bằng","Số hóa nhanh","AI dẫn dắt","Bao trùm"]
        Q=st.session_state["bai11_Q"]
        ac=np.bincount(Q.reshape(-1,5).argmax(1),minlength=5)
        fig2=go.Figure(go.Bar(x=an,y=ac/81*100,
            marker_color=["#e74c3c","#3498db","#f39c12","#9b59b6","#27ae60"],
            text=[f"{v:.0f}%" for v in ac/81*100],textposition="outside"))
        fig2.update_layout(template="plotly_dark",height=300,
                           yaxis_title="% trạng thái chọn hành động",
                           title="Phân phối hành động π*(s) trên 81 trạng thái")
        st.plotly_chart(fig2,use_container_width=True)

def page_bai12():
    section("Bài 12 — AIDEOM-VN: Đồ án tích hợp")
    with st.spinner("Đang chạy pipeline AIDEOM-VN (M1–M5)..."):
        d=run_bai12()
    kpi=d["kpi"]; s_keys=list(kpi.keys())
    s_names=[kpi[k]["name"] for k in s_keys]
    colors=[kpi[k]["col"] for k in s_keys]
    tab1,tab2,tab3,tab4=st.tabs([
        "📊 Tổng quan (M1-M2)",
        "💰 Phân bổ (M3)",
        "🗺️ 5 Kịch bản (M6)",
        "⚠️ Cảnh báo rủi ro (M4-M5)"
    ])
    with tab1:
        section("M1 — Dự báo kinh tế (Cobb-Douglas)")
        b1d=run_bai1()
        c1,c2,c3=st.columns(3)
        pcard("MAPE (Cobb-Douglas)",f"{b1d['MAPE']:.2f}%","",c1)
        pcard("Ā (TFP trung bình)",f"{b1d['Am']:.4f}","",c2)
        pcard("Y 2030 dự báo",f"{b1d['Y30']:,.0f} ng.tỷ","",c3)
        lbs=list(b1d["contrib"].keys()); vs=list(b1d["contrib"].values())
        fig=go.Figure(go.Bar(x=lbs,y=[v*100 for v in vs],
            marker_color=["#1abc9c","#3498db","#e74c3c","#f39c12","#9b59b6","#27ae60"],
            text=[f"{v*100:.2f}%" for v in vs],textposition="outside"))
        fig.update_layout(template="plotly_dark",height=340,
                          yaxis_title="Đóng góp_pct",
                          title="Phân rã đóng góp tăng trưởng 2020-2025")
        st.plotly_chart(fig,use_container_width=True)
        st.divider()
        section("M2 — Đánh giá sẵn sàng số (TOPSIS)")
        m2=d["m2"]
        df_m2=pd.DataFrame({"Vùng":m2["vung"],"C*":m2["C"].round(4),"Hạng":m2["rank"]})
        df_m2=df_m2.sort_values("Hạng").reset_index(drop=True)
        c1m,c2m=st.columns(2)
        with c1m: st.dataframe(df_m2,use_container_width=True,hide_index=True)
        with c2m:
            fig2=go.Figure(go.Bar(x=df_m2["Vùng"],y=df_m2["C*"],
                marker_color=["#27ae60" if r<=2 else "#f39c12" if r<=4 else "#e74c3c"
                              for r in df_m2["Hạng"]],
                text=[f"Hạng {r}" for r in df_m2["Hạng"]],textposition="outside"))
            fig2.update_layout(template="plotly_dark",height=300,
                               title="Xếp hạng Sẵn sàng số 6 vùng")
            st.plotly_chart(fig2,use_container_width=True)
    with tab2:
        section("M3 — LP tối ưu phân bổ ngân sách (S5 – Tối ưu LP)")
        c1,c2=st.columns([1,2])
        with c1:
            pcard("Z* (LP Optimal)",f"{d['Zstar']:,.0f} tỷ VND",
                  "50.000 tỷ / 6 vùng × 4 hạng mục",None)
            rn_f=["Trung du MN","ĐB sông Hồng","BT+DH Trung Bộ",
                  "Tây Nguyên","Đông Nam Bộ","ĐB sông CL"]
            it_f=["Hạ tầng","Số hóa","AI","Nhân lực"]
            df_a=pd.DataFrame(d["alloc"].round(0),
                              index=rn_f,columns=it_f)
            df_a["Tổng"]=df_a.sum(1)
            st.dataframe(df_a,use_container_width=True)
        with c2:
            fig=go.Figure(go.Heatmap(z=d["alloc"],x=it_f,y=rn_f,
                colorscale="YlOrRd",texttemplate="%{z:.0f}",
                colorbar=dict(title="tỷ VND")))
            fig.update_layout(template="plotly_dark",height=380,
                              title="Phân bổ LP tối ưu (tỷ VND)")
            st.plotly_chart(fig,use_container_width=True)
    with tab3:
        section("M6 — So sánh 5 kịch bản chính sách")
        years=[2026,2027,2028,2029,2030,2031]
        fig_t=go.Figure()
        for k,nm,col in zip(s_keys,s_names,colors):
            fig_t.add_trace(go.Scatter(
                x=years,y=d["gdp_paths"][k],name=nm,
                line=dict(color=col,width=2.5),mode="lines+markers"))
        fig_t.update_layout(template="plotly_dark",height=380,
                            xaxis_title="Năm",yaxis_title="GDP (ng.tỷ VND)",
                            title="Quỹ đạo GDP 2026–2030 – 5 kịch bản",
                            legend=dict(x=.01,y=.99))
        st.plotly_chart(fig_t,use_container_width=True)
        # Radar
        cats=["GDP 2030","CAGR","NetJob","Số hóa D","AI Capacity"]
        def _n(vs):
            mn,mx=min(vs),max(vs)
            return [.5 if mn==mx else (v-mn)/(mx-mn) for v in vs]
        g30=[kpi[k]["Y30"] for k in s_keys]; ca=[kpi[k]["CAGR"] for k in s_keys]
        nj=[kpi[k]["nj_tot"]/1000 for k in s_keys]
        d30=[kpi[k]["D30"] for k in s_keys]; ai30=[kpi[k]["AI30"] for k in s_keys]
        gn,cn,nn,dn,an=_n(g30),_n(ca),_n(nj),_n(d30),_n(ai30)
        fig_r=go.Figure()
        for idx,(k,nm,col) in enumerate(zip(s_keys,s_names,colors)):
            vs=[gn[idx],cn[idx],nn[idx],dn[idx],an[idx]]
            vs_c=vs+[vs[0]]; ct_c=cats+[cats[0]]
        fig_r.add_trace(go.Scatterpolar(r=vs_c,theta=ct_c,
                name=nm,fill="toself",line_color=col,opacity=.5))
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                            template="plotly_dark",height=420,
                            title="Radar – So sánh đa chiều 5 kịch bản")
        st.plotly_chart(fig_r,use_container_width=True)
        # KPI table
        rows2=[{"Kịch bản":kpi[k]["name"],
                "GDP 2030":round(kpi[k]["Y30"],1),
                "CAGR(%/năm)":round(kpi[k]["CAGR"],2),
                "D 2030(%)":round(kpi[k]["D30"],1),
                "NetJob(ng)":round(kpi[k]["nj_tot"]/1000,1),
                "Ngành+":kpi[k]["nj_pos"]} for k in s_keys]
        st.dataframe(pd.DataFrame(rows2),use_container_width=True,hide_index=True)
    with tab4:
        section("M4-M5 — Cảnh báo rủi ro & Lao động")
        f1v=[kpi[k]["f1"] for k in s_keys]; f4v=[kpi[k]["f4"] for k in s_keys]
        f3v=[kpi[k]["f3"] for k in s_keys]; f2v=[kpi[k]["f2"] for k in s_keys]
        col1,col2=st.columns(2)
        with col1:
            fig5=go.Figure(go.Scatter(
                x=f4v,y=f1v,mode="markers+text",text=s_names,
                textposition="top center",
                marker=dict(size=18,color=colors)))
            fig5.update_layout(template="plotly_dark",height=360,
                xaxis_title="Rủi ro an ninh (f₄)",
                yaxis_title="GDP gain (f₁, tỷ VND)",
                title="GDP gain vs Rủi ro an ninh mạng")
            st.plotly_chart(fig5,use_container_width=True)
        with col2:
            fig6=go.Figure(go.Scatter(
                x=f3v,y=f2v,mode="markers+text",text=s_names,
                textposition="top center",
                marker=dict(size=18,color=colors)))
            fig6.update_layout(template="plotly_dark",height=360,
                xaxis_title="Phát thải CO₂ (f₃)",
                yaxis_title="Bất bình đẳng vùng MAD (f₂)",
                title="Phát thải vs Bất bình đẳng vùng")
            st.plotly_chart(fig6,use_container_width=True)
        best_g=max(s_keys,key=lambda k:kpi[k]["Y30"])
        best_j=max(s_keys,key=lambda k:kpi[k]["nj_tot"])
        best_s=min(s_keys,key=lambda k:kpi[k]["f4"])
        best_e=min(s_keys,key=lambda k:kpi[k]["f3"])
        st.markdown(f"""
| Tiêu chí | Kịch bản tốt nhất |
|---|---|
|📈 GDP 2030 cao nhất|**{kpi[best_g]['name']}** ({kpi[best_g]['Y30']:,.1f} ng.tỷ)|
|👷 NetJob ròng cao nhất|**{kpi[best_j]['name']}** ({kpi[best_j]['nj_tot']:,.0f} việc)|
|🔒 Rủi ro AN mạng thấp nhất|**{kpi[best_s]['name']}**|
|🌱 Phát thải thấp nhất|**{kpi[best_e]['name']}**|
        """)
# ══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════
PAGES = {
    "🏠 Trang chủ"               : "home",
    "🌱 Bài 1 — Cobb-Douglas + AI": "b1",
    "💰 Bài 2 — LP ngân sách số"  : "b2",
    "🎯 Bài 3 — Priority 10 ngành": "b3",
    "🗺️ Bài 4 — LP ngành-vùng"   : "b4",
    "📋 Bài 5 — MIP 15 dự án"     : "b5",
    "🌐 Bài 6 — TOPSIS 6 vùng"   : "b6",
    "🔬 Bài 7 — NSGA-II Pareto"   : "b7",
    "⚙️ Bài 8 — Động 2026-2035"  : "b8",
    "👥 Bài 9 — Lao động & AI"    : "b9",
    "🎲 Bài 10 — Stochastic SP"   : "b10",
    "🤖 Bài 11 — Q-learning RL"   : "b11",
    "🇻🇳 Bài 12 — AIDEOM tích hợp": "b12",
}

with st.sidebar:
    st.markdown("### 🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.divider()
    sel=st.radio("",list(PAGES.keys()),label_visibility="collapsed")
    st.divider()
    st.markdown("##### 📂 Nguồn dữ liệu")
    st.caption("NSO, MoST, MIC, MPI, WB, GII 2025")

# ── ROUTING ──
route=PAGES[sel]
if   route=="home": page_home()
elif route=="b1":   page_bai1()
elif route=="b2":   page_bai2()
elif route=="b3":   page_bai3()
elif route=="b4":   page_bai4()
elif route=="b5":   page_bai5()
elif route=="b6":   page_bai6()
elif route=="b7":   page_bai7()
elif route=="b8":   page_bai8()
elif route=="b9":   page_bai9()
elif route=="b10":  page_bai10()
elif route=="b11":  page_bai11()
elif route=="b12":  page_bai12()
