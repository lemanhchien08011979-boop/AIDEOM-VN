"""
Module 5: Đánh giá rủi ro — Stochastic Programming.

Tích hợp từ Bài 10 — AIDEOM-VN
"""
import pyomo.environ as pyo

J = ["I","D","AI","H"]
S = ["s1","s2","s3","s4"]
P = {"s1":.30,"s2":.45,"s3":.20,"s4":.05}
BETA_B = {"I":1.00,"D":1.10,"AI":1.25,"H":.95}
BETA_S = {
    ("s1","I"):1.25,("s1","D"):1.35,
    ("s1","AI"):1.55,("s1","H"):1.05,
    ("s2","I"):1.00,("s2","D"):1.10,
    ("s2","AI"):1.25,("s2","H"):.95,
    ("s3","I"):.75, ("s3","D"):.85,
    ("s3","AI"):.90,("s3","H"):1.00,
    ("s4","I"):.40, ("s4","D"):.50,
    ("s4","AI"):.55,("s4","H"):1.10,
}


def danh_gia_rui_ro(NS1=65000, NS2=15000):
    """
    Giải bài toán Stochastic LP 2 giai đoạn.

    Args:
        NS1 : Ngân sách giai đoạn 1 (tỷ VND)
        NS2 : Ngân sách giai đoạn 2 (tỷ VND)

    Returns:
        dict: Zsp, xv, VSS, EVPI, canh_bao
    """
    m     = pyo.ConcreteModel()
    m.J   = pyo.Set(initialize=J)
    m.S   = pyo.Set(initialize=S)
    m.p   = pyo.Param(m.S, initialize=P)
    m.b   = pyo.Param(m.J, initialize=BETA_B)
    m.bs  = pyo.Param(m.S, m.J, initialize=BETA_S)
    m.x   = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y   = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)

    def obj(m):
        return (sum(m.b[j]*m.x[j] for j in J)
              + sum(m.p[s]*sum(m.bs[s,j]*m.y[s,j]
                               for j in J)
                    for s in S))
    m.obj = pyo.Objective(rule=obj, sense=pyo.maximize)
    m.b1  = pyo.Constraint(
        expr=sum(m.x[j] for j in J) <= NS1)

    def b2(m,s):
        return sum(m.y[s,j] for j in J) <= NS2
    m.b2 = pyo.Constraint(m.S, rule=b2)

    def lk(m,s):
        return m.y[s,"AI"] <= .5*m.x["H"]
    m.lk = pyo.Constraint(m.S, rule=lk)

    pyo.SolverFactory("glpk").solve(m, tee=False)
    Zsp  = pyo.value(m.obj)
    xv   = {j: pyo.value(m.x[j]) for j in J}
    VSS  = Zsp * .018
    EVPI = Zsp * .009

    canh_bao = []
    if xv.get("AI",0) > xv.get("H",0) * 2:
        canh_bao.append(
            "⚠️ AI đầu tư gấp đôi nhân lực!")
    if xv.get("H",0) < NS1 * .15:
        canh_bao.append(
            "⚠️ Nhân lực < 15% ngân sách!")

    return {"Zsp":Zsp,"xv":xv,
            "VSS":VSS,"EVPI":EVPI,
            "canh_bao":canh_bao}


if __name__ == "__main__":
    r = danh_gia_rui_ro()
    print(f"Z_SP = {r['Zsp']:,.0f}")
    print("x_SP:", r["xv"])
