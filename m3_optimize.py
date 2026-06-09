"""
Module 3: Tối ưu LP phân bổ ngân sách.

Tích hợp từ Bài 2 (LP đơn giản) và Bài 4 (LP ngành-vùng).
"""
import numpy as np
import pulp

REGIONS = ["NMM","RRD","NCC","CH","SE","MD"]
ITEMS   = ["I","D","AI","H"]
BETA    = {
    ("NMM","I"):1.15,("NMM","D"):.85,("NMM","AI"):.55,("NMM","H"):1.30,
    ("RRD","I"):.95, ("RRD","D"):1.25,("RRD","AI"):1.40,("RRD","H"):1.05,
    ("NCC","I"):1.05,("NCC","D"):.95, ("NCC","AI"):.85,("NCC","H"):1.15,
    ("CH","I") :1.20,("CH","D") :.75, ("CH","AI") :.45,("CH","H") :1.35,
    ("SE","I") :.90, ("SE","D") :1.30,("SE","AI"):1.55,("SE","H") :1.00,
    ("MD","I") :1.10,("MD","D") :.85, ("MD","AI") :.65,("MD","H") :1.25,
}
D0 = {"NMM":38,"RRD":78,"NCC":55,"CH":32,"SE":82,"MD":48}


def lp_don_gian(ngan_sach=100):
    """
    Giải LP 4 hạng mục đơn giản (Bài 2).

    Args:
        ngan_sach : Tổng ngân sách (nghìn tỷ VND)

    Returns:
        dict: Z, x (phân bổ), shadow_price
    """
    m  = pulp.LpProblem("LP_B2", pulp.LpMaximize)
    x1 = pulp.LpVariable("x1", lowBound=0)
    x2 = pulp.LpVariable("x2", lowBound=0)
    x3 = pulp.LpVariable("x3", lowBound=0)
    x4 = pulp.LpVariable("x4", lowBound=0)
    m += .85*x1 + 1.20*x2 + .95*x3 + 1.35*x4
    cb = x1+x2+x3+x4 <= ngan_sach
    m += cb, "budget"
    m += x1 >= 25; m += x2 >= 15
    m += x3 >= 20; m += x4 >= 10
    m += x2+x4 >= .35*(x1+x2+x3+x4)
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    sp = (m.constraints["budget"].pi
          if hasattr(m.constraints["budget"],"pi")
          else 1.35)
    return {
        "Z" : pulp.value(m.objective),
        "x" : [x1.value(),x2.value(),
               x3.value(),x4.value()],
        "sp": sp,
    }


def lp_nganh_vung(ngan_sach=50000, co_cong_bang=True):
    """
    Giải LP ngành-vùng 6×4 (Bài 4).

    Args:
        ngan_sach    : Tổng ngân sách (tỷ VND)
        co_cong_bang : Có ràng buộc công bằng C5 không

    Returns:
        dict: Z, alloc (6×4 ndarray)
    """
    m = pulp.LpProblem("LP_B4", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x",(REGIONS,ITEMS),lowBound=0)
    M = pulp.LpVariable("Dm", lowBound=0)
    m += pulp.lpSum(BETA[(r,j)]*x[r][j]
                    for r in REGIONS for j in ITEMS)
    m += pulp.lpSum(x[r][j]
                    for r in REGIONS for j in ITEMS) <= ngan_sach
    for r in REGIONS:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000
    m += pulp.lpSum(x[r]["H"] for r in REGIONS) >= 12000
    if co_cong_bang:
        for r in REGIONS:
            m += D0[r] + .002*x[r]["D"] <= M
            m += D0[r] + .002*x[r]["D"] >= .7*M
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    alloc = np.array([[x[r][j].value()
                       for j in ITEMS]
                      for r in REGIONS])
    return {"Z":pulp.value(m.objective),"alloc":alloc}


if __name__ == "__main__":
    r1 = lp_don_gian(100)
    print(f"LP đơn giản Z* = {r1['Z']:.2f}")
    r2 = lp_nganh_vung(50000)
    print(f"LP ngành-vùng Z* = {r2['Z']:,.0f}")
