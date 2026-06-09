"""
Module 4: Mô phỏng lao động dưới tác động AI.

Tích hợp từ Bài 9 — AIDEOM-VN
"""
import numpy as np
import pandas as pd
import cvxpy as cp

SECTORS = ["Nông-Lâm-TS","CN Chế biến","Xây dựng",
           "Bán buôn-BL","Tài chính-NH","Logistics",
           "CNTT-TT","Giáo dục"]
N = 8

RISK = np.array([18,42,25,38,52,35,28,22]) / 100.0
A1   = np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
B1   = np.array([45.,28.,35.,32.,22.,30.,20.,55.])
C1   = np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
D1   = np.array([50.,32.,42.,38.,26.,36.,24.,62.])


def mo_phong_lao_dong(ngan_sach=30000):
    """
    Tối đa hóa tổng NetJob ròng (CVXPY LP).

    Args:
        ngan_sach : Ngân sách tổng (tỷ VND)

    Returns:
        dict: xA, xH, nj, total_netjob, df
    """
    xA = cp.Variable(N, nonneg=True)
    xH = cp.Variable(N, nonneg=True)
    NJ = (cp.multiply(A1,xA) + cp.multiply(B1,xH)
          - cp.multiply(C1*RISK, xA))
    cons = [
        cp.sum(xA+xH) <= ngan_sach,
        NJ >= 0,
        cp.multiply(C1*RISK,xA) <= cp.multiply(D1,xH),
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(NJ)), cons)
    prob.solve(solver=cp.GLPK)

    if prob.status not in ["optimal","optimal_inaccurate"]:
        return None

    nj_val = (A1*xA.value + B1*xH.value
              - C1*RISK*xA.value)
    df = pd.DataFrame({
        "Ngành"    : SECTORS,
        "x_AI(tỷ)" : xA.value.round(0),
        "x_H(tỷ)"  : xH.value.round(0),
        "NetJob"   : nj_val.round(0).astype(int),
    })
    return {
        "xA"          : xA.value,
        "xH"          : xH.value,
        "nj"          : nj_val,
        "total_netjob": nj_val.sum(),
        "df"          : df,
    }


if __name__ == "__main__":
    r = mo_phong_lao_dong(30000)
    print(f"Tổng NetJob = {r['total_netjob']:,.0f}")
    print(r["df"].to_string(index=False))
