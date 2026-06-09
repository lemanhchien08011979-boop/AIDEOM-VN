"""
Module 1: Dự báo GDP Cobb-Douglas mở rộng.

Tích hợp từ Bài 1 — AIDEOM-VN
Nguồn dữ liệu: GSO/NSO Việt Nam 2020-2025
"""
import numpy as np
import pandas as pd

ALPHA, BETA_L, GAMMA_D, DELTA_AI, THETA_H = .33, .42, .10, .08, .07

YEARS  = [2020,2021,2022,2023,2024,2025]
Y_GDP  = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
K_ARR  = np.array([16500,17800,19600,21300,23500,25900], float)
L_ARR  = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D_ARR  = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
AI_ARR = np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H_ARR  = np.array([24.1,26.1,26.2,27.0,28.4,29.2])


def tinh_TFP(Y, K, L, D, AI, H):
    """
    Tính TFP bằng cách giải ngược hàm Cobb-Douglas.

    Args:
        Y  : GDP (nghìn tỷ VND)
        K  : Vốn tích lũy (nghìn tỷ VND)
        L  : Lao động (triệu người)
        D  : Kinh tế số/GDP (%)
        AI : Số DN công nghệ số (nghìn)
        H  : Lao động qua đào tạo (%)

    Returns:
        A  : TFP từng năm (ndarray)
    """
    return Y / (K**ALPHA * L**BETA_L * D**GAMMA_D
                * AI**DELTA_AI * H**THETA_H)


def du_bao_GDP_2030():
    """
    Dự báo GDP năm 2030 theo kịch bản mở rộng.

    Giả định: K,L tăng 6%/năm, TFP tăng 1.2%/năm,
    D→30%, AI→100k, H→35%.

    Returns:
        dict: Y30, A, Am, MAPE, Yp, contrib
    """
    A    = tinh_TFP(Y_GDP, K_ARR, L_ARR,
                    D_ARR, AI_ARR, H_ARR)
    Am   = A.mean()
    Yp   = Am * (K_ARR**ALPHA * L_ARR**BETA_L
                 * D_ARR**GAMMA_D * AI_ARR**DELTA_AI
                 * H_ARR**THETA_H)
    MAPE = np.mean(np.abs((Y_GDP-Yp)/Y_GDP)) * 100

    gY  = np.diff(np.log(Y_GDP))
    gK  = np.diff(np.log(K_ARR))
    gL  = np.diff(np.log(L_ARR))
    gD  = np.diff(np.log(D_ARR))
    gAI = np.diff(np.log(AI_ARR))
    gH  = np.diff(np.log(H_ARR))
    gA  = np.diff(np.log(A))

    contrib = {
        "TFP (A)"      : gA.mean(),
        "Vốn (K)"      : (ALPHA   * gK).mean(),
        "Lao động (L)" : (BETA_L  * gL).mean(),
        "Số hóa (D)"   : (GAMMA_D * gD).mean(),
        "AI"           : (DELTA_AI* gAI).mean(),
        "Nhân lực (H)" : (THETA_H * gH).mean(),
    }

    K30 = 25900  * (1.06**5)
    L30 = 53.4   * (1.06**5)
    A30 = A[-1]  * (1.012**5)
    Y30 = A30 * (K30**ALPHA * L30**BETA_L
                 * 30.0**GAMMA_D * 100.0**DELTA_AI
                 * 35.0**THETA_H)

    return {"A":A,"Am":Am,"MAPE":MAPE,
            "Yp":Yp,"contrib":contrib,"Y30":Y30}


def chay_5_kich_ban():
    """
    Mô phỏng GDP 2026-2030 theo 5 kịch bản chính sách.

    Returns:
        dict: {ten_kich_ban: list GDP 6 điểm}
    """
    BUDGET = 1200.
    K0,L0,D0,AI0,H0 = 27500.,53.9,20.3,86.,30.
    Y25  = 12847.6
    A0   = (Y25/(25900**.33*53.4**.42*19.5**.10
                 *80.1**.08*29.2**.07)) * 1.012

    kich_ban = {
        "S1":{"name":"Truyền thống",
              "r":{"K":.70,"D":.10,"AI":.10,"H":.10}},
        "S2":{"name":"Số hóa nhanh",
              "r":{"K":.25,"D":.45,"AI":.15,"H":.15}},
        "S3":{"name":"AI dẫn dắt",
              "r":{"K":.20,"D":.20,"AI":.45,"H":.15}},
        "S4":{"name":"Bao trùm số",
              "r":{"K":.30,"D":.20,"AI":.10,"H":.40}},
        "S5":{"name":"Tối ưu LP",
              "r":{"K":.25,"D":.25,"AI":.25,"H":.25}},
    }

    gdp_paths = {}
    for sk, si in kich_ban.items():
        r   = si["r"]
        K,L,D,AI,H,A = K0,L0,D0,AI0,H0,A0
        path = []
        for _ in range(5):
            Y = A*K**.33*L**.42*D**.10*AI**.08*H**.07
            path.append(round(Y,1))
            K  = (1-.05)*K  + r["K"]*BUDGET
            D  = min((1-.12)*D  + r["D"]*BUDGET/500, 50.)
            AI = (1-.15)*AI + r["AI"]*BUDGET/15
            H  = float(np.clip(
                H + .8*r["H"]*BUDGET/300 - .02*H,5,65))
            A  = A*(1+.003*D+.002*AI+.004*H)
        Yf = A*K**.33*L**.42*D**.10*AI**.08*H**.07
        path.append(round(Yf,1))
        gdp_paths[sk] = path

    return gdp_paths


if __name__ == "__main__":
    r = du_bao_GDP_2030()
    print(f"MAPE = {r['MAPE']:.2f}%")
    print(f"Y2030 = {r['Y30']:,.0f} ng.tỷ VND")
    paths = chay_5_kich_ban()
    for k,v in paths.items():
        print(f"  {k}: GDP2030 = {v[-1]:,.1f}")
