"""
Module 2: Đánh giá sẵn sàng số 6 vùng (TOPSIS + Entropy).

Tích hợp từ Bài 6 — AIDEOM-VN
"""
import numpy as np
import pandas as pd

CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]
IS_BENEFIT = [True,True,True,True,True,True,True,False]
W_EXPERT   = np.array([.10,.10,.15,.20,.15,.15,.05,.10])

VUNG_VI = [
    "Trung du MN phía Bắc","Đồng bằng sông Hồng",
    "Bắc Trung Bộ+DH TB","Tây Nguyên",
    "Đông Nam Bộ","ĐB sông Cửu Long"
]

X_DEFAULT = np.array([
    [57.0,3.5,38,22,21.5,.18,72,.405],
    [152.3,20.0,78,68,36.8,.85,92,.358],
    [87.5,8.2,55,40,27.5,.32,84,.372],
    [68.9,.8,32,18,18.2,.15,68,.412],
    [158.9,18.5,82,75,42.5,.78,94,.385],
    [80.5,2.1,48,30,16.8,.22,78,.392],
])


def topsis(X, weights, is_benefit):
    """
    Cài đặt TOPSIS 5 bước từ đầu bằng numpy.

    Args:
        X          : Ma trận quyết định (n_alt × n_crit)
        weights    : Trọng số (tổng = 1)
        is_benefit : list[bool]

    Returns:
        C_star : Hệ số gần gũi (ndarray)
        rank   : Thứ hạng (1 = tốt nhất)
    """
    w  = np.array(weights)
    R  = X / np.sqrt((X**2).sum(0) + 1e-12)
    V  = R * w
    As = np.where(is_benefit, V.max(0), V.min(0))
    An = np.where(is_benefit, V.min(0), V.max(0))
    Ss = np.sqrt(((V-As)**2).sum(1))
    Sn = np.sqrt(((V-An)**2).sum(1))
    C  = Sn / (Ss + Sn + 1e-12)
    rank = pd.Series(C).rank(ascending=False).astype(int).values
    return C, rank


def entropy_weights(X):
    """
    Tính trọng số Entropy khách quan.

    Args:
        X : Ma trận dữ liệu (n × m)

    Returns:
        w : Trọng số Entropy (ndarray, tổng=1)
    """
    P = X / X.sum(0)
    k = 1.0 / np.log(len(X))
    E = -k * np.nansum(P * np.log(P + 1e-12), 0)
    d = 1 - E
    return d / d.sum()


def danh_gia_vung(X=None):
    """
    Đánh giá 6 vùng bằng TOPSIS chuyên gia + Entropy.

    Args:
        X : Ma trận 6×8 (dùng mặc định nếu None)

    Returns:
        DataFrame xếp hạng
    """
    if X is None:
        X = X_DEFAULT
    C_exp,  rank_exp  = topsis(X, W_EXPERT,  IS_BENEFIT)
    w_ent = entropy_weights(X)
    C_ent,  rank_ent  = topsis(X, w_ent,     IS_BENEFIT)

    df = pd.DataFrame({
        "Vùng"         : VUNG_VI,
        "C* Chuyên gia": C_exp.round(4),
        "Hạng CG"      : rank_exp,
        "C* Entropy"   : C_ent.round(4),
        "Hạng EN"      : rank_ent,
    })
    return df.sort_values("Hạng CG").reset_index(drop=True)


if __name__ == "__main__":
    df = danh_gia_vung()
    print(df.to_string(index=False))
