"""
Unit Tests AIDEOM-VN — pytest test_modules.py -v
Yêu cầu (a) đề bài Bài 12
"""
import pytest
import numpy as np
import sys
sys.path.insert(0, "/content")


class TestM1:
    """Tests Module 1 — Cobb-Douglas"""

    def test_TFP_shape(self):
        from m1_forecast import tinh_TFP
        Y  = np.array([8044.4, 12847.6])
        K  = np.array([16500., 25900.])
        L  = np.array([53.6,   53.4  ])
        D  = np.array([12.0,   19.5  ])
        AI = np.array([55.6,   80.1  ])
        H  = np.array([24.1,   29.2  ])
        A  = tinh_TFP(Y, K, L, D, AI, H)
        assert A.shape == (2,)

    def test_TFP_positive(self):
        from m1_forecast import tinh_TFP
        Y  = np.array([8044.4, 12847.6])
        K  = np.array([16500., 25900.])
        L  = np.array([53.6,   53.4  ])
        D  = np.array([12.0,   19.5  ])
        AI = np.array([55.6,   80.1  ])
        H  = np.array([24.1,   29.2  ])
        A  = tinh_TFP(Y, K, L, D, AI, H)
        assert all(A > 0)

    def test_5_kich_ban_count(self):
        from m1_forecast import chay_5_kich_ban
        r = chay_5_kich_ban()
        assert len(r) == 5

    def test_S1_S3_S5_exist(self):
        from m1_forecast import chay_5_kich_ban
        r = chay_5_kich_ban()
        for k in ["S1","S3","S5"]:
            assert k in r, f"Thiếu kịch bản {k}"

    def test_GDP_increase(self):
        from m1_forecast import chay_5_kich_ban
        r = chay_5_kich_ban()
        for k,v in r.items():
            assert v[-1] > v[0], f"{k}: GDP không tăng"

    def test_MAPE_acceptable(self):
        from m1_forecast import du_bao_GDP_2030
        r = du_bao_GDP_2030()
        assert r["MAPE"] < 10, f"MAPE={r['MAPE']:.2f}% quá cao"


class TestM2:
    """Tests Module 2 — TOPSIS"""

    def test_output_6_rows(self):
        from m2_readiness import danh_gia_vung
        df = danh_gia_vung()
        assert len(df) == 6

    def test_C_star_in_01(self):
        from m2_readiness import danh_gia_vung
        df = danh_gia_vung()
        assert all(0 <= c <= 1
                   for c in df["C* Chuyên gia"])

    def test_rank_complete(self):
        from m2_readiness import danh_gia_vung
        df = danh_gia_vung()
        assert set(df["Hạng CG"]) == {1,2,3,4,5,6}

    def test_entropy_sum_1(self):
        from m2_readiness import entropy_weights, X_DEFAULT
        w = entropy_weights(X_DEFAULT)
        assert abs(w.sum() - 1.0) < 1e-6


class TestM3:
    """Tests Module 3 — LP Optimize"""

    def test_lp_don_gian_feasible(self):
        from m3_optimize import lp_don_gian
        r = lp_don_gian(100)
        assert r["Z"] > 0

    def test_lp_monotone(self):
        from m3_optimize import lp_don_gian
        z1 = lp_don_gian(100)["Z"]
        z2 = lp_don_gian(120)["Z"]
        assert z2 >= z1

    def test_lp_min_constraints(self):
        from m3_optimize import lp_don_gian
        r  = lp_don_gian(100)
        x  = r["x"]
        assert x[0] >= 25-0.01  # x1 >= 25
        assert x[1] >= 15-0.01  # x2 >= 15
        assert x[2] >= 20-0.01  # x3 >= 20
        assert x[3] >= 10-0.01  # x4 >= 10

    def test_lp_vung_feasible(self):
        from m3_optimize import lp_nganh_vung
        r = lp_nganh_vung(50000)
        assert r["Z"] > 0


class TestM4:
    """Tests Module 4 — Labor"""

    def test_feasible(self):
        from m4_labor import mo_phong_lao_dong
        r = mo_phong_lao_dong(30000)
        assert r is not None

    def test_positive_netjob(self):
        from m4_labor import mo_phong_lao_dong
        r = mo_phong_lao_dong(30000)
        assert r["total_netjob"] > 0

    def test_budget_not_exceeded(self):
        from m4_labor import mo_phong_lao_dong
        NS = 30000
        r  = mo_phong_lao_dong(NS)
        assert r["xA"].sum() + r["xH"].sum() <= NS+1


class TestM5:
    """Tests Module 5 — Stochastic SP"""

    def test_feasible(self):
        from m5_risk import danh_gia_rui_ro
        r = danh_gia_rui_ro()
        assert r["Zsp"] > 0

    def test_first_stage_budget(self):
        from m5_risk import danh_gia_rui_ro
        r    = danh_gia_rui_ro(65000)
        tong = sum(r["xv"].values())
        assert tong <= 65000 + 1

    def test_VSS_positive(self):
        from m5_risk import danh_gia_rui_ro
        r = danh_gia_rui_ro()
        assert r["VSS"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
