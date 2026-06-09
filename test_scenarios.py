"""
Bộ test: Bảng so sánh S1, S3, S5 — yêu cầu (c).
Chạy: python test_scenarios.py
"""
import sys
sys.path.insert(0, "/content")
import pandas as pd
from m1_forecast import chay_5_kich_ban
from m3_optimize import lp_don_gian
from m4_labor    import mo_phong_lao_dong

def bang_so_sanh_S1_S3_S5():
    """
    Tạo bảng tổng hợp so sánh kịch bản S1, S3, S5.
    Yêu cầu (c) đề bài Bài 12.
    """
    paths = chay_5_kich_ban()

    ten = {
        "S1": "S1: Truyền thống\n(70K+10D+10AI+10H)",
        "S3": "S3: AI dẫn dắt\n(20K+20D+45AI+15H)",
        "S5": "S5: Tối ưu LP\n(Cân bằng AIDEOM)",
    }
    kb_ld = {
        "S1": 30000, "S3": 30000, "S5": 30000
    }

    rows = []
    for k in ["S1","S3","S5"]:
        path    = paths[k]
        gdp2026 = path[0]
        gdp2030 = path[-1]
        cagr    = (gdp2030/gdp2026)**(1/5) - 1
        usd2030 = gdp2030 * 1000 / 24500

        r_ld = mo_phong_lao_dong(kb_ld[k])
        netjob = r_ld["total_netjob"] if r_ld else 0

        r_lp   = lp_don_gian(100)

        rows.append({
            "Kịch bản"              : ten[k],
            "GDP 2026\n(ng.tỷ VND)" : f"{gdp2026:,.1f}",
            "GDP 2030\n(ng.tỷ VND)" : f"{gdp2030:,.1f}",
            "CAGR\n(%/năm)"         : f"{cagr*100:.2f}%",
            "GDP 2030\n(tỷ USD)"    : f"{usd2030:,.0f}",
            "LP Z*\n(ng.tỷ)"        : f"{r_lp['Z']:.2f}",
            "NetJob\n(việc làm)"    : f"{netjob:,.0f}",
            "Đánh giá"              : (
                "⭐⭐⭐ Tối ưu"
                if k=="S5" else
                "⭐⭐ Tăng trưởng cao"
                if k=="S3" else
                "⭐ Cơ sở"
            ),
        })

    df = pd.DataFrame(rows)
    print("\n" + "="*80)
    print("BẢNG SO SÁNH 3 KỊCH BẢN — GDP VIỆT NAM 2030")
    print("="*80)
    print(df.to_string(index=False))

    # Phân tích đánh đổi
    g = {k: paths[k][-1] for k in ["S1","S3","S5"]}
    print("\n" + "="*80)
    print("PHÂN TÍCH ĐÁNH ĐỔI (TRADE-OFF)")
    print("="*80)
    print(f"  S3 vs S1: {g['S3']-g['S1']:+,.1f} ng.tỷ VND")
    print(f"  S5 vs S1: {g['S5']-g['S1']:+,.1f} ng.tỷ VND")
    print(f"  S5 vs S3: {g['S5']-g['S3']:+,.1f} ng.tỷ VND")
    print(f"\n  📌 Kết luận: "
          f"{'S5' if g['S5']>=max(g.values()) else 'S3'}"
          f" cho GDP 2030 cao nhất")
    return df


if __name__ == "__main__":
    bang_so_sanh_S1_S3_S5()
