# 🇻🇳 AIDEOM-VN
## AI-Driven Decision Optimization Model for Vietnam

> Web app giải 12 bài toán mô hình ra quyết định
> phát triển kinh tế Việt Nam trong kỉ nguyên AI
> — dữ liệu thực 2020-2025.

## Cấu trúc

| File | Mô tả |
|------|-------|
| `m1_forecast.py`   | Module 1: Cobb-Douglas + TFP + 5 kịch bản |
| `m2_readiness.py`  | Module 2: TOPSIS + Entropy 6 vùng |
| `m3_optimize.py`   | Module 3: LP đơn giản + LP ngành-vùng |
| `m4_labor.py`      | Module 4: NetJob tối ưu (CVXPY) |
| `m5_risk.py`       | Module 5: Stochastic SP (Pyomo) |
| `app.py`           | Module 6: Dashboard Streamlit 4 tab |
| `test_modules.py`  | Unit tests pytest (M1-M5) |
| `test_scenarios.py`| Bảng so sánh S1, S3, S5 |

## Cài đặt

```bash
pip install -r requirements.txt
apt-get install -y glpk-utils
```

## Chạy Dashboard

```bash
streamlit run app.py
```

## Chạy Tests

```bash
# Unit tests
pytest test_modules.py -v

# Bảng so sánh kịch bản
python test_scenarios.py
```

## 5 Kịch bản chính sách

| Mã | Tên | Phân bổ |
|----|-----|---------|
| S1 | Truyền thống  | 70K+10D+10AI+10H |
| S2 | Số hóa nhanh  | 25K+45D+15AI+15H |
| S3 | AI dẫn dắt    | 20K+20D+45AI+15H |
| S4 | Bao trùm số   | 30K+20D+10AI+40H |
| S5 | Tối ưu LP     | Cân bằng AIDEOM-VN |

## Dữ liệu: GSO, World Bank, MoST 2025
