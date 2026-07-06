import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from pathlib import Path
import torch
import torch.nn as nn
import implicit

# ── BPRMF 클래스 정의 ───────────────────────
class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, embed_dim):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, embed_dim)
        self.item_emb  = nn.Embedding(n_items, embed_dim)
        self.item_bias = nn.Embedding(n_items, 1)

    @torch.no_grad()
    def recommend(self, user_idx, user_items, N=50,
                  filter_already_liked_items=True, items=None):
        u = torch.tensor([user_idx])
        eu = self.user_emb(u)
        scores = (eu @ self.item_emb.weight.T +
                  self.item_bias.weight.squeeze(-1)).squeeze(0)
        if items is not None:
            mask = torch.zeros(scores.shape[0], dtype=torch.bool)
            mask[torch.tensor(items)] = True
            scores[~mask] = -1e9
        if filter_already_liked_items:
            liked = user_items.indices
            scores[liked] = -1e9
        topk = torch.topk(scores, N)
        return topk.indices.numpy(), topk.values.numpy()

# ── CPU Unpickler ────────────────────────────
class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'BPRMF':
            return BPRMF
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b),
                                        map_location='cpu',
                                        weights_only=False)
        return super().find_class(module, name)

# ── 경로 설정 ────────────────────────────────
DATA_DIR = Path("data")
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 로드 ──────────────────────────────
@st.cache_resource
def load_all():
    with open(DATA_DIR / "model_data.pkl", "rb") as f:
        md = CPUUnpickler(f).load()

    with open(DATA_DIR / "dml_results.pkl", "rb") as f:
        dml = pickle.load(f)

    step3    = pd.read_csv(DATA_DIR / "step3_results.csv")
    item_daily = pd.read_csv(DATA_DIR / "item_daily_features.csv")
    item_info  = (item_daily.sort_values("date")
                            .groupby("video_id").last()[
                                ["video_tag_name", "like_cnt"]
                            ].reset_index())

    big = pd.read_csv(DATA_DIR / "big_matrix.csv",
                      usecols=["user_id", "video_id", "watch_ratio"])

    # exposure_raw → item_id 기준 dict로 변환
    er_raw = md["exposure_raw"]
    ic     = md["item_code2id"]
    if not isinstance(er_raw, dict):
        exp_raw = {int(ic[code]): int(er_raw.iloc[code])
                   for code in range(len(er_raw))
                   if code in ic}
    else:
        exp_raw = er_raw
    md["exposure_raw"] = exp_raw

    # propensity_big → alpha=0.5 Series를 item_id 기준 dict로 변환
    pb = md["propensity_big"]
    if isinstance(pb, dict) and not all(isinstance(v, (int, float)) for v in pb.values()):
        # {alpha: array} 구조
        alpha_key = 0.5 if 0.5 in pb else list(pb.keys())[1]
        p_arr = pb[alpha_key]
        md["propensity_big"] = {int(ic[code]): float(p_arr[code])
                                 for code in range(len(p_arr)) if code in ic}
    elif not isinstance(pb, dict):
        md["propensity_big"] = {int(ic[code]): float(pb.iloc[code])
                                 for code in range(len(pb)) if code in ic}

    return md, dml, step3, item_info, big

md, dml, step3, item_info, big = load_all()

# ── 변수 꺼내기 ──────────────────────────────
model      = md["model_baseline"]
prop       = md["propensity_big"]
exp_raw    = md["exposure_raw"]
id2code    = md["user_id2code"]
code2id    = md["item_code2id"]
candidates = md["candidate_item_codes"]
n_users    = md["n_users"]
n_items    = md["n_items"]

# train_ui 재생성
big_clean = big.dropna(subset=["user_id","video_id"]).reset_index(drop=True)
user_cat  = pd.Categorical(big_clean["user_id"])
item_cat  = pd.Categorical(big_clean["video_id"])
train_ui  = sparse.csr_matrix(
    (np.ones(len(big_clean)), (user_cat.codes, item_cat.codes)),
    shape=(len(user_cat.categories), len(item_cat.categories))
)

# ── 추천 함수 ────────────────────────────────
def get_recommendations(user_idx, lam, N=10, pool=50):
    rec_ids, rec_scores = model.recommend(
        user_idx, train_ui[user_idx],
        N=pool, filter_already_liked_items=True,
        items=candidates
    )
    reranked = []
    for code, score in zip(rec_ids, rec_scores):
        item_id = int(code2id[int(code)])
        p = prop.get(item_id, 1.0)
        ips_score = score * (1 - lam) + (score / max(p, 0.01)) * lam
        reranked.append((item_id, score, p, ips_score))
    reranked.sort(key=lambda x: x[3], reverse=True)
    return reranked[:N]

def get_item_info(item_id):
    row = item_info[item_info["video_id"] == item_id]
    if len(row) == 0:
        return "알 수 없음", 0
    tag  = row["video_tag_name"].values[0]
    like = row["like_cnt"].values[0]
    return (tag if pd.notna(tag) else "태그 없음",
            int(like) if pd.notna(like) else 0)

# ── 페이지 설정 ──────────────────────────────
st.set_page_config(page_title="Causal Debiasing for RecSys", layout="wide")
st.set_page_config(page_title="Causal Debiasing for RecSys", layout="wide")

# ── 커스텀 스타일 ─────────────────────────────
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #F8F9FA;
        color: #1A1A2E;
    }

    /* 메인 컨텐츠 영역 */
    .block-container {
        background-color: #F8F9FA;
        padding-top: 2rem;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        color: #666;
        font-weight: 500;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90D9;
        color: white !important;
    }

    /* 카드 스타일 (dataframe, metric) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 4px solid #4A90D9;
    }

    /* 제목 색상 */
    h1 { color: #1A1A2E; font-weight: 700; }
    h2 { color: #2C3E50; font-weight: 600; }
    h3 { color: #34495E; font-weight: 600; }

    /* 사이드 컬럼 배경 */
    section[data-testid="column"] > div {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
    }

    /* 슬라이더 색상 */
    .stSlider [data-baseweb="slider"] {
        color: #4A90D9;
    }

    /* info 박스 */
    .stAlert {
        background-color: #EBF4FF;
        border: 1px solid #4A90D9;
        border-radius: 8px;
        color: #1A1A2E;
    }

    /* 구분선 */
    hr {
        border-color: #E0E0E0;
    }
</style>
""", unsafe_allow_html=True)
st.title("Causal Debiasing for Recommender Systems")
st.markdown("추천 로그의 인기 편향을 진단하고 인과추론으로 교정합니다.")

tab1, tab2, tab3 = st.tabs([
    "IPS 편향 보정",
    "편향 시각화",
    "DML 분석"
])

# ══════════════════════════════════════════════
# 탭 1 — 추천 데모
# ══════════════════════════════════════════════
with tab1:
    st.markdown("**IPS Re-ranking**으로 인기 편향을 보정한 추천 결과를 비교합니다.")

    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.subheader("⚙️ 설정")
        user_list     = list(id2code.keys())
        selected_user = st.selectbox("유저 ID 선택", user_list)
        lam = st.slider("λ (IPS 보정 강도)", 0.0, 1.0, 0.0, 0.1)
        st.markdown("""
**λ 값 해석:**
- λ=0.0 → Baseline (편향 보정 없음)
- λ=0.5 → 중간 수준 보정
- λ=1.0 → 최대 IPS 보정
""")

    user_idx = id2code[selected_user]

    with col_main:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Baseline (λ=0)")
            recs_base = get_recommendations(user_idx, lam=0.0)
            rows = []
            for rank, (item_id, score, p, _) in enumerate(recs_base, 1):
                tag, like = get_item_info(item_id)
                pop = exp_raw.get(item_id, 0)
                rows.append({"순위": rank, "video_id": item_id,
                              "태그": tag, "인기도": int(pop), "좋아요": like})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with col2:
            st.subheader(f"IPS Re-ranking (λ={lam})")
            recs_ips = get_recommendations(user_idx, lam=lam)
            rows = []
            for rank, (item_id, score, p, _) in enumerate(recs_ips, 1):
                tag, like = get_item_info(item_id)
                pop = exp_raw.get(item_id, 0)
                rows.append({"순위": rank, "video_id": item_id,
                              "태그": tag, "인기도": int(pop), "좋아요": like})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()
        st.subheader("추천 아이템 인기도 비교")

        pop_base = [exp_raw.get(r[0], 0) for r in recs_base]
        pop_ips  = [exp_raw.get(r[0], 0) for r in recs_ips]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(range(1, 11), pop_base, color="steelblue", alpha=0.8)
        axes[0].set_title("Baseline Top-10 popularity")
        axes[0].set_xlabel("rank"); axes[0].set_ylabel("interaction count")

        axes[1].bar(range(1, 11), pop_ips, color="coral", alpha=0.8)
        axes[1].set_title(f"IPS (λ={lam}) Top-10 popularity")
        axes[1].set_xlabel("rank"); axes[1].set_ylabel("interaction count")

        plt.tight_layout()
        st.pyplot(fig)

        st.divider()
        c1, c2, c3 = st.columns(3)
        avg_base = np.mean(pop_base) if pop_base else 0
        avg_ips  = np.mean(pop_ips)  if pop_ips  else 0
        c1.metric("Baseline 평균 인기도", f"{avg_base:.1f}")
        c2.metric("IPS 평균 인기도",     f"{avg_ips:.1f}")
        if avg_base > 0:
            c3.metric("인기도 감소",
                      f"{avg_base - avg_ips:.1f}",
                      delta=f"-{(avg_base-avg_ips)/avg_base*100:.1f}%")

# ══════════════════════════════════════════════
# 탭 2 — Bias Dashboard
# ══════════════════════════════════════════════
with tab2:
    st.markdown("**KuaiRec big_matrix**의 인기 편향을 시각화합니다.")

    item_pop = big.groupby("video_id").size().sort_values(ascending=False)
    g_val    = item_pop.values.astype(float)
    g_sorted = np.sort(g_val)
    n        = len(g_sorted)
    cum      = np.cumsum(g_sorted) / g_sorted.sum()
    x        = np.arange(1, n+1) / n
    gini     = float((2*np.sum(np.arange(1,n+1)*g_sorted) -
                      (n+1)*g_sorted.sum()) / (n*g_sorted.sum()))

    col1, col2, col3 = st.columns(3)
    col1.metric("Gini 계수", f"{gini:.4f}", help="0=완전평등, 1=완전불평등")
    top10 = item_pop.head(int(n*0.1)).sum() / item_pop.sum()
    col2.metric("상위 10% 아이템 노출 점유율", f"{top10*100:.1f}%")
    col3.metric("전체 아이템 수", f"{n:,}개")

    st.divider()

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Lorenz curve
    axes2[0].plot(x, cum, color="crimson", lw=2, label=f"Lorenz (Gini={gini:.3f})")
    axes2[0].plot([0,1],[0,1], "k--", alpha=0.5, label="Perfect equality")
    axes2[0].fill_between(x, x, cum, alpha=0.15, color="crimson")
    axes2[0].set_xlabel("Cumulative share of items")
    axes2[0].set_ylabel("Cumulative share of exposure")
    axes2[0].set_title("Lorenz Curve")
    axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

    # Long-tail
    axes2[1].plot(np.arange(1, n+1), item_pop.values,
                  color="steelblue", lw=1.2)
    axes2[1].set_xscale("log"); axes2[1].set_yscale("log")
    axes2[1].set_xlabel("Item rank (log)")
    axes2[1].set_ylabel("Exposure count (log)")
    axes2[1].set_title("Long-tail Distribution")
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig2)

    st.divider()
    st.subheader("α sweep — 정확도 vs 다양성 Trade-off")

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

    axes3[0].plot(step3["alpha"], step3["recall@20"],
                  "o-", color="steelblue", lw=2, ms=8, label="Recall@20")
    axes3[0].plot(step3["alpha"], step3["recall@50"],
                  "o-", color="green", lw=2, ms=8, label="Recall@50")
    axes3[0].set_xlabel("α (IPS strength)")
    axes3[0].set_ylabel("Recall")
    axes3[0].set_title("Accuracy vs α")
    axes3[0].legend(); axes3[0].grid(True, alpha=0.3)

    axes3[1].plot(step3["alpha"], step3["coverage@20"],
                  "o-", color="orange", lw=2, ms=8)
    axes3[1].set_xlabel("α (IPS strength)")
    axes3[1].set_ylabel("Coverage@20")
    axes3[1].set_title("Coverage vs α")
    axes3[1].grid(True, alpha=0.3)

    axes3[2].plot(step3["coverage@20"], step3["recall@20"],
                  "o-", color="purple", lw=2, ms=10)
    for _, row in step3.iterrows():
        axes3[2].annotate(f"α={row['alpha']}",
                          (row["coverage@20"], row["recall@20"]),
                          textcoords="offset points", xytext=(6, 4), fontsize=9)
    axes3[2].set_xlabel("Coverage@20")
    axes3[2].set_ylabel("Recall@20")
    axes3[2].set_title("Trade-off Curve")
    axes3[2].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)

# ══════════════════════════════════════════════
# 탭 3 — DML Analysis
# ══════════════════════════════════════════════
with tab3:
    st.markdown("**Double Machine Learning**으로 인기도의 순수 인과효과를 추정합니다.")

    theta_dml   = 0.0021
    theta_naive = -0.0649        # 저장된 값 없음 → 하드코딩
    ci_dml      = dml["conf_int"]
    ci_naive    = [-0.0667, -0.0631]  # 하드코딩
    pval_dml    = 0.349
    pval_naive  = 0.0000         # 하드코딩
    r2_T        = 0.067          # 하드코딩
    r2_Y        = 0.431          # 하드코딩
    T_resid     = dml["T_resid"]
    Y_resid     = dml["Y_resid"]

    st.subheader("인과효과 θ 추정 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("Naive θ", f"{theta_naive:.4f}",
              help="X 통제 없이 단순 회귀")
    c2.metric("DML θ", f"{theta_dml:.4f}",
              help="X 통제 후 deconfounded estimate")
    c3.metric("p-value (DML)", f"{pval_dml:.3f}",
              delta="유의하지 않음" if pval_dml > 0.05 else "유의함")

    st.info(f"""
**해석:** X(유저·아이템 특성)를 통제하면 인기도의 효과가
**{theta_naive:.4f} → {theta_dml:.4f}** 로 수렴합니다.
음의 상관관계의 **103%** 가 confounding으로 설명됩니다.

**"인기도의 deconfounded partial effect는 0과 구분되지 않는다"** (p={pval_dml:.3f} > 0.05)
""")

    st.divider()

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))

    # Naive vs DML 비교
    methods = ["Naive\n(no control)", "DML\n(controlled)"]
    thetas  = [theta_naive, theta_dml]
    colors  = ["#E8593C", "#3B8BD4"]
    bars = axes4[0].bar(methods, thetas, color=colors, alpha=0.8, width=0.4)
    axes4[0].axhline(0, color="gray", lw=1, linestyle="--")
    for bar, val in zip(bars, thetas):
        axes4[0].text(bar.get_x() + bar.get_width()/2,
                      val + (0.001 if val > 0 else -0.003),
                      f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    axes4[0].set_ylabel("θ (popularity → watch_ratio)")
    axes4[0].set_title("Naive vs DML — confounding 크기")
    axes4[0].grid(True, alpha=0.3, axis="y")

    # 잔차 산점도
    sample_idx = np.random.choice(len(T_resid), min(5000, len(T_resid)), replace=False)
    axes4[1].scatter(T_resid[sample_idx], Y_resid[sample_idx],
                     alpha=0.2, s=5, color="purple")
    x_line = np.linspace(T_resid.min(), T_resid.max(), 100)
    axes4[1].plot(x_line, theta_dml * x_line, color="red", lw=2,
                  label=f"θ = {theta_dml:.4f}")
    axes4[1].axhline(0, color="gray", lw=0.5, linestyle="--")
    axes4[1].axvline(0, color="gray", lw=0.5, linestyle="--")
    axes4[1].set_xlabel("T residual (ê_T)")
    axes4[1].set_ylabel("Y residual (ê_Y)")
    axes4[1].set_title("DML Stage 3: ê_Y ~ ê_T")
    axes4[1].legend()

    plt.tight_layout()
    st.pyplot(fig4)

    st.divider()
    st.subheader("Nuisance R² 진단")
    r1, r2 = st.columns(2)
    r1.metric("T ~ X  R²", f"{r2_T:.3f}",
              help="낮을수록 인기도가 특성과 무관 → 좋음")
    r2.metric("Y ~ X  R²", f"{r2_Y:.3f}",
              help="높을수록 특성이 시청률을 잘 설명 → 좋음")
    st.markdown("""
> **R²(T~X) = {:.3f}** → 인기도는 유저·아이템 특성으로 거의 설명되지 않음  
> **R²(Y~X) = {:.3f}** → 시청률은 특성으로 잘 설명됨  
> → 두 R²의 비대칭이 **100% confounding** 의 핵심 증거
""".format(r2_T, r2_Y))