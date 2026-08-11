from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split


# ============================================================
# 1. 기본 설정
# ============================================================
st.set_page_config(
    page_title="아동학대 의심 예측 설문조사",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1240px;
        padding-top: 4.20rem !important;
        padding-bottom: 2.5rem;
    }
    section[data-testid="stSidebar"] {width: 18rem !important;}
    section[data-testid="stSidebar"] > div {width: 18rem !important;}

    h1 {font-size: 1.58rem !important; line-height: 1.25 !important; margin-top: 0 !important; margin-bottom: .30rem !important; padding-top:.12rem !important;}
    h2 {font-size: 1.15rem !important; line-height: 1.3 !important; margin-top: .9rem !important;}
    h3 {font-size: 1.12rem !important; line-height: 1.3 !important; margin-top: .82rem !important;}

    .subtitle {
        color:#5f6c79; font-size:.86rem; line-height:1.5; margin:.05rem 0 .55rem 0;
    }
    .mini-note {
        color:#687480; font-size:.77rem; line-height:1.5; margin:.2rem 0 .55rem 0;
    }
    .result-line {
        background:#f4f8fc; border-left:3px solid #5a83ad; border-radius:6px;
        padding:.62rem .82rem; margin:.35rem 0 .7rem 0;
        font-size:.82rem; line-height:1.6; color:#2f4254;
    }
    .warning-line {
        color:#7a5b1b; font-size:.77rem; line-height:1.5; margin:.25rem 0 .55rem 0;
    }

    .flow-wrap {
        display:flex; align-items:stretch; gap:.10rem; overflow-x:auto;
        padding:.15rem 0 .62rem 0;
    }
    .flow-step {
        flex:1; min-width:102px; background:#f6f8fb; border:1px solid #dbe2ea;
        border-radius:7px; padding:.46rem .26rem; text-align:center;
        font-size:.76rem; font-weight:650; color:#596979;
    }
    .flow-step-selected {
        flex:1; min-width:102px; background:#e8f2fd; border:1px solid #8eb4da;
        border-radius:7px; padding:.46rem .26rem; text-align:center;
        font-size:.76rem; font-weight:800; color:#174f7e;
    }
    .flow-arrow {
        display:flex; align-items:center; justify-content:center;
        color:#8a9bad; font-size:.88rem; padding:0;
    }


    .score-result-box {
        background:#eef8f0; border:1px solid #c6e4cc; border-left:4px solid #67a875;
        border-radius:10px; padding:1.05rem 1.05rem .95rem 1.05rem;
        margin:.15rem 0 .65rem 0; color:#284b31;
    }
    .score-result-label {font-size:.76rem; color:#64806b; margin-bottom:.18rem;}
    .score-result-value {font-size:1.55rem; font-weight:800; line-height:1.15; margin-bottom:.55rem;}
    .score-result-title {font-size:.92rem; font-weight:800; margin:.18rem 0 .28rem 0;}
    .score-result-text {font-size:.81rem; line-height:1.55; color:#35583d;}
    div[data-testid="stMetricValue"] {font-size:1.28rem;}
    div[data-testid="stMetricLabel"] {font-size:.76rem;}
    
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap:.28rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        font-size:.92rem !important;
        padding:.08rem 0 !important;
    }
    .metric-card {
        background:#ffffff; border:1px solid #d9e0e7; border-radius:9px;
        padding:.72rem .85rem; min-height:78px; box-shadow:0 1px 2px rgba(28,45,65,.04);
    }
    .metric-card-label {font-size:.76rem; color:#6b7885; margin-bottom:.22rem;}
    .metric-card-value {font-size:1.30rem; font-weight:800; color:#243746;}
    .explain-card {
        background:#f8fafc; border:1px solid #dce4ec; border-radius:9px;
        padding:.82rem .95rem; font-size:.80rem; line-height:1.65; color:#374957;
        margin:.10rem 0 .55rem 0;
    }
    .survey-title {
        font-size:.96rem; font-weight:800; color:#6b4521;
        margin:0 0 .55rem 0; padding:0; text-align:center;
    }
    div[data-testid="stColumn"]:has(.survey-panel-marker) {
        background:#fff3e4; border:1px solid #f0d2ad; border-left:5px solid #e3a15a;
        border-radius:12px; padding:.90rem 1.00rem 1.00rem 1.00rem;
        min-height:0;
        align-self:flex-start;
        height:fit-content;
    }
    div[data-testid="stColumn"]:has(.survey-panel-marker) div[data-testid="stCheckbox"] {
        background:transparent !important; border:0 !important; border-radius:0 !important;
        border-bottom:1px solid rgba(190,145,92,.18) !important;
        padding:.34rem .10rem !important; margin:0 !important;
    }
    .survey-panel-marker {display:none;}
    .hypothesis-card {
        background:#f8fafc; border:1px solid #dce4ec; border-radius:9px;
        padding:.72rem .88rem; min-height:92px; color:#344957;
        font-size:.81rem; line-height:1.55; margin:.08rem 0 .35rem 0;
    }
    .hypothesis-card b {font-size:.86rem; color:#253c4c;}
    .pvalue-good {
        background:#eef7f2; border:1px solid #cfe4d5; border-radius:9px;
        padding:.68rem .82rem; color:#315742; font-size:.82rem; line-height:1.55;
    }
    .pvalue-neutral {
        background:#f6f7f9; border:1px solid #dde2e7; border-radius:9px;
        padding:.68rem .82rem; color:#4d5b66; font-size:.82rem; line-height:1.55;
    }
    .test-flow {
        display:flex; align-items:center; gap:.36rem; flex-wrap:wrap;
        margin:.25rem 0 .65rem 0;
    }
    .test-flow-box {
        background:#f7f9fb; border:1px solid #dce3ea; border-radius:8px;
        padding:.52rem .70rem; font-size:.78rem; color:#3f5261; font-weight:650;
    }
    .test-flow-arrow {color:#91a0ad; font-size:.90rem;}

.split-diagram {margin:.28rem 0 .82rem 0;}
.split-root-row {display:flex; justify-content:center;}
.split-box {
    background:#f8fafc; border:1px solid #dce4ec; border-radius:10px;
    padding:.72rem .80rem; text-align:center; color:#354a5a;
    font-size:.82rem; line-height:1.5;
}
.split-box strong {font-size:.94rem; color:#243b4c;}
.split-box.root {background:#f4f8fc; border-color:#cdddea; width:34%;}
.split-box.train {background:#f6f8fb;}
.split-box.test {background:#eef7f2; border-color:#cfe4d5;}
.split-branches {display:grid; grid-template-columns:1fr 1fr; gap:.50rem; width:70%; margin:.18rem auto 0;}
.split-branch-arrows {display:flex; justify-content:center; gap:28%; color:#8ea0af; font-size:1.18rem; line-height:1; margin:.12rem 0 .12rem 0;}
.survey-section-spacer {height:.65rem;}
.student-result-callout {
    background:#f4f8fc; border:1px solid #d5e2ee; border-left:4px solid #6e96bb;
    border-radius:9px; padding:.78rem .88rem; margin:.20rem 0 .70rem 0;
    color:#2f485c; font-size:.98rem; line-height:1.58; font-weight:650;
}
.action-list-wrap {
    background:#fbfcfd; border:1px solid #dde5ec; border-radius:9px;
    padding:.58rem .68rem; margin:.18rem 0 .55rem 0;
}
.action-list-title {font-size:.80rem; font-weight:800; color:#405462; margin-bottom:.30rem;}
.action-row {
    padding:.34rem .18rem; border-top:1px solid #edf0f3;
    font-size:.70rem; line-height:1.45; color:#53626d;
}
.action-row:first-of-type {border-top:0;}
.action-row.active {background:#eef7f2; border-radius:6px; padding:.38rem .42rem; color:#315742; font-weight:700;}
.formula-box {
    background:#f8fafc; border:1px solid #dce4ec; border-radius:9px;
    padding:.72rem .85rem; margin:.20rem 0 .55rem 0; color:#334956;
    font-size:.88rem; line-height:1.65;
}


    /* ===== 최종 개요 카드 ===== */
    .overview-card-marker {display:none;}
    div[data-testid="stColumn"]:has(.overview-card-marker) {
        background:#ffffff; border:1px solid #dce4ec; border-radius:12px;
        padding:.28rem .80rem .92rem .80rem; min-height:258px;
        box-shadow:0 1px 3px rgba(30,48,67,.045);
    }
    div[data-testid="stElementContainer"]:has(.overview-card-marker) {display:none;}
    div[data-testid="stColumn"]:has(.overview-card-marker) div[data-testid="stMarkdown"]:has(.overview-card-marker) {display:none;}
    .overview-step {display:inline-block; font-size:.74rem; font-weight:850; letter-spacing:.03em; color:#77511a; background:#fff1b8; border-radius:5px; padding:.12rem .34rem; margin:-.12rem 0 .16rem 0;}
    .overview-title {font-size:1.10rem; font-weight:700; color:#263d4d; margin-bottom:.42rem;}
    .overview-kpi {font-size:.98rem; font-weight:400; color:#243b4c; line-height:1.25; margin:.05rem 0 .18rem 0;}
    .overview-text {font-size:.68rem; line-height:1.45; color:#536572; margin:.08rem 0;}
    .overview-result {font-size:.70rem; line-height:1.42; color:#315742; background:#f1f8f3; border-radius:7px; padding:.38rem .45rem; margin-top:.35rem;}
    .overview-model-row {display:flex; justify-content:space-between; gap:.4rem; padding:.24rem .36rem; margin:.10rem 0; border-radius:6px; font-size:.69rem; color:#52616d;}
    .overview-model-row.selected {background:#eaf6ed; border:1px solid #c9e2cf; color:#2f6040; font-weight:800;}
    .overview-weight-row {display:grid; grid-template-columns:46px 1fr 30px; gap:.20rem; align-items:center; margin:.10rem 0; font-size:.64rem; color:#52616d;}
    .overview-weight-row b {white-space:nowrap;}
    .overview-weight-track {height:5px; background:#edf1f4; border-radius:10px; overflow:hidden;}
    .overview-weight-fill {height:100%; background:#8ea9bb; border-radius:10px;}
    .overview-weight-summary {font-size:.62rem; color:#7a8893; text-align:center; margin:.18rem 0 .08rem 0;}
    .overview-flow {display:flex; align-items:center; justify-content:center; column-gap:.48rem; row-gap:.42rem; flex-wrap:wrap; margin:.62rem 0 1.05rem 0;}
    .overview-flow-node {background:#f6f8fb; border:1px solid #dce4ec; border-radius:7px; padding:.34rem .42rem; font-size:.67rem; font-weight:700; color:#465966;}
    .overview-flow-arrow {color:#91a0ad; font-size:.74rem;}
    .selected-model-note {background:#eaf6ed; border:1px solid #c9e2cf; border-left:4px solid #6fa77d; border-radius:8px; padding:.55rem .72rem; color:#315742; font-size:.79rem; line-height:1.5; margin:.38rem 0 .62rem 0;}
    .result-panel-marker {display:none;}
    div[data-testid="stColumn"]:has(.result-panel-marker) {
        background:#ffffff; border:1px solid #dce4ec; border-radius:12px;
        padding:.88rem .92rem 1.00rem .92rem; box-shadow:0 1px 3px rgba(30,48,67,.045);
    }
    .report-kicker {font-size:.70rem; font-weight:800; color:#758693; text-align:center; letter-spacing:.02em;}
    .report-name {font-size:.96rem; font-weight:800; color:#314958; text-align:center; margin-top:.14rem;}
    .report-score {font-size:1.72rem; font-weight:900; color:#3f7750; text-align:center; margin:.12rem 0 .06rem 0; line-height:1.1;}
    .report-band {display:table; margin:.10rem auto .42rem auto; background:#eef7f1; border:1px solid #cfe4d5; border-radius:999px; padding:.20rem .55rem; color:#356047; font-size:.72rem; font-weight:800;}
    .report-comment {background:#f5f8fb; border:1px solid #dce5ed; border-radius:9px; padding:.68rem .76rem; margin:.40rem 0 .62rem 0; color:#334d60; font-size:.88rem; line-height:1.58; font-weight:600;}
    .report-section-title {font-size:.79rem; font-weight:850; color:#405462; margin:.50rem 0 .24rem 0;}
    .action-list-wrap {background:#fbfcfd; border:1px solid #dde5ec; border-radius:9px; padding:.46rem .52rem; margin:.12rem 0 .55rem 0;}
    .action-list-title {font-size:1.00rem; font-weight:850; color:#405462; margin:.04rem .08rem .26rem .08rem;}
    .action-row {padding:.36rem .40rem; border-top:1px solid #edf0f3; font-size:.78rem; line-height:1.45; color:#53626d; border-radius:6px;}
    .action-row:first-of-type {border-top:0;}
    .action-row.active {background:#eaf6ed; border:1px solid #c9e2cf; color:#315742; font-weight:750;}
    .radar-note {font-size:.66rem; line-height:1.4; color:#71808b; text-align:center; margin-top:-.20rem;}

    </style>
    """,
    unsafe_allow_html=True,
)

available_fonts = {f.name for f in font_manager.fontManager.ttflist}
for candidate in [
    "Malgun Gothic",
    "NanumGothic",
    "NanumSquare",
    "Noto Sans CJK KR",
    "AppleGothic",
    "DejaVu Sans",
]:
    if candidate in available_fonts:
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 2. 데이터 / 결과 로드
# ============================================================
APP_DIR = Path(__file__).resolve().parent
ART = APP_DIR / "model_artifacts"
DATA_PATH = APP_DIR / "Autism-Child-Data.csv"


def load_csv(name):
    path = ART / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_json(name):
    path = ART / name
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


if not DATA_PATH.exists():
    st.error("Autism-Child-Data.csv 파일을 찾을 수 없습니다.")
    st.stop()

raw = pd.read_csv(DATA_PATH)
meta = load_json("final_model_metadata.json")
all_assoc = load_csv("all_feature_association.csv")
selected_importance = load_csv("selected_feature_importance.csv")
if selected_importance.empty:
    selected_importance = load_csv("all_feature_importance.csv")
model_compare = load_csv("model_validation_comparison.csv")
final_result = load_csv("final_candidate_results.csv")
roc_points = load_csv("final_roc_curve.csv")
limitations = load_csv("project_limitations.csv")
grouping_summary_df = load_csv("category_grouping_summary.csv")
weighted_checklist = load_csv("weighted_checklist_weights.csv")
weighted_validation = load_csv("weighted_checklist_threshold_validation.csv")
weighted_final = load_csv("weighted_checklist_threshold_final_test.csv")
weighted_band_validation = load_csv("weighted_checklist_band_validation.csv")
weighted_band_final = load_csv("weighted_checklist_band_final_test.csv")

final_row = final_result.iloc[0] if not final_result.empty else pd.Series(dtype="object")


# ============================================================
# 3. 데이터 정리 / 표시 이름
# ============================================================
def normalize_raw(df):
    out = df.copy()
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype("string").str.strip().str.strip("'").str.strip('"')
        out[c] = out[c].replace("?", pd.NA)
    out = out.rename(
        columns={
            "jundice": "jaundice",
            "austim": "family_asd",
            "contry_of_res": "country_of_res",
        }
    )
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
    return out


cleaned_before_fill = normalize_raw(raw)
analysis = cleaned_before_fill.drop_duplicates().reset_index(drop=True)
age_median = float(analysis["age"].median())
analysis["age"] = analysis["age"].fillna(age_median)
analysis["ethnicity"] = analysis["ethnicity"].fillna("Unknown")
analysis["relation"] = analysis["relation"].fillna("Unknown")

BEHAVIOR = [f"A{i}_Score" for i in range(1, 11)]
BACKGROUND = [
    "age",
    "gender",
    "ethnicity",
    "jaundice",
    "family_asd",
    "country_of_res",
    "used_app_before",
    "relation",
]
ALL_CANDIDATES = BEHAVIOR + BACKGROUND

BEHAVIOR_LABELS = {f"A{i}_Score": f"A{i} 문항" for i in range(1, 11)}
AQ10_OBSERVATION = {
    "A1_Score": "다른 사람이 잘 알아차리지 못하는 작은 소리를 자주 알아차림",
    "A2_Score": "전체보다 작은 세부사항에 더 집중하는 경향",
    "A3_Score": "여러 사람의 대화를 동시에 따라가기 어려움",
    "A4_Score": "한 활동에서 다른 활동으로 전환하기 어려움",
    "A5_Score": "또래와 대화를 계속 이어가는 방법을 어려워함",
    "A6_Score": "가벼운 사회적 대화(잡담)를 어려워함",
    "A7_Score": "이야기 속 인물의 의도나 감정을 파악하기 어려움",
    "A8_Score": "또래와 가상놀이·역할놀이를 즐기는 경향이 적음",
    "A9_Score": "표정만 보고 상대의 생각이나 감정을 파악하기 어려움",
    "A10_Score": "새 친구를 사귀기 어려움",
}
BACKGROUND_LABELS = {
    "age": "나이(Age)",
    "gender": "성별(Gender)",
    "ethnicity": "인종·민족(Ethnicity)",
    "jaundice": "황달 이력(Jaundice)",
    "family_asd": "가족 ASD 이력(Family ASD History)",
    "country_of_res": "거주 국가(Country)",
    "used_app_before": "이전 선별 앱 사용(Used App Before)",
    "relation": "응답자 관계(Relation)",
}
MODEL_LABELS = {
    "Logistic Regression": "로지스틱 회귀(Logistic Regression)",
    "KNN": "KNN(K-최근접 이웃)",
    "Decision Tree": "의사결정나무(Decision Tree)",
    "Random Forest": "랜덤 포레스트(Random Forest)",
}


def feature_name(name):
    name = str(name)
    return BEHAVIOR_LABELS.get(name, BACKGROUND_LABELS.get(name, name))


def short_feature_name(name):
    return feature_name(name).split("(")[0].strip()


def model_name(name):
    return MODEL_LABELS.get(str(name), str(name))


def fmt(x, digits=3):
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


if not all_assoc.empty and "significant_0_05" in all_assoc.columns:
    related = all_assoc[all_assoc["significant_0_05"].eq(True)].copy()
    not_related = all_assoc[all_assoc["significant_0_05"].eq(False)].copy()
else:
    related = pd.DataFrame()
    not_related = pd.DataFrame()

selected_features = meta.get("final_features", [])
selected_model = meta.get("selected_model", "Logistic Regression")


# ============================================================
# 4. 공통 UI - 표/그래프 20% 확대 + 왼쪽 정렬
# ============================================================
PIPELINE = [
    "프로젝트 개요",
    "데이터 전처리",
    "연관성 확인",
    "머신러닝 모델",
    "모델 성능 평가",
    "가중치 산출",
    "결론 및 활용",
]


def pipeline(active):
    if isinstance(active, int):
        active = [active]
    active = set(active or [])
    html = []
    for i, step in enumerate(PIPELINE):
        css = "flow-step-selected" if i in active else "flow-step"
        html.append(f'<div class="{css}">{i+1}. {step}</div>')
        if i < len(PIPELINE) - 1:
            html.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow-wrap">{"".join(html)}</div>', unsafe_allow_html=True)


def page_title(title, subtitle=""):
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)


def plain_list(items):
    st.markdown("\n".join([f"- **{k}**: {v}" for k, v in items]))


def result_line(text):
    st.markdown(f'<div class="result-line">{text}</div>', unsafe_allow_html=True)


def mini_note(text):
    st.markdown(f'<div class="mini-note">{text}</div>', unsafe_allow_html=True)


def score_result_box(score, title):
    st.markdown(
        f'''<div class="score-result-box">
            <div class="score-result-label">최종 가중 관찰 점수</div>
            <div class="score-result-value">{int(score)} / 100점</div>
            <div class="score-result-title">{title}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def result_report_header(student_name, score, title, comment):
    who = student_name.strip() if student_name and student_name.strip() else "해당 아동"
    st.markdown(
        f'''<div class="report-kicker">개별 관찰 결과</div>
        <div class="report-name">{who}</div>
        <div class="report-score">{int(score)} / 100점</div>
        <div class="report-band">{title}</div>
        <div class="report-comment"><b>{who}</b> {comment}</div>''',
        unsafe_allow_html=True,
    )


def student_result_callout(student_name, text):
    who = f"{student_name.strip()} 아동은" if student_name and student_name.strip() else "해당 아동은"
    st.markdown(
        f'<div class="student-result-callout">{who} {text}</div>',
        unsafe_allow_html=True,
    )


def compact_action_list(observe_cutoff, high_cutoff, very_high_cutoff, score):
    rows = [
        (0, observe_cutoff - 1, "일반 관찰", "일상 행동 변화를 지속적으로 관찰"),
        (observe_cutoff, high_cutoff - 1, "추가 관찰", "반복 행동의 빈도·상황·지속 여부를 기록"),
        (high_cutoff, very_high_cutoff - 1, "ASD 선별 고관찰", "관찰 내용을 보호자와 공유하고 공식 선별검사·전문 상담을 고려"),
        (very_high_cutoff, 100, "ASD 선별 매우 고관찰", "보호자와 협의하여 공식 선별검사 또는 전문기관 평가 연계를 우선 고려"),
    ]
    html = ['<div class="action-list-wrap"><div class="action-list-title">대응방안</div>']
    for lo, hi, label, action in rows:
        active = " active" if lo <= score <= hi else ""
        html.append(f'<div class="action-row{active}"><b>{lo}~{hi}점 · {label}</b><br>{action}</div>')
    html.append('</div>')
    st.markdown(''.join(html), unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-card-label">{label}</div><div class="metric-card-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def explain_card(html):
    st.markdown(f'<div class="explain-card">{html}</div>', unsafe_allow_html=True)


def left_table(df, width_ratio=0.68, height=None, hide_index=True):
    """표를 왼쪽에 배치. pandas Styler도 지원한다."""
    if df is None:
        return
    base_df = getattr(df, "data", df)
    if hasattr(base_df, "empty") and base_df.empty:
        return
    width_ratio = max(.40, min(float(width_ratio), .95))
    cols = st.columns([width_ratio, 1 - width_ratio], gap="small")
    kwargs = {"width": "stretch", "hide_index": hide_index}
    if isinstance(height, int) and height > 0:
        kwargs["height"] = height
    with cols[0]:
        st.dataframe(df, **kwargs)


def left_plot(fig, width_ratio=0.34):
    """그래프를 왼쪽에 배치."""
    width_ratio = max(.26, min(float(width_ratio), .82))
    cols = st.columns([width_ratio, 1 - width_ratio], gap="small")
    with cols[0]:
        st.pyplot(fig, width="stretch")
    plt.close(fig)


def center_plot(fig, width_ratio=0.36):
    """그래프를 페이지 가운데에 배치."""
    width_ratio = max(.26, min(float(width_ratio), .82))
    side = (1 - width_ratio) / 2
    cols = st.columns([side, width_ratio, side], gap="small")
    with cols[1]:
        st.pyplot(fig, width="stretch")
    plt.close(fig)


# ============================================================
# 5. Sidebar
# ============================================================
st.sidebar.markdown("### 📊 TP2 Dashboard")
menu = st.sidebar.radio(
    "메뉴",
    [
        "1. 프로젝트 개요",
        "2. 데이터 전처리",
        "3. 연관성 확인",
        "4. 머신러닝 모델",
        "5. 모델 성능 평가",
        "6. 가중치 산출",
        "7. 결론 및 활용",
    ],
    label_visibility="collapsed",
)


# ============================================================
# 6. Page 1 - 프로젝트 개요
# ============================================================
if menu.startswith("1."):
    page_title("아동학대 의심 예측 설문조사", "전처리부터 최종 활용까지 핵심 분석 결과를 한 화면에 요약한다.")

    behavior_related_count = 0
    background_related_count = 0
    if not all_assoc.empty and {"group", "significant_0_05"}.issubset(all_assoc.columns):
        behavior_related_count = int(
            all_assoc[(all_assoc["group"].eq("behavior")) & (all_assoc["significant_0_05"].eq(True))].shape[0]
        )
        background_related_count = int(
            all_assoc[(all_assoc["group"].eq("background")) & (all_assoc["significant_0_05"].eq(True))].shape[0]
        )

    row1 = st.columns(3, gap="small")

    with row1[0]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 1</div><div class="overview-title">데이터 전처리</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="overview-kpi">{len(raw)} → {len(analysis)}명</div>'
            f'<div class="overview-text">중복 {int(cleaned_before_fill.duplicated().sum())}건 제거 · '
            f'age 결측값 중앙값 {age_median:.0f}로 대체 · ethnicity/relation 결측값 Unknown 처리</div>',
            unsafe_allow_html=True,
        )
        if "Class/ASD" in analysis.columns:
            gm = analysis.groupby("Class/ASD")[BEHAVIOR].mean().reindex(["NO", "YES"])
            fig, ax = plt.subplots(figsize=(3.25, 1.35))
            x = np.arange(1, 11)
            ax.plot(x, gm.loc["NO"].values, marker="o", markersize=2.8, linewidth=1.2, label="NO 평균")
            ax.plot(x, gm.loc["YES"].values, marker="o", markersize=2.8, linewidth=1.2, label="YES 평균")
            ax.set_ylim(0, 1.03)
            ax.set_xticks(x)
            ax.set_xticklabels([f"A{i}" for i in x], fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(alpha=.18)
            ax.legend(fontsize=6, ncol=2, loc="lower center")
            fig.tight_layout(pad=.5)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        st.markdown(
            f'<div class="overview-result">결과 · 분석 가능한 {len(analysis)}명의 데이터 구성</div>',
            unsafe_allow_html=True,
        )

    with row1[1]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 2</div><div class="overview-title">연관성 확인</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="overview-kpi">행동 {behavior_related_count}개 · 배경 {background_related_count}개</div>'
            '<div class="overview-text">행동·범주형 요인은 카이제곱 검정, 나이는 독립표본 t-검정으로 확인 · p-value 0.05 미만을 통계적으로 유의한 기준으로 적용</div>',
            unsafe_allow_html=True,
        )
        if not all_assoc.empty:
            sig_preview = all_assoc[all_assoc["significant_0_05"].eq(True)].sort_values("p_value").head(4)
            lines = []
            for _, r in sig_preview.iterrows():
                p = float(r["p_value"])
                lines.append(
                    f'<div class="overview-model-row"><span>{short_feature_name(r["feature"])}</span>'
                    f'<b>p={p:.1e}</b></div>'
                )
            st.markdown("".join(lines), unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-result">A1~A10은 모두 p&lt;0.05, 개인·배경 요인은 p&lt;0.05 없음</div>',
            unsafe_allow_html=True,
        )

    with row1[2]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 3</div><div class="overview-title">머신러닝 모델 비교</div>',
            unsafe_allow_html=True,
        )
        if not model_compare.empty:
            for _, r in model_compare.sort_values("selection_rank").iterrows():
                selected_css = " selected" if str(r["model"]) == "Logistic Regression" else ""
                check = "✓ " if selected_css else ""
                display_model = model_name(r["model"]).split("(")[0].strip()
                st.markdown(
                    f'<div class="overview-model-row{selected_css}"><span>{check}{display_model}</span>'
                    f'<b>F1 {float(r["f1"]):.3f}</b></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            '<div class="overview-result">결과 · Validation 성능 비교 후 로지스틱 회귀를 최종 모델로 선택</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:.42rem"></div>', unsafe_allow_html=True)
    row2 = st.columns(3, gap="small")

    with row2[0]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 4</div><div class="overview-title">모델 성능 평가</div>',
            unsafe_allow_html=True,
        )
        if not final_row.empty:
            final_n = int(
                final_row.get("tn", 0) + final_row.get("fp", 0)
                + final_row.get("fn", 0) + final_row.get("tp", 0)
            )
            st.markdown(
                f'<div class="overview-kpi">Accuracy {float(final_row.get("accuracy",0)):.2f} · '
                f'F1 {float(final_row.get("f1",0)):.2f}</div>'
                f'<div class="overview-text">Final Test {final_n}명 · ROC-AUC {float(final_row.get("roc_auc",0)):.2f}</div>',
                unsafe_allow_html=True,
            )
            mc1, mc2 = st.columns(2, gap="small")
            with mc1:
                cm = np.array([
                    [int(final_row.get("tn", 0)), int(final_row.get("fp", 0))],
                    [int(final_row.get("fn", 0)), int(final_row.get("tp", 0))],
                ])
                fig, ax = plt.subplots(figsize=(1.55, 1.28))
                ax.imshow(
                    cm,
                    cmap=LinearSegmentedColormap.from_list("mini_cm", ["#fbf9fd", "#ded1ee"]),
                    vmin=0,
                    vmax=max(int(cm.max()), 1),
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
                ax.set_title("혼동행렬", fontsize=7)
                fig.tight_layout(pad=.2)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with mc2:
                if not roc_points.empty:
                    fig, ax = plt.subplots(figsize=(1.55, 1.28))
                    ax.plot(roc_points["fpr"], roc_points["tpr"], linewidth=1.2)
                    ax.plot([0, 1], [0, 1], "--", linewidth=.6)
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.tick_params(labelsize=5)
                    ax.set_title("ROC", fontsize=7)
                    fig.tight_layout(pad=.2)
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
        st.markdown(
            '<div class="overview-result">결과 · Final Test에서 선택 모델의 최종 성능 확인</div>',
            unsafe_allow_html=True,
        )

    with row2[1]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 5</div><div class="overview-title">문항별 가중치</div>',
            unsafe_allow_html=True,
        )
        if not weighted_checklist.empty:
            topw = weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)].sort_values("rank").head(6)
            maxp = max(float(topw["points"].max()), 1)
            for _, r in topw.iterrows():
                pct = int(round(float(r["points"]) / maxp * 100))
                st.markdown(
                    f'<div class="overview-weight-row"><b>{short_feature_name(r["feature"])}</b>'
                    f'<div class="overview-weight-track"><div class="overview-weight-fill" style="width:{pct}%"></div></div>'
                    f'<b>{int(r["points"])}점</b></div>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                '<div class="overview-weight-summary">⋮ &nbsp; A1~A10 전체 10문항 중 중요도 상위 6개 문항만 요약 표시</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="overview-result">결과 · 로지스틱 회귀계수 크기를 이용해 A1~A10의 합계 100점 가중점수 구성</div>',
            unsafe_allow_html=True,
        )

    with row2[2]:
        st.markdown('<span class="overview-card-marker"></span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="overview-step">STEP 6</div><div class="overview-title">최종 활용</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="overview-flow">'
            '<div class="overview-flow-node">A1~A10 설문</div><div class="overview-flow-arrow">→</div>'
            '<div class="overview-flow-node">가중점수</div><div class="overview-flow-arrow">→</div>'
            '<div class="overview-flow-node">관찰 구간</div><div class="overview-flow-arrow">→</div>'
            '<div class="overview-flow-node">대응방안</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="overview-text">개별 아동의 설문 응답을 100점 가중 관찰 점수로 계산하고, '
            '점수 구간에 맞는 관찰·상담·전문 선별 연계 방향을 제시한다.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="overview-result">결과 · 설문 입력부터 개별 결과 리포트까지 한 화면에서 확인</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 7. Page 2 - 데이터 전처리
# ============================================================
elif menu.startswith("2."):
    page_title("2. 데이터 전처리", "결측값과 중복값을 정리하고 분석에 사용할 형태로 변환한다.")
    pipeline(1)

    duplicate_count = int(cleaned_before_fill.duplicated().sum())
    missing_count = int(cleaned_before_fill.isna().sum().sum())
    st.subheader("전처리 내용")
    preprocess_tbl = pd.DataFrame(
        [
            ["중복 데이터 처리", "중복 데이터", duplicate_count, "중복 행 제거"],
            ["결측값 처리", "age", int(cleaned_before_fill["age"].isna().sum()), f"중앙값 {age_median:.0f}로 대체"],
            ["결측값 처리", "ethnicity", int(cleaned_before_fill["ethnicity"].isna().sum()), "Unknown으로 대체"],
            ["결측값 처리", "relation", int(cleaned_before_fill["relation"].isna().sum()), "Unknown으로 대체"],
            ["변수 제거", "age_desc", 0, "상수 변수라 제외"],
            ["데이터 누수 방지", "result", 0, "ML 입력에서 제외"],
        ],
        columns=["전처리 종류", "항목", "개수", "처리 방법"],
    )
    left_table(preprocess_tbl, .84)
    mini_note(
        "<b>사용하지 않은 전처리: 정규화·표준화(Scaling)</b><br>"
        "최종 입력 변수 A1~A10이 모두 0 또는 1의 동일한 범위를 가지므로 별도의 정규화·표준화는 적용하지 않음."
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        metric_card("행(Row)", len(raw))
    with c2:
        metric_card("열(Column)", raw.shape[1])
    with c3:
        metric_card("중복", duplicate_count)
    with c4:
        metric_card("결측값", missing_count)

    st.subheader("원본 데이터")
    preview_top = raw.head(10).copy().astype(object)
    preview_last = raw.tail(1).copy().astype(object)
    ellipsis = pd.DataFrame([{c: "…" for c in raw.columns}], index=["…"])
    preview = pd.concat([preview_top, ellipsis, preview_last], axis=0)
    preview.index = list(range(min(10, len(raw)))) + ["…"] + [len(raw)-1]
    excluded_preview_cols = [c for c in ["result", "age_desc"] if c in preview.columns]
    preview_style = preview.style
    if excluded_preview_cols:
        preview_style = preview_style.set_properties(
            subset=excluded_preview_cols,
            **{"background-color": "#e6e6e6", "color": "#6f6f6f"}
        )
    st.dataframe(preview_style, width="stretch", hide_index=False, height=455)

    st.subheader("행동 문항 분포 확인(EDA)그래프")
    eda_df = analysis.copy()
    if "Class/ASD" in eda_df.columns:
        group_means = eda_df.groupby("Class/ASD")[BEHAVIOR].mean().reindex(["NO", "YES"])
        fig, ax = plt.subplots(figsize=(7.0, 3.1))
        x = np.arange(len(BEHAVIOR))
        width = .36
        ax.bar(x - width/2, group_means.loc["NO"].values, width, label="ASD 선별 NO")
        ax.bar(x + width/2, group_means.loc["YES"].values, width, label="ASD 선별 YES")
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{i}" for i in range(1, 11)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("행동 문항 (A1~A10)", fontsize=8)
        ax.set_ylabel("평균 응답 비율 (1점)", fontsize=8)
        ax.set_title("ASD 선별 YES/NO별 행동 문항 평균", fontsize=10)
        ax.legend(fontsize=8, ncol=2)
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        left_plot(fig, .64)

        overview_means = analysis.groupby("Class/ASD")[BEHAVIOR].mean().reindex(["NO", "YES"])
        fig, ax = plt.subplots(figsize=(7.0, 3.1))
        x = np.arange(1, 11)
        ax.plot(x, overview_means.loc["NO"].values, marker="o", markersize=4, linewidth=1.5, label="NO 평균")
        ax.plot(x, overview_means.loc["YES"].values, marker="o", markersize=4, linewidth=1.5, label="YES 평균")
        ax.set_ylim(0, 1.03)
        ax.set_xticks(x)
        ax.set_xticklabels([f"A{i}" for i in x], fontsize=8)
        ax.set_xlabel("행동 문항 (A1~A10)", fontsize=8)
        ax.set_ylabel("평균 응답 비율 (1점)", fontsize=8)
        ax.set_title("ASD 선별 YES/NO별 행동 문항 평균 추이", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        left_plot(fig, .64)


# ============================================================
# 8. Page 3 - P-value / Hypothesis
# ============================================================
elif menu.startswith("3."):
    page_title("3. 연관성 확인", "각 요인이 ASD 선별 결과와 통계적으로 관련이 있는지 확인한다.")
    pipeline(2)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown(
            '<div class="hypothesis-card"><b>검정 방법</b><br>행동·범주형 요인은 카이제곱 검정, 나이는 독립표본 t-검정을 사용함.<br>각 요인이 ASD 선별 YES/NO 결과와 통계적으로 관계가 있는지 확인함.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="hypothesis-card"><b>P-value</b><br>A1~A10 행동 문항은 모두 p&lt;0.05로 귀무가설을 기각.<br>개인·배경 요인 중 p&lt;0.05인 변수는 없어 귀무가설을 기각하지 못함.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '''<div class="test-flow">
        <div class="test-flow-box">각 문항·요인</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">검정통계 계산</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">P-value 계산</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">0.05 기준 판정</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">귀무가설 기각 / 기각 못함</div>
        </div>''',
        unsafe_allow_html=True,
    )

    if not all_assoc.empty:
        st.subheader("가설검정 결과(Hypothesis Testing Result)")
        result_tbl = all_assoc[["feature", "group", "test", "statistic", "p_value", "significant_0_05"]].copy()
        result_tbl["feature"] = result_tbl["feature"].map(short_feature_name)
        result_tbl["group"] = result_tbl["group"].map({"behavior": "행동", "background": "개인·배경"})
        result_tbl["test"] = result_tbl["test"].replace(
            {"Chi-square": "카이제곱 검정", "Welch t-test": "독립표본 t-검정"}
        )
        result_tbl["statistic"] = pd.to_numeric(result_tbl["statistic"], errors="coerce").map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
        )
        result_tbl["p_value_num"] = pd.to_numeric(result_tbl["p_value"], errors="coerce")
        result_tbl = result_tbl.sort_values(["significant_0_05", "p_value_num"], ascending=[False, True])
        result_tbl["p_value"] = result_tbl["p_value_num"].map(
            lambda x: f"{x:.2e}" if pd.notna(x) and x < 0.001 else (f"{x:.3f}" if pd.notna(x) else "-")
        )
        result_tbl["significant_0_05"] = result_tbl["significant_0_05"].map(
            {True: "귀무가설 기각 (유의)", False: "기각 못함"}
        )
        result_tbl = result_tbl[["feature", "group", "test", "statistic", "p_value", "significant_0_05"]]
        result_tbl.columns = ["요인", "구분", "검정 방법", "검정통계량", "P-value", "판정"]
        left_table(result_tbl, .94, 610)

    with st.expander("P-value 계산 예시 · A4"):
        example_df = analysis.copy()
        example_df["target"] = example_df["Class/ASD"].map({"NO": 0, "YES": 1}).astype(int)
        dev_example, _ = train_test_split(
            example_df, test_size=0.20, stratify=example_df["target"], random_state=42
        )
        selection_example, _ = train_test_split(
            dev_example, test_size=0.25, stratify=dev_example["target"], random_state=42
        )
        observed = pd.crosstab(selection_example["A4_Score"], selection_example["target"]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        chi2_value, p_example, _, expected = chi2_contingency(observed)

        observed_view = observed.copy()
        observed_view.index = ["A4=0", "A4=1"]
        observed_view.columns = ["ASD 선별 NO", "ASD 선별 YES"]
        expected_view = pd.DataFrame(expected, index=observed_view.index, columns=observed_view.columns).round(1)

        ec1, ec2 = st.columns(2, gap="small")
        with ec1:
            st.markdown("**실제 관측값**")
            st.dataframe(observed_view, width="stretch")
        with ec2:
            st.markdown("**관계가 없다고 가정했을 때의 기대값**")
            st.dataframe(expected_view, width="stretch")

        st.latex(r"\chi^2 = \sum \frac{(\text{관측값}-\text{기대값})^2}{\text{기대값}}")
        result_line(
            f"A4 검정통계량 χ²={chi2_value:.2f}, P-value={p_example:.2e}. "
            "P-value가 0.05보다 작으므로 귀무가설을 기각한다."
        )


# ============================================================
# 9. Page 4 - 머신러닝 모델
# ============================================================
elif menu.startswith("4."):
    page_title("4. 머신러닝 모델", "4개 분류 모델을 같은 조건에서 학습하고 Validation 성능을 비교한다.")
    pipeline(3)

    plain_list(
        [
            ("분류 / 회귀", "분류(Classification) — Target이 YES/NO 두 범주이며, 로지스틱 회귀도 이진 분류에 사용하는 모델"),
            ("비교 모델", "로지스틱 회귀(Logistic Regression) / K-최근접 이웃(KNN) / 의사결정나무(Decision Tree) / 랜덤 포레스트(Random Forest)"),
        ]
    )

    st.subheader("4개 모델 Validation 비교")
    if not model_compare.empty:
        tbl = model_compare[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
        tbl["model"] = tbl["model"].map(model_name)
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            tbl[c] = tbl[c].map(lambda x: f"{float(x):.3f}")
        tbl.columns = ["사용 모델", "정확도(Accuracy)", "정밀도(Precision)", "재현율(Recall)", "F1 점수", "ROC-AUC"]

        def highlight_selected_model(row):
            is_lr = str(row["사용 모델"]).startswith("로지스틱 회귀")
            style = "background-color:#eaf6ed; color:#315742; font-weight:700;" if is_lr else ""
            return [style] * len(row)

        styled_tbl = tbl.style.apply(highlight_selected_model, axis=1)
        left_table(styled_tbl, .94)
        st.markdown(
            '<div class="selected-model-note"><b>✓ 로지스틱 회귀(Logistic Regression)</b> · '
            'Validation에서 가장 높은 종합 성능을 보여 최종 모델로 선택</div>',
            unsafe_allow_html=True,
        )

        plot_df = model_compare.sort_values("f1", ascending=True)
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        labels = plot_df["model"].replace(
            {"Logistic Regression": "Logistic", "Decision Tree": "Tree", "Random Forest": "RF"}
        )
        colors = ["#88b895" if m == "Logistic Regression" else "#c8d1d8" for m in plot_df["model"]]
        bars = ax.barh(labels, plot_df["f1"], color=colors)
        ax.set_xlim(0.75, 1.03)
        ax.set_xlabel("Validation F1")
        for bar, val in zip(bars, plot_df["f1"]):
            ax.text(
                float(val) + .005,
                bar.get_y() + bar.get_height() / 2,
                f"{float(val):.3f}",
                va="center",
                fontsize=8,
            )
        fig.tight_layout()
        center_plot(fig, .38)

        best = model_compare.sort_values(["f1", "recall", "roc_auc"], ascending=False).iloc[0]
        result_line(
            f"Validation 기준 최종 선택 모델: <b>{model_name(best['model'])}</b> &nbsp; | &nbsp; "
            f"F1 {fmt(best['f1'])} / Recall {fmt(best['recall'])} / ROC-AUC {fmt(best['roc_auc'])}"
        )


# ============================================================
# 10. Page 5 - 모델 성능 평가
# ============================================================
elif menu.startswith("5."):
    page_title("5. 모델 성능 평가")
    pipeline(4)

    st.subheader("학습·테스트 데이터 분할")
    st.markdown(
        f'''<div class="split-diagram">
            <div class="split-root-row"><div class="split-box root"><strong>전체 데이터 {len(analysis)}명</strong><br>전처리 완료 데이터</div></div>
            <div class="split-branch-arrows"><span>↙</span><span>↘</span></div>
            <div class="split-branches">
                <div class="split-box train"><strong>80% · {int(round(len(analysis)*0.8))}명</strong><br>학습·검증용<br>모델 학습·선택에 사용</div>
                <div class="split-box test"><strong>20% · {int(round(len(analysis)*0.2))}명</strong><br>Final Test<br>마지막 성능 평가에 사용</div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:.70rem"></div>', unsafe_allow_html=True)
    st.subheader("성능평가 결과")

    if not final_row.empty:
        metric_table = pd.DataFrame(
            [
                ["정확도(Accuracy)", final_row.get("accuracy"), "전체 중 맞힌 비율"],
                ["정밀도(Precision)", final_row.get("precision"), "YES라고 분류한 것 중 실제 YES 비율"],
                ["재현율(Recall)", final_row.get("recall"), "실제 YES 중 찾아낸 비율"],
                ["F1 점수(F1-score)", final_row.get("f1"), "Precision과 Recall의 균형"],
                ["ROC-AUC", final_row.get("roc_auc"), "YES와 NO를 전반적으로 구분하는 능력"],
            ],
            columns=["평가 지표", "Final Test 결과", "설명"],
        )
        metric_table["Final Test 결과"] = metric_table["Final Test 결과"].map(lambda x: fmt(x))
        left_table(metric_table, .82)

        tn = int(final_row.get("tn", 0))
        fp = int(final_row.get("fp", 0))
        fn = int(final_row.get("fn", 0))
        tp = int(final_row.get("tp", 0))
        cm = np.array([[tn, fp], [fn, tp]])

        st.subheader("혼동행렬(Confusion Matrix)")
        cm_left, cm_right = st.columns([.32, .68], gap="large")
        with cm_left:
            fig, ax = plt.subplots(figsize=(3.0, 2.45))
            light_purple = LinearSegmentedColormap.from_list(
                "light_purple", ["#fbf9fd", "#eee7f8", "#ddcef1"]
            )
            ax.imshow(cm, cmap=light_purple, vmin=0, vmax=max(int(cm.max()), 1))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["예측 NO", "예측 YES"], fontsize=7)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["실제 NO", "실제 YES"], fontsize=7)
            labels_cm = [["TN", "FP"], ["FN", "TP"]]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{labels_cm[i][j]}\n{cm[i, j]}", ha="center", va="center", fontsize=9, color="#3d3650")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        with cm_right:
            explain_card(
                f"<b>TP</b> 실제 ASD 선별 YES → 모델도 YES라고 예측: <b>{tp}명</b><br>"
                f"<b>TN</b> 실제 ASD 선별 NO → 모델도 NO라고 예측: <b>{tn}명</b><br>"
                f"<b>FP</b> 실제 NO인데 모델이 YES라고 잘못 예측: <b>{fp}명</b><br>"
                f"<b>FN</b> 실제 YES인데 모델이 NO라고 잘못 예측: <b>{fn}명</b><br><br>"
                f"<b>결과:</b> Final Test {tp+tn+fp+fn}명 중 {tp+tn}명을 정확히 분류했다."
            )

        if not roc_points.empty and {"fpr", "tpr"}.issubset(roc_points.columns):
            st.subheader("ROC 곡선(ROC Curve)")
            roc_left, roc_right = st.columns([.32, .68], gap="large")
            with roc_left:
                fig, ax = plt.subplots(figsize=(3.0, 2.45))
                ax.plot(roc_points["fpr"], roc_points["tpr"], label=f"AUC={fmt(final_row.get('roc_auc'))}")
                ax.plot([0, 1], [0, 1], "--", linewidth=.8)
                ax.set_xlabel("FPR", fontsize=8)
                ax.set_ylabel("TPR", fontsize=8)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            with roc_right:
                auc = float(final_row.get("roc_auc", 0))
                explain_card(
                    "ROC 곡선(ROC Curve)은 ASD 선별 YES/NO를 구분할 때 모델의 판단 기준을 바꿔가며 성능 변화를 보여준다.<br>"
                    "세로축(TPR)은 실제 ASD 선별 YES를 YES로 찾아낸 비율이고, 가로축(FPR)은 실제 ASD 선별 NO를 YES로 잘못 판단한 비율이다.<br>"
                    "그래프가 왼쪽 위에 가까울수록 구분 성능이 좋고, AUC는 1에 가까울수록 좋다.<br><br>"
                    f"<b>현재 로지스틱 회귀 Final Test 결과: ROC-AUC {auc:.3f}</b>"
                )


# ============================================================
# 11. Page 6 - 가중치 산출
# ============================================================
elif menu.startswith("6."):
    page_title("6. 가중치 산출")
    pipeline(5)

    st.subheader("가중치 산출 방법")
    plain_list(
        [
            ("1. 모델 학습", "A1~A10으로 Class/ASD YES/NO를 분류하는 로지스틱 회귀(Logistic Regression) 학습"),
            ("2. 문항 중요도", "각 문항의 회귀계수 절댓값을 사용"),
        ]
    )

    if not weighted_checklist.empty:
        checklist_weight_view = weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)].copy()
        checklist_weight_view = checklist_weight_view.sort_values("rank").head(10)
        coef_sum = float(checklist_weight_view["raw_importance"].astype(float).sum())
        example = checklist_weight_view.iloc[0]
        st.markdown(
            '<div class="formula-box">'
            '<b>문항별 가중점수</b> = |해당 문항 회귀계수| ÷ '
            '(|A1 회귀계수| + |A2 회귀계수| + ··· + |A10 회귀계수|) × 100'
            '<br><span style="display:inline-block; margin-top:8px;">'
            f'계산 예: <b>{short_feature_name(example["feature"])}</b> |회귀계수| {float(example["raw_importance"]):.3f} ÷ '
            f'A1~A10 |회귀계수| 합 {coef_sum:.3f} × 100 = <b>{float(example["raw_importance"])/coef_sum*100:.2f}%</b> → <b>{int(example["points"])}점</b>'
            '</span></div>',
            unsafe_allow_html=True,
        )

        st.subheader("최종 문항별 가중점수")
        wtbl = checklist_weight_view[["rank", "feature", "raw_importance", "normalized_weight", "points"]].copy()
        wtbl["feature"] = wtbl["feature"].map(short_feature_name)
        wtbl["raw_importance"] = wtbl["raw_importance"].map(lambda x: f"{float(x):.3f}")
        wtbl["normalized_weight"] = wtbl["normalized_weight"].map(lambda x: f"{float(x)*100:.2f}%")
        wtbl.columns = ["순위", "문항", "|회귀계수|", "상대 가중치", "최종 점수"]
        styled_wtbl = (
            wtbl.style
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
        )
        left_table(styled_wtbl, .74, 390)

    checklist_meta = meta.get("weighted_checklist", {})
    observe_cutoff = int(checklist_meta.get("observe_cutoff", 50))
    high_cutoff = int(checklist_meta.get("high_cutoff", 65))
    very_high_cutoff = int(checklist_meta.get("very_high_cutoff", 80))
    st.subheader("점수 기준 결정")
    result_line(
        f"가중점수와 ASD 선별 YES/NO 분포를 다시 비교해 고관찰 시작점을 <b>{high_cutoff}점</b>으로 설정했다. "
        f"고관찰 구간은 YES 사례의 점수 분포 중앙값을 기준으로 <b>{very_high_cutoff}점</b>에서 한 번 더 나눴다.<br>"
        f"최종 구간: <b>0~{observe_cutoff-1} / {observe_cutoff}~{high_cutoff-1} / "
        f"{high_cutoff}~{very_high_cutoff-1} / {very_high_cutoff}~100점</b>"
    )


# ============================================================
# 12. Page 7 - 결론 및 활용
# ============================================================
elif menu.startswith("7."):
    page_title("7. 결론 및 활용")
    pipeline(6)

    checklist_meta = meta.get("weighted_checklist", {})
    observe_cutoff = int(checklist_meta.get("observe_cutoff", 50))
    high_cutoff = int(checklist_meta.get("high_cutoff", 65))
    very_high_cutoff = int(checklist_meta.get("very_high_cutoff", 80))

    total_score = 0
    checked_items = []
    if not weighted_checklist.empty:
        checklist_df = (
            weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)]
            .sort_values("rank")
            .head(10)
            .copy()
        )

        st.subheader("최종 설문조사지")
        st.markdown('<div class="survey-section-spacer"></div>', unsafe_allow_html=True)
        check_col, result_col = st.columns([.57, .43], gap="large")

        with check_col:
            st.markdown('<span class="survey-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="survey-title">아동학대 의심 설문조사</div>',
                unsafe_allow_html=True,
            )
            student_name = st.text_input(
                "학생 이름",
                placeholder="예: 김민수",
                key="student_name",
            )
            for _, row in checklist_df.iterrows():
                feature = str(row["feature"])
                points = int(row["points"])
                label = (
                    f"{short_feature_name(feature)} — "
                    f"{AQ10_OBSERVATION.get(feature, 'AQ-10 행동 문항')}"
                )
                checked = st.checkbox(label, key=f"teacher_check_{feature}")
                if checked:
                    total_score += points
                    checked_items.append(feature)

        if total_score >= very_high_cutoff:
            result_title = "ASD 선별 매우 고관찰 구간"
            result_comment = (
                "아동에게 ASD 관련 행동 특성이 비교적 뚜렷하게 관찰됩니다.<br>"
                "설문 결과만으로 진단할 수는 없으므로, 보호자와 상의한 뒤 "
                "공식 선별검사 또는 전문기관 평가를 권장합니다."
            )
        elif total_score >= high_cutoff:
            result_title = "ASD 선별 고관찰 구간"
            result_comment = (
                "아동에게 여러 ASD 관련 행동 특성이 함께 관찰됩니다.<br>"
                "관찰 내용을 보호자와 공유하고, 필요하면 공식 선별검사나 "
                "전문 상담을 고려하는 것이 좋습니다."
            )
        elif total_score >= observe_cutoff:
            result_title = "추가 관찰 구간"
            result_comment = (
                "아동에게 일부 ASD 관련 행동 특성이 관찰됩니다.<br>"
                "같은 행동이 특정 상황에서 반복되는지 기록하고, 변화가 지속되는지 "
                "조금 더 주의 깊게 살펴보는 것이 좋습니다."
            )
        else:
            result_title = "일반 관찰 구간"
            result_comment = (
                "아동은 현재 설문에서 ASD 관련 행동 특성이 낮은 수준으로 관찰됩니다.<br>"
                "일상생활에서 특정 행동이 반복되거나 변화가 나타나는지 "
                "평소 관찰을 지속해 주세요."
            )

        with result_col:
            st.markdown('<span class="result-panel-marker"></span>', unsafe_allow_html=True)
            result_report_header(
                student_name,
                total_score,
                result_title,
                result_comment,
            )
            compact_action_list(
                observe_cutoff,
                high_cutoff,
                very_high_cutoff,
                total_score,
            )

            st.markdown(
                '<div class="report-section-title">행동 특성 비교</div>',
                unsafe_allow_html=True,
            )

            no_mean = (
                analysis.loc[analysis["Class/ASD"].eq("NO"), BEHAVIOR]
                .mean()
                .values.astype(float)
            )
            child_values = np.array(
                [1.0 if f in checked_items else 0.0 for f in BEHAVIOR],
                dtype=float,
            )
            angles = np.linspace(0, 2 * np.pi, len(BEHAVIOR), endpoint=False)
            angles_closed = np.r_[angles, angles[0]]
            no_closed = np.r_[no_mean, no_mean[0]]
            child_closed = np.r_[child_values, child_values[0]]

            fig, ax = plt.subplots(
                figsize=(2.30, 1.92),
                subplot_kw={"polar": True},
            )
            ax.plot(
                angles_closed,
                no_closed,
                linewidth=.70,
                marker="o",
                markersize=1.5,
                label="ASD 선별 NO 그룹 평균",
            )
            ax.fill(angles_closed, no_closed, alpha=.08)
            ax.plot(
                angles_closed,
                child_closed,
                linewidth=.82,
                marker="o",
                markersize=1.6,
                label="현재 아동",
            )
            ax.fill(angles_closed, child_closed, alpha=.08)
            ax.set_xticks(angles)
            ax.set_xticklabels(
                [f"A{i}" for i in range(1, 11)],
                fontsize=5.5,
                fontweight="normal",
            )
            ax.tick_params(axis="x", pad=-3)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([.25, .50, .75, 1.0])
            ax.set_yticklabels(
                [".25", ".50", ".75", "1.0"],
                fontsize=5,
                color="#7b8790",
            )
            ax.grid(alpha=.25)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                fontsize=6.5,
                ncol=2,
                frameon=False,
            )
            fig.tight_layout(pad=.8)
            st.pyplot(fig, width="content")
            plt.close(fig)
            st.markdown(
                '<div class="radar-note">ASD 선별 NO 그룹의 A1~A10 평균 응답과 '
                '현재 설문 응답 패턴을 비교합니다.</div>',
                unsafe_allow_html=True,
            )

    with st.expander("점수 구간별 선정 기준"):
        st.markdown(
            "이 점수 구간은 <b>의료 진단 기준이 아니라 프로젝트 데이터의 "
            "가중점수 분포를 바탕으로 만든 관찰 단계</b>입니다.",
            unsafe_allow_html=True,
        )

        cutoff_basis = checklist_meta.get("cutoff_basis", {})
        development_no_max = int(
            cutoff_basis.get("development_no_max", 64)
        )
        development_yes_min = int(
            cutoff_basis.get("development_yes_min", 67)
        )
        development_yes_median = float(
            cutoff_basis.get("development_yes_median", 80.0)
        )

        criteria_tbl = pd.DataFrame(
            [
                [
                    f"0~{observe_cutoff-1}점",
                    "일반 관찰",
                    "낮은 점수 구간의 기본 관찰 단계",
                ],
                [
                    f"{observe_cutoff}~{high_cutoff-1}점",
                    "추가 관찰",
                    f"{high_cutoff}점 미만을 일반/추가 관찰로 구분하기 위해 "
                    "설정한 프로젝트 운영 기준",
                ],
                [
                    f"{high_cutoff}~{very_high_cutoff-1}점",
                    "ASD 선별 고관찰",
                    f"Development에서 NO 최고 {development_no_max}점, "
                    f"YES 최저 {development_yes_min}점 사이를 기준으로 "
                    f"{high_cutoff}점부터 설정",
                ],
                [
                    f"{very_high_cutoff}~100점",
                    "ASD 선별 매우 고관찰",
                    f"Development의 ASD 선별 YES 그룹 가중점수 중앙값"
                    f"({development_yes_median:.0f}점)을 기준으로 구분",
                ],
            ],
            columns=["점수 구간", "관찰 단계", "선정 기준"],
        )
        st.dataframe(
            criteria_tbl,
            width="stretch",
            hide_index=True,
        )
        mini_note(
            f"정리하면 <b>{observe_cutoff}점</b>은 프로젝트 운영 기준, "
            f"<b>{high_cutoff}점</b>은 Development 데이터의 NO/YES 점수 분포 경계, "
            f"<b>{very_high_cutoff}점</b>은 Development의 ASD 선별 YES 그룹 "
            "중앙값을 기준으로 설정했습니다."
        )
