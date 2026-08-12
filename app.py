from pathlib import Path
import json
import pickle

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
    .learning-card {
        height:100%; min-height:132px; background:#f8fafc; border:1px solid #dce4ec;
        border-top:4px solid #7ea6c4; border-radius:10px; padding:.72rem .78rem;
        font-size:.76rem; line-height:1.55; color:#374957;
    }
    .learning-card-title {font-weight:800; color:#315b78; font-size:.82rem; margin-bottom:.38rem;}
    .flow-step-card {
        height:100%; min-height:170px; background:#ffffff; border:1px solid #d9e3eb;
        border-radius:11px; padding:.82rem .88rem; box-shadow:0 1px 3px rgba(29,56,77,.05);
    }
    .flow-step-card.final {border-top:5px solid #5c8fae; background:#f1f7fb;}
    .flow-step-number {font-size:.69rem; font-weight:800; color:#5b88a5; letter-spacing:.04em;}
    .flow-step-title {font-size:.94rem; font-weight:800; color:#2e526b; margin:.18rem 0 .42rem;}
    .flow-step-text {font-size:.75rem; line-height:1.48; color:#536774;}
    .flow-step-result {font-size:.80rem; font-weight:800; color:#315b78; margin-top:.50rem;}
    .flow-step-arrow {text-align:center; color:#7ea6c4; font-size:1.12rem; margin:.20rem 0;}
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
    .environment-panel-marker {display:none;}
    div[data-testid="stColumn"]:has(.environment-panel-marker) {
        background:#edf5fb; border:1px solid #c9ddeb; border-left:5px solid #6f9fbe;
        border-radius:12px; padding:.90rem 1.00rem 1.00rem 1.00rem;
        min-height:0; align-self:flex-start; height:fit-content;
    }
    .environment-title {font-size:.96rem; font-weight:800; color:#365f79; margin:0 0 .55rem 0; padding:0; text-align:center;}
    .combined-card {border-radius:12px; padding:1.05rem 1.15rem; text-align:center; min-height:142px;}
    .combined-card.asd {background:#fff3e4; border:1px solid #f0d2ad; border-top:5px solid #e3a15a; color:#6b4521;}
    .combined-card.env {background:#edf5fb; border:1px solid #c9ddeb; border-top:5px solid #6f9fbe; color:#365f79;}
    .combined-card-label {font-size:.84rem; font-weight:800; margin-bottom:.45rem;}
    .combined-card-score {font-size:1.75rem; font-weight:900; line-height:1.1;}
    .combined-card-band {font-size:.82rem; font-weight:800; margin-top:.35rem;}
    .combined-result {background:#f5f8fb; border:1px solid #d5e2ee; border-top:5px solid #7f9db6; border-radius:12px; padding:1.1rem 1.2rem; text-align:center; margin:.8rem auto .2rem auto; max-width:720px; color:#2f485c;}
    .combined-result-title {font-size:.82rem; font-weight:800; color:#647a8d; margin-bottom:.40rem;}
    .combined-result-value {font-size:1.35rem; font-weight:900; margin-bottom:.42rem;}
    .combined-result-text {font-size:.86rem; line-height:1.55;}
    .criteria-wrap {margin:.90rem 0 .30rem;}
    .criteria-heading {font-size:.98rem; font-weight:850; color:#2f485c; margin:0 0 .48rem;}
    .criteria-panel {background:#fff; border:1px solid #dce4ec; border-radius:10px; padding:.72rem .78rem; height:100%;}
    .criteria-panel-title {font-size:.80rem; font-weight:850; color:#3e627b; margin-bottom:.45rem;}
    .criteria-row {border:1px solid #e1e7ec; border-radius:7px; padding:.34rem .44rem; margin:.24rem 0; font-size:.72rem; line-height:1.38; color:#4b606f;}
    .criteria-row b {color:#2f485c;}
    .criteria-row.low, .criteria-row.mid, .criteria-row.high, .criteria-row.vhigh {background:#ffffff; border-color:#e1e7ec;}
    .criteria-row.current {background:#eef8f1; border-color:#a8d3b4;}
    .combine-algorithm {background:#f5f8fb; border:1px solid #d5e2ee; border-radius:10px; padding:.76rem .85rem; margin-top:.65rem; text-align:center;}
    .combine-algorithm-title {font-size:.80rem; font-weight:850; color:#315b78; margin-bottom:.44rem;}
    .combine-grid {display:grid; grid-template-columns:1fr 1fr; gap:.34rem; text-align:left;}
    .combine-cell {background:#fff; border:1px solid #dce4ec; border-radius:7px; padding:.38rem .46rem; font-size:.72rem; color:#4b606f;}
    .combine-cell.active {background:#e8f3fb; border-color:#8fb4cd; color:#2f5c78; font-weight:750;}
    div[data-testid="stDataFrame"], div[data-testid="stPyplot"] {margin-top:.55rem; margin-bottom:.50rem;}
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
ART = APP_DIR / "model_artifacts" / "asd"
DATA_PATH = APP_DIR / "data" / "asd" / "Autism-Child-Data.csv"


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

# NSCH artifacts (separate from the existing ASD pipeline)
NSCH_ART = APP_DIR / "model_artifacts" / "nsch"
NSCH_ASD_ART = APP_DIR / "model_artifacts" / "nsch_asd"
NSCH_ASD_META = (json.loads((NSCH_ASD_ART / "final_model_metadata.json").read_text(encoding="utf-8")) if (NSCH_ASD_ART / "final_model_metadata.json").exists() else {})
NSCH_ASD_FEATURES = NSCH_ASD_META.get("final_features", [])
NSCH_FEATURES = (pd.read_csv(APP_DIR / "model_artifacts" / "nsch" / "selected_features.csv")["column"].tolist() if (APP_DIR / "model_artifacts" / "nsch" / "selected_features.csv").exists() else ["family_r", "fpl_i1", "foodsit", "screentime", "sc_age_years", "missmortgage", "a1_grade", "k7q04r_r", "k7q82_r", "k10q41_r", "makefriend", "hoursleep"])
NSCH_LABELS = {
    "family_r": "가족 형태", "fpl_i1": "가구 빈곤수준", "foodsit": "가구 식품 안정성",
    "screentime": "화면 사용 시간", "sc_age_years": "아동 연령",
    "missmortgage": "주거비·임대료 납부 곤란", "a1_grade": "보호자 1의 최종 학력",
    "k7q04r_r": "학교가 문제로 가정에 연락한 횟수", "k7q82_r": "학교에서 잘하고 싶은 마음",
    "k10q41_r": "학교가 안전하다는 인식", "makefriend": "친구를 사귀거나 유지하는 어려움",
    "hoursleep": "수면 시간",
    "sc_sex": "아동 성별", "sc_race_r": "아동 인종·민족",
    "k2q35a": "자폐 스펙트럼 장애 진단 경험", "k2q31a": "ADHD 진단 경험",
    "k2q33a": "불안 진단 경험", "k2q32a": "우울 진단 경험", "k2q34a": "행동문제 진단 경험",
    "a1_menthealth": "보호자 1의 정신·정서 건강", "a2_menthealth": "보호자 2의 정신·정서 건강",
    "a1_employed_r": "보호자 1의 고용 상태", "a2_employed_r": "보호자 2의 고용 상태",
    "k10q40_r": "아동의 동네 안전 인식", "k10q22": "동네의 노후·불량 주택",
    "k10q23": "동네의 기물파손", "moves": "아동의 이사 횟수",
}
NSCH_CODE_OPTIONS = {
    "family_r": [(1, "생물학적/입양 부모 2명·현재 결혼"), (2, "생물학적/입양 부모 2명·현재 미혼"), (3, "부모 2명·현재 결혼"), (4, "부모 2명·현재 미혼"), (5, "싱글 어머니"), (6, "싱글 아버지"), (7, "조부모 가정"), (8, "기타 관계")],
    "foodsit": [(1, "항상 영양가 있는 식사를 감당"), (2, "충분히 먹지만 원하는 종류는 항상 아님"), (3, "때때로 충분한 식사를 감당하지 못함"), (4, "자주 충분한 식사를 감당하지 못함")],
    "missmortgage": [(1, "예"), (2, "아니오"), (3, "모름")],
    "a1_grade": [(1, "8학년 이하"), (2, "9~12학년·졸업장 없음"), (3, "고등학교 졸업/GED"), (4, "직업·기술·상업학교"), (5, "대학 학점·학위 없음"), (6, "준학사"), (7, "학사"), (8, "석사"), (9, "박사·전문학위")],
    "k7q04r_r": [(1, "없음"), (2, "1회"), (3, "2회 이상")],
    "k7q82_r": [(1, "항상"), (2, "대부분"), (3, "때때로"), (4, "전혀 아님")],
    "k10q41_r": [(1, "확실히 동의"), (2, "어느 정도 동의"), (3, "어느 정도 반대"), (4, "확실히 반대")],
    "makefriend": [(3, "많이 어려움"), (2, "조금 어려움"), (1, "어려움 없음")],
    "a1_employed_r": [(1, "전일제 고용"), (2, "시간제 고용"), (3, "무급 근무"), (4, "구직 중"), (5, "구직하지 않는 미고용"), (6, "은퇴")],
    "k10q40_r": [(1, "확실히 동의"), (2, "어느 정도 동의"), (3, "어느 정도 반대"), (4, "확실히 반대")],
    "screentime": [(1, "1시간 미만"), (2, "1시간"), (3, "2시간"), (4, "3시간"), (5, "4시간 이상")],
    "hoursleep": [(1, "6시간 미만"), (2, "6시간"), (3, "7시간"), (4, "8시간"), (5, "9시간"), (6, "10시간"), (7, "11시간 이상")],
}

NSCH_SURVEY_QUESTIONS = {
    "makefriend": "아이가 친구를 사귀거나 유지하는 데 어려움이 있습니까?",
    "screentime": "평일에 아이가 TV·휴대폰·컴퓨터 화면을 사용하는 시간은 어느 정도입니까?",
    "k7q82_r": "아이는 학교에서 잘하고 싶어 하는 모습을 얼마나 보입니까?",
    "k7q04r_r": "최근 12개월 동안 아이의 문제로 학교에서 가정에 연락한 적이 있습니까?",
    "a1_employed_r": "아이의 보호자(주 보호자)의 현재 고용 상태는 무엇입니까?",
    "a1_grade": "아이의 보호자(주 보호자)의 최종 학력은 어디에 해당합니까?",
    "family_r": "아이와 함께 사는 가족의 형태는 무엇입니까?",
    "foodsit": "최근 12개월 동안 아이 가정의 식품 상황은 어떠했습니까?",
    "k10q40_r": "아이는 사는 동네가 안전하다고 생각합니까?",
}

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
    "age": "나이",
    "gender": "성별",
    "ethnicity": "인종·민족",
    "jaundice": "황달 이력",
    "family_asd": "가족의 ASD 이력",
    "country_of_res": "거주 국가",
    "used_app_before": "이전 선별 앱 사용 여부",
    "relation": "응답자와 아동의 관계",
}
MODEL_LABELS = {
    "Logistic Regression": "로지스틱 회귀",
    "KNN": "K-최근접 이웃",
    "Decision Tree": "의사결정나무",
    "Random Forest": "랜덤 포레스트",
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


NSCH_PIPELINE = [
    "데이터 정리",
    "연관성 분석",
    "요인 선정",
    "모델 비교",
    "성능 평가",
    "점수 산출",
]


def nsch_pipeline(active):
    html = []
    for i, step in enumerate(NSCH_PIPELINE):
        css = "flow-step-selected" if i == active else "flow-step"
        html.append(f'<div class="{css}">{i+1}. {step}</div>')
        if i < len(NSCH_PIPELINE) - 1:
            html.append('<div class="flow-arrow">→</div>')
    st.markdown(f'<div class="flow-wrap">{"".join(html)}</div>', unsafe_allow_html=True)


def render_nsch_asd_page():
    """NSCH-ASD results, rendered with the existing 2~6 page design system."""
    target = pd.read_csv(NSCH_ASD_ART / "target_summary.csv").set_index("metric")["value"]
    candidates = pd.read_csv(NSCH_ASD_ART / "candidate_features.csv")
    stats = pd.read_csv(NSCH_ASD_ART / "statistical_tests.csv")
    selection = pd.read_csv(NSCH_ASD_ART / "final_selection.csv")
    comparison = pd.read_csv(NSCH_ASD_ART / "model_validation_comparison.csv")
    final = pd.read_csv(NSCH_ASD_ART / "final_test_metrics.csv").iloc[0]
    split_summary = pd.read_csv(NSCH_ASD_ART / "split_summary.csv")
    threshold = float(NSCH_ASD_META.get("validation_threshold", .5))
    model_names = {"LogisticRegression": "로지스틱 회귀", "KNN": "K-최근접 이웃", "DecisionTree": "의사결정나무", "RandomForest": "랜덤 포레스트"}
    selected = selection[selection["selected_final"]].copy()
    chosen_model = str(final["model"])
    validation_row = comparison[comparison["model"].eq(chosen_model)].iloc[0]
    significant_count = int(stats["significant_0_05"].sum())
    split_counts = split_summary.groupby("split")["count"].sum().to_dict()

    page_title("8. NSCH 외부데이터 분석", "NSCH 2024 자료에서 현재 ASD 여부와 함께 나타나는 환경·생활 특성을 분석했습니다.")
    tabs = st.tabs(["8-1. 데이터 확인 및 전처리", "8-2. 연관성 분석", "8-3. 환경·생활 요인 선정", "8-4. 머신러닝 모델 비교", "8-5. 모델 성능 평가", "8-6. 생활환경 점수 산출"])

    with tabs[0]:
        nsch_pipeline(0)
        st.subheader("8-1. 데이터 확인 및 전처리")
        st.caption("NSCH 2024 데이터에서 현재 ASD 여부를 확인하고 환경·생활 특성 분석에 필요한 데이터를 정리했습니다.")
        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1: metric_card("전체 아동", f"{int(target['raw_rows']):,}명")
        with c2: metric_card("전체 설문항목", f"{int(target['raw_columns']):,}개")
        with c3: metric_card("현재 ASD 있음", f"{int(target['current_asd_yes']):,}명")
        with c4: metric_card("현재 ASD 없음", f"{int(target['current_asd_no']):,}명")
        st.subheader("분석 데이터 정리")
        prep_table = pd.DataFrame([
            ["현재 ASD 여부", f"{int(target['usable_current_asd']):,}명", "유효응답 아동만 분석에 사용"],
            ["결측·무응답", f"{int(target['excluded_target_invalid']):,}명", "현재 ASD 여부를 정할 수 없어 제외"],
            ["ID·조사관리 변수", "분석 입력 제외", "개별 분류와 직접 관련 없는 관리 정보"],
            ["진단 후 정보", "분석 입력 제외", "결과를 미리 알려줄 수 있는 정보 제외"],
            ["환경·생활 후보", f"{len(candidates)}개", "가족·경제·주거·학교·생활습관 중심으로 선별"],
        ], columns=["항목", "확인 결과", "처리 방법"])
        left_table(prep_table, .84)
        st.markdown(f'''<div class="test-flow">
        <div class="test-flow-box">전체 {int(target['raw_columns']):,}개 항목</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">불필요한 항목 제외</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">환경·생활 후보 {len(candidates)}개</div>
        </div>''', unsafe_allow_html=True)
        mini_note("현재 ASD 여부는 분석 결과로만 사용하고, 환경·생활 특성만 모델 입력 후보로 사용했습니다.")

    with tabs[1]:
        nsch_pipeline(1)
        st.subheader("8-2. ASD 여부와 환경·생활 특성의 연관성 확인")
        h1, h2 = st.columns(2, gap="small")
        with h1:
            st.markdown('<div class="hypothesis-card"><b>귀무가설 H0</b><br>현재 ASD 여부와 해당 환경·생활 요인은 관계가 없다.</div>', unsafe_allow_html=True)
        with h2:
            st.markdown('<div class="hypothesis-card"><b>대립가설 H1</b><br>현재 ASD 여부와 해당 환경·생활 요인은 관계가 있다.</div>', unsafe_allow_html=True)
        st.subheader("검정 방법")
        method_table = pd.DataFrame([
            ["범주형 항목 ↔ 현재 ASD 있음/없음", "카이제곱 검정 · Cramér's V"],
            ["순서형·숫자형 항목 ↔ 현재 ASD 있음/없음", "Spearman 순위상관분석"],
        ], columns=["변수 형태", "사용한 검정"])
        left_table(method_table, .82)
        st.markdown('''<div class="test-flow">
        <div class="test-flow-box">분석 항목</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">통계검정</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">P-value 계산</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">관계 크기 확인</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">관련성 판단</div></div>''', unsafe_allow_html=True)
        mini_note("p &lt; 0.05이면 귀무가설을 기각하고, p ≥ 0.05이면 귀무가설을 기각하지 못합니다.")
        stats_view = stats.copy()
        stats_view["검정방법"] = stats_view["test_method"].replace({"Chi-square / Cramér's V": "카이제곱 검정 · Cramér's V", "Spearman rank correlation": "Spearman 순위상관분석"})
        stats_view["검정통계량 또는 관계크기"] = stats_view["effect_size"].map(lambda x: f"{float(x):.3f}")
        stats_view["p-value"] = stats_view["p_value"].map(lambda x: "< 0.001" if float(x) < .001 else f"{float(x):.3f}")
        stats_view["판정"] = np.where(stats_view["significant_0_05"], "귀무가설 기각", "기각하지 못함")
        stats_view = stats_view[["korean_name", "domain", "검정방법", "검정통계량 또는 관계크기", "p-value", "판정"]].rename(columns={"korean_name": "분석 항목", "domain": "영역"})
        left_table(stats_view, .94, 500)
        result_line(f"전체 후보 {len(candidates)}개 중 <b>{significant_count}개</b>에서 통계적으로 관련성이 확인되었습니다.")
        mini_note("p-value만으로 최종 항목을 정하지 않고, 관계 크기·랜덤 포레스트 변수 중요도·설문 활용 가능성을 함께 확인했습니다.")

    with tabs[2]:
        nsch_pipeline(2)
        st.subheader("8-3. 최종 환경·생활 요인 선정")
        st.markdown('''<div class="test-flow">
        <div class="test-flow-box">통계적 관계</div><div class="test-flow-arrow">+</div>
        <div class="test-flow-box">관계 크기</div><div class="test-flow-arrow">+</div>
        <div class="test-flow-box">Random Forest 변수 중요도</div><div class="test-flow-arrow">+</div>
        <div class="test-flow-box">설문 활용 가능성</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">최종 {len(selected)}개 항목</div></div>''', unsafe_allow_html=True)
        selected_plot = selected.sort_values("rf_importance", ascending=True)
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        bars = ax.barh(selected_plot["korean_name"], selected_plot["rf_importance"], color="#88b895")
        for bar, value in zip(bars, selected_plot["rf_importance"]):
            ax.text(float(value) + .002, bar.get_y() + bar.get_height()/2, f"{float(value):.3f}", va="center", fontsize=7)
        ax.set_xlabel("랜덤 포레스트 변수 중요도", fontsize=8)
        ax.set_title("ASD 분류에 중요하게 사용된 환경·생활 특성", fontsize=10)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        left_plot(fig, .50)
        selected_view = selected.sort_values("rf_rank")[["rf_rank", "korean_name", "domain", "effect_size", "rf_importance", "selected_final"]].copy()
        selected_view["effect_size"] = selected_view["effect_size"].map(lambda x: f"{float(x):.3f}")
        selected_view["rf_importance"] = selected_view["rf_importance"].map(lambda x: f"{float(x):.3f}")
        selected_view["selected_final"] = "선정"
        selected_view.columns = ["순위", "환경·생활 항목", "영역", "통계 관계", "변수 중요도", "선정 여부"]
        left_table(selected_view, .94, 360)
        with st.expander("전체 후보 결과 보기"):
            all_view = selection[["korean_name", "domain", "effect_size", "rf_importance", "selected_final"]].copy()
            all_view["selected_final"] = all_view["selected_final"].map({True: "선정", False: "미선정"})
            all_view.columns = ["분석 항목", "영역", "통계 관계", "변수 중요도", "선정 여부"]
            st.dataframe(all_view, hide_index=True, width="stretch")
        result_line(f"통계적 관계와 머신러닝 중요도를 함께 확인하여 최종 <b>{len(selected)}개</b>의 생활환경 설문 항목을 선정했습니다.")

    with tabs[3]:
        nsch_pipeline(3)
        st.subheader("8-4. 머신러닝 모델 비교")
        st.caption("선정된 환경·생활 항목을 동일하게 사용하여 4개의 분류모델을 같은 조건에서 비교했습니다.")
        plain_list([
            ("왜 분류인가?", "현재 ASD 여부가 있음 / 없음의 두 범주이기 때문"),
            ("입력", f"최종 환경·생활 항목 {len(selected)}개"),
            ("비교 모델", "로지스틱 회귀 / K-최근접 이웃 / 의사결정나무 / 랜덤 포레스트"),
            ("평가", "정확도 / 정밀도 / 재현율 / F1 점수 / ROC-AUC"),
        ])
        model_view = comparison[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
        model_view["model"] = model_view["model"].map(model_names)
        for column in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            model_view[column] = model_view[column].map(lambda x: f"{float(x):.3f}")
        model_view.columns = ["모델", "정확도", "정밀도", "재현율", "F1 점수", "ROC-AUC"]
        left_table(model_view, .94)
        plot_df = comparison.sort_values("f1", ascending=True)
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        colors = ["#88b895" if name == chosen_model else "#c8d1d8" for name in plot_df["model"]]
        bars = ax.barh([model_names[name] for name in plot_df["model"]], plot_df["f1"], color=colors)
        ax.set_xlabel("Validation F1", fontsize=8)
        ax.set_xlim(0, 1.0)
        for bar, value in zip(bars, plot_df["f1"]):
            ax.text(float(value) + .01, bar.get_y() + bar.get_height()/2, f"{float(value):.3f}", va="center", fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        center_plot(fig, .38)
        result_line(f"Validation 기준 최종 선택 모델: <b>{model_names[chosen_model]}</b> &nbsp; | &nbsp; F1 {float(validation_row['f1']):.3f} / Recall {float(validation_row['recall']):.3f} / ROC-AUC {float(validation_row['roc_auc']):.3f}")

    with tabs[4]:
        nsch_pipeline(4)
        st.subheader("8-5. 최종 모델 성능 평가")
        st.markdown(f'''<div class="split-diagram">
            <div class="split-root-row"><div class="split-box root"><strong>전체 분석 데이터 {int(target['usable_current_asd']):,}명</strong><br>현재 ASD 여부 유효응답</div></div>
            <div class="split-branch-arrows"><span>↙</span><span>↓</span><span>↘</span></div>
            <div class="split-branches">
                <div class="split-box train"><strong>Train · {int(split_counts.get('train', 0)):,}명</strong><br>모델 학습</div>
                <div class="split-box train"><strong>Validation · {int(split_counts.get('validation', 0)):,}명</strong><br>모델 선택·기준 설정</div>
                <div class="split-box test"><strong>Test · {int(split_counts.get('test', 0)):,}명</strong><br>최종 성능 평가</div>
            </div></div>''', unsafe_allow_html=True)
        st.markdown('<div style="height:.70rem"></div>', unsafe_allow_html=True)
        st.subheader("성능평가 결과")
        metric_cols = st.columns(5, gap="small")
        for column, label, key in zip(metric_cols, ["정확도", "정밀도", "재현율", "F1 점수", "ROC-AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            with column: metric_card(label, f"{float(final[key])*100:.1f}%")
        metric_table = pd.DataFrame([
            ["정확도", final["accuracy"], "전체 응답 중 올바르게 구분한 비율"],
            ["정밀도", final["precision"], "있음으로 분류한 응답의 일치 비율"],
            ["재현율", final["recall"], "현재 ASD 있음 응답을 찾아낸 비율"],
            ["F1 점수", final["f1"], "정밀도와 재현율의 균형"],
            ["ROC-AUC", final["roc_auc"], "있음/없음을 전반적으로 구분하는 정도"],
        ], columns=["평가 지표", "Test 결과", "설명"])
        metric_table["Test 결과"] = metric_table["Test 결과"].map(lambda x: f"{float(x):.3f}")
        left_table(metric_table, .82)
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        labels = ["정확도", "정밀도", "재현율", "F1", "ROC-AUC"]
        bars = ax.bar(labels, [float(final[key]) for key in keys], color="#88b895")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Test 성능", fontsize=8); ax.tick_params(labelsize=8)
        for bar, value in zip(bars, [float(final[key]) for key in keys]):
            ax.text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.3f}", ha="center", fontsize=8)
        fig.tight_layout(); left_plot(fig, .52)
        result_line("환경·생활 특성만으로 현재 ASD 여부를 얼마나 구분할 수 있는지 최종 Test 자료에서 확인했습니다.")

    with tabs[5]:
        nsch_pipeline(5)
        st.subheader("8-6. 생활환경 관찰점수 산출")
        st.markdown('''<div class="test-flow">
        <div class="test-flow-box">설문 응답</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">결측값 처리</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">표준화 / 원-핫 인코딩</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">최종 모델</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">모델 점수 × 100</div><div class="test-flow-arrow">→</div>
        <div class="test-flow-box">생활환경 관찰점수</div></div>''', unsafe_allow_html=True)
        plain_list([
            ("1. 설문 응답 입력", "최종 환경·생활 항목의 응답을 입력"),
            ("2. 입력 전처리", "숫자형은 중앙값 처리·표준화, 선택지형은 최빈값 처리·원-핫 인코딩"),
            ("3. 모델 결과", f"{model_names[chosen_model]}의 predict_proba 결과를 0~100점으로 표시"),
        ])
        input_view = selected.sort_values("rf_rank")[["rf_rank", "korean_name", "domain", "column"]].copy()
        input_view["입력 형태"] = np.where(input_view["column"].eq("fpl_i1"), "숫자형", "선택지형")
        input_view = input_view[["rf_rank", "korean_name", "domain", "입력 형태"]]
        input_view.columns = ["순위", "설문 항목", "영역", "입력 형태"]
        left_table(input_view, .88, 360)
        st.markdown('<div class="formula-box"><b>계산 예시</b><br>설문 응답 → 모델 입력 변환 → predict_proba 출력값 0.63 → 생활환경 관찰점수 63점</div>', unsafe_allow_html=True)
        result_line("선택지 번호를 단순 합산한 점수가 아니라, 학습된 머신러닝 모델이 전체 응답을 함께 계산한 결과입니다.")


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


def learning_cards(what, why, how, result):
    """Compact, presentation-friendly explanation cards for the NSCH analysis."""
    cards = st.columns(4, gap="small")
    for column, title, body in zip(
        cards,
        ["무엇을 했나", "왜 했나", "어떻게 했나", "결과"],
        [what, why, how, result],
    ):
        with column:
            st.markdown(
                f"<div class='learning-card'><div class='learning-card-title'>{title}</div>{body}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)


def flow_step_card(step, title, text, result, final=False):
    final_class = " final" if final else ""
    st.markdown(
        f"<div class='flow-step-card{final_class}'><div class='flow-step-number'>STEP {step}</div>"
        f"<div class='flow-step-title'>{title}</div><div class='flow-step-text'>{text}</div>"
        f"<div class='flow-step-result'>{result}</div></div>",
        unsafe_allow_html=True,
    )


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
        <div class="report-score">{int(score)} / 100점</div>''',
        unsafe_allow_html=True,
    )
    if title:
        st.markdown(f'<div class="report-band">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-comment"><b>{who}</b> {comment}</div>', unsafe_allow_html=True)


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
        (high_cutoff, very_high_cutoff - 1, "ASD 선별 고관찰", "관찰 내용을 아이의 보호자와 공유하고 반복 양상을 함께 확인"),
        (very_high_cutoff, 100, "ASD 선별 매우 고관찰", "보호자와 협의하여 행동 특성을 조금 더 자세히 기록"),
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
        "8. NSCH 외부데이터 분석",
    ],
    label_visibility="collapsed",
)


# ============================================================
# 6. Page 1 - 프로젝트 개요
# ============================================================
if menu.startswith("1."):
    page_title("아동학대 의심 예측 설문조사", "전처리부터 최종 활용까지 핵심 분석 결과를 한 화면에 요약한다.")
    st.markdown("<div class='explain-card'><b>두 단계 관찰 구조</b><br><br><b>기존 UCI 데이터</b> → 행동특성으로 ASD 선별 → <b>1차 행동특성 설문</b><br><br><b>NSCH 전체 아동 데이터</b> → 현재 ASD 있음/없음과 가족·경제·학교·생활습관 비교 → ASD와 함께 나타나는 환경·생활 특성 선정 → 머신러닝으로 재확인 → <b>2차 생활환경 설문</b><br><br><b>두 설문 결과</b> → 행동 관찰점수 + 환경·생활 특성 점수 → <b>종합 관찰 결과</b></div>", unsafe_allow_html=True)

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
                f'<div class="overview-kpi">정확도 {float(final_row.get("accuracy",0)):.2f} · '
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
    preprocess_style = (
        preprocess_tbl.style
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
        ])
    )
    left_table(preprocess_style, .84)
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
            ("분류 / 회귀", "분류 — 예측할 결과가 두 범주이며, 로지스틱 회귀는 이진 분류에 사용하는 모델"),
            ("비교 모델", "로지스틱 회귀(Logistic Regression) / K-최근접 이웃(KNN) / 의사결정나무(Decision Tree) / 랜덤 포레스트(Random Forest)"),
        ]
    )

    st.subheader("4개 모델 Validation 비교")
    if not model_compare.empty:
        tbl = model_compare[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
        tbl["model"] = tbl["model"].map(model_name)
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            tbl[c] = tbl[c].map(lambda x: f"{float(x):.3f}")
        tbl.columns = ["사용 모델", "정확도", "정밀도", "재현율", "F1 점수", "ROC-AUC"]

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
elif menu.startswith("8."):
    render_nsch_asd_page()
    st.stop()
    page_title("8. NSCH 외부데이터 분석", "가족·경제·주거·생활환경 항목으로 생활환경 위험 수준(ACE 0~1개 / 2개 이상)을 분석한 결과입니다.")
    nt = st.tabs(["1. 분석 대상", "2. 데이터 정리·후보 선정", "3. 데이터 분리·결측 처리", "4. 통계적 연관성", "5. 최종 12개 선정", "6. 모델 입력 변환", "7. 모델 비교", "8. 최종 모델 결과", "9. 변수 중요도"])
    with nt[0]:
        st.subheader("무엇을 분석했나: 생활환경 항목과 ACE 기반 생활환경 위험 수준")
        result_line("기존 ASD 행동 설문에 가족·경제·주거·생활환경 정보를 더해, 추가 관찰이 필요한 생활환경 패턴을 함께 살펴봅니다.")
        c1, c2, c3 = st.columns(3, gap="small")
        c1.metric("사용 데이터", "NSCH 2024")
        c2.metric("분석 대상", "48,042명")
        c3.metric("최종 분석 항목", "12개")
        st.markdown("<div class='explain-card'><b>ACE(아동기 부정적 경험) 9개 문항</b>은 가정폭력, 경제적 어려움, 가족 문제 등 성장 과정에서 경험할 수 있는 부정적인 생활환경을 의미합니다.<br><br>9개 문항의 경험 개수를 합산해 <b>0~1개는 생활환경 위험 낮음</b>, <b>2개 이상은 생활환경 위험 높음</b>으로 구분했습니다. 이 분석은 아동학대를 직접 판별하거나 의료적 진단을 내리는 것이 아니라, <b>생활환경·부정적 경험 추가관찰</b>에 활용하기 위한 분류 모델입니다.</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([["원본 데이터", "51,375행 × 457열", "NSCH 2024 전체 설문"], ["중복 데이터", "0행", "중복 없음"], ["생활환경 위험 수준을 판단할 수 있는 데이터", "48,042명", "정상 ACE 응답"], ["필수 설문 응답 부족으로 제외된 데이터", "3,333명", "정확한 분류가 어려운 응답"], ["초기 전체 컬럼", "457개", "건강·가족·경제·주거·학교 등"], ["1차 분석 항목", "27개", "목적 관련 변수"], ["최종 분석 항목", "12개", "검정·변수 중요도·해석 가능성"], ["예측할 결과", "생활환경 위험 그룹", "분석 항목에는 포함하지 않음"]], columns=["항목", "값", "설명"]), hide_index=True, width="stretch")
    with nt[1]:
        st.subheader("생활환경 위험 수준을 분석할 수 있도록 데이터를 어떻게 준비했나")
        st.markdown("<div class='formula-box'><b>NSCH 2024 원본</b><br>51,375명 × 457개 항목<br>│<br>├─ <b>행 정리</b> · 중복 확인 0명 → 필수 문항 응답 확인 → 분석 가능 48,042명 / 제외 3,333명<br>│<br>└─ <b>변수 정리</b> · 전체 457개 → 분석 불가 항목 제외 → 목적 기반 1차 분석 항목 27개 → 통계적 관계 확인 → 랜덤 포레스트 변수 중요도 → 최종 12개</div>", unsafe_allow_html=True)
        process_df = pd.DataFrame([
            ["1", "중복 응답 확인", "0건", "같은 응답이 반복되지 않았는지 확인"],
            ["2", "분석 대상 확정", "48,042명", "생활환경 위험 수준을 판단할 수 있는 응답만 사용"],
            ["3", "분석 항목 정리", "457개 → 27개", "프로젝트 목적 관련 영역을 먼저 정의한 뒤 후보 선정"],
        ], columns=["단계", "무엇을 했나", "결과", "왜 했나"])
        left_table(process_df, .94, 260)
        result_line("이 단계의 핵심은 <b>예측할 결과를 미리 알 수 있는 정보가 모델에 들어가지 않게 하는 것</b>입니다. ACE 문항 자체, ACE 개수, 조사 식별번호·표본설계 정보는 생활환경을 설명하는 입력값으로 사용하지 않았습니다.")
        st.subheader("왜 457개 전체 항목을 그대로 사용하지 않았나")
        exclusion_df = pd.DataFrame([
            ["식별·조사관리", "가구 ID, 조사번호 등", "개인의 생활환경을 설명하는 입력값이 아니기 때문"],
            ["표본설계", "가중치, 층화, 조사관리 정보", "설문조사 운영을 위한 정보이며 개인 분류용 입력값이 아니기 때문"],
            ["정답 누수", "ACE 개수를 직접 만드는 문항", "정답을 미리 알려주는 것과 같아 모델 평가가 왜곡되기 때문"],
            ["결측 과다", "정상 응답이 지나치게 적은 항목", "안정적인 학습과 해석이 어렵기 때문"],
            ["중복·대체", "같은 정보를 다른 형태로 저장한 항목", "같은 정보가 반복 입력되는 것을 막기 위해"],
            ["목적 관련성 낮음", "현재 생활환경 분석과 거리가 있는 설문", "분석 목적에 맞는 범위로 좁히기 위해"],
        ], columns=["제외 유형", "무엇을 제외했나", "왜 제외했나"])
        left_table(exclusion_df, .94, 300)
        result_line("457개 항목을 자동으로 줄인 것이 아닙니다. <b>프로젝트 목적과 관련 있는 영역을 먼저 정의</b>하고, 분석에 사용할 수 없는 항목을 제외한 뒤 27개를 1차 분석 후보로 선정했습니다.")
        st.subheader("목적 기반 1차 분석 후보 27개")
        candidate_rows = [
            ["아동 기본정보", "아동 연령", "연령에 따른 생활환경 경험 차이 확인"], ["아동 기본정보", "아동 성별", "기본 인구학적 특성 확인"], ["아동 기본정보", "아동 인종·민족", "사회적 맥락 확인"],
            ["건강·발달 특성", "자폐 스펙트럼 장애 진단 경험", "관련성 확인용 건강 특성"], ["건강·발달 특성", "ADHD 진단 경험", "관련성 확인용 건강 특성"], ["건강·발달 특성", "불안 진단 경험", "관련성 확인용 건강 특성"], ["건강·발달 특성", "우울 진단 경험", "관련성 확인용 건강 특성"], ["건강·발달 특성", "행동문제 진단 경험", "관련성 확인용 건강 특성"],
            ["가족·보호자 환경", "보호자 1의 정신·정서 건강", "보호자 환경 확인"], ["가족·보호자 환경", "보호자 2의 정신·정서 건강", "보호자 환경 확인"], ["가족·보호자 환경", "보호자 1의 고용 상태", "가구 고용 환경 확인"], ["가족·보호자 환경", "보호자 2의 고용 상태", "가구 고용 환경 확인"], ["가족·보호자 환경", "보호자 1의 최종 학력", "가구 교육 자원 확인"], ["가족·보호자 환경", "가족 형태", "가구 구성 맥락 확인"],
            ["경제 상황", "가구 빈곤수준", "경제적 자원 수준 확인"], ["경제 상황", "최근 12개월 가구 식품 상황", "식품 접근성 확인"], ["경제 상황", "주거비·임대료 납부 곤란", "주거비 부담 확인"],
            ["주거·동네 환경", "아동의 동네 안전 인식", "거주 지역 안전 확인"], ["주거·동네 환경", "동네의 노후·불량 주택", "주거 주변 환경 확인"], ["주거·동네 환경", "동네의 기물파손", "거주 지역 위험 신호 확인"], ["주거 안정성", "아동의 이사 횟수", "주거 안정성 확인"],
            ["생활습관", "수면 시간", "건강 행동 지표 확인"], ["생활습관", "화면 사용 시간", "일상 행동 지표 확인"],
            ["학교·사회생활", "학교가 안전하다는 인식", "학교 환경 안전 확인"], ["학교·사회생활", "학교가 문제로 가정에 연락한 횟수", "학교 적응 신호 확인"], ["학교·사회생활", "학교에서 잘하고 싶은 마음", "학교 참여·태도 확인"], ["학교·사회생활", "친구를 사귀거나 유지하는 어려움", "또래 관계 확인"],
        ]
        candidate_df = pd.DataFrame(candidate_rows, columns=["영역", "한국어 분석 항목", "선정 이유"])
        st.dataframe(candidate_df, hide_index=True, width="stretch", height=430)
    with nt[2]:
        st.subheader("학습·검증·최종평가 자료를 먼저 분리하고 결측값을 처리")
        st.markdown("<div class='explain-card'><b>무엇을 했나?</b><br>생활환경 위험 수준을 판단할 수 있는 48,042명을 학습 28,825명, 검증 9,608명, 최종평가 9,609명으로 나눴습니다.<br><br><b>왜 먼저 나눴나?</b><br>전체 자료를 보고 변수선정이나 결측 처리를 하면 최종평가 자료의 정보가 학습 과정에 들어가는 데이터 누수가 생길 수 있습니다. 그래서 분리한 뒤 학습자료에서만 기준을 정했습니다.</div>", unsafe_allow_html=True)
        st.markdown("<div class='overview-flow'><div class='overview-flow-node'>사용 가능 48,042명</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>학습 28,825명<br>모델 학습</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>검증 9,608명<br>모델 선택</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>최종평가 9,609명<br>마지막 성능 확인</div></div>", unsafe_allow_html=True)
        split_df = pd.DataFrame([["학습자료", "모델이 패턴을 배우는 자료"], ["검증자료", "여러 모델 중 가장 좋은 모델을 고르는 자료"], ["최종평가 자료", "모든 선택이 끝난 뒤 성능을 한 번 확인하는 자료"]], columns=["자료", "사용 목적"])
        left_table(split_df, .82, 180)
        st.subheader("결측값 처리")
        missing_df = pd.DataFrame([["수치형 항목", "학습자료 중앙값", "극단값의 영향을 덜 받는 대표값"], ["범주형 항목", "학습자료 최빈값", "가장 자주 응답된 범주로 일관되게 처리"], ["검증·최종평가 자료", "학습자료에서 정한 같은 기준", "평가 자료의 정보가 학습에 섞이지 않도록 방지"]], columns=["항목 형태", "처리 방법", "이유"])
        left_table(missing_df, .82, 180)
    with nt[3]:
        st.subheader("수업에서 배운 상관분석과 p-value로 생활환경 위험 수준과의 관계 확인")
        st.markdown("<div class='explain-card'><b>무엇을 검정했나?</b><br>27개 후보 각각이 <b>생활환경 위험 수준(ACE 0~1개 / 2개 이상)</b>과 관계가 없는지 확인했습니다. 귀무가설은 ‘이 분석 항목과 생활환경 위험 그룹 사이에는 관계가 없다’입니다.<br><br><b>어떤 방법을 썼나?</b><br>두 범주형 응답에는 Pearson 상관분석, 순서형·등급형·수치형 응답에는 Spearman 순위상관분석을 사용했습니다. p-value가 0.05보다 작으면 우연이라고 보기 어려워 귀무가설을 기각하고, 통계적으로 관련성이 있다고 판단합니다.<br><br><b>주의할 점</b><br>학습자료가 약 2만8천 명으로 크기 때문에 아주 작은 관계도 p-value가 작게 나올 수 있습니다. 따라서 p-value만으로 최종 항목을 결정하지 않았습니다.</div>", unsafe_allow_html=True)
        sd = pd.read_csv(NSCH_ART / "statistical_tests.csv"); sd["p_value_raw"] = pd.to_numeric(sd["p_value"], errors="coerce"); sd["분석 항목"] = sd["column"].map(NSCH_LABELS).fillna("기타 분석 항목"); sd["사용한 통계방법"] = sd["test_method"].replace({"Pearson (binary/point-biserial)": "Pearson 상관분석(두 범주형)", "Spearman (ordinal/numeric)": "Spearman 순위상관분석(순서형·수치형)"}); sd["상관계수"] = sd["statistic"].map(lambda x: f"{float(x):.3f}"); sd["p-value"] = sd["p_value_raw"].map(lambda x: "< 0.001" if float(x) < .001 else f"{float(x):.3f}"); sd["결과 해석"] = np.where(sd["p_value_raw"] >= .05, "통계적으로 유의하지 않음", np.where(sd["absolute_relationship_strength"] < .10, "유의하지만 관계가 매우 약함", np.where(sd["statistic"] > 0, "유의한 양의 관계", "유의한 음의 관계")))
        st.metric("통계적으로 유의한 항목", f"{int((sd['p_value_raw'] < .05).sum())}개")
        st.dataframe(sd[["분석 항목", "사용한 통계방법", "상관계수", "p-value", "결과 해석"]], hide_index=True, width="stretch", height=340)
        mini_note("상관계수는 +이면 위험 수준이 높아질수록 함께 증가하는 경향, -이면 반대 경향을 뜻합니다. 0에 가까울수록 관계가 매우 약합니다.")
        st.subheader("상관계수 절댓값이 큰 분석 항목")
        top_stat = sd.nlargest(8, "absolute_relationship_strength").sort_values("absolute_relationship_strength")
        fig, ax = plt.subplots(figsize=(5.4, 2.8))
        bars = ax.barh(top_stat["분석 항목"], top_stat["absolute_relationship_strength"], color="#86aac5")
        for bar, value in zip(bars, top_stat["absolute_relationship_strength"]):
            ax.text(float(value) + .005, bar.get_y() + bar.get_height()/2, f"{float(value):.3f}", va="center", fontsize=8)
        ax.set_xlabel("상관계수 절댓값", fontsize=8)
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        center_plot(fig, .60)
    with nt[4]:
        st.subheader("생활환경 위험 수준을 분류하기 위해 어떤 항목과 모델을 사용했나")
        result_line("1차 분석 항목 27개 중 ASD·ADHD·불안·우울·행동문제 및 보호자 정신건강 7개는 <b>연관성 확인용으로만 분리</b>했습니다. 나머지 20개 생활환경 항목에 통계적 관계와 랜덤 포레스트 중요도를 함께 적용해 최종 12개를 선정했습니다.")
        st.markdown("<div class='formula-box'><b>1차 분석 후보 27개</b><br>│<br>├─ 통계검정 · p-value + 상관계수 확인<br>├─ 랜덤 포레스트 · 위험 수준 분류에 중요하게 사용된 항목 확인<br>└─ 해석 가능성 · 실제 설문으로 이해·응답할 수 있는지 검토<br>　　　　↓<br><b>최종 분석 항목 12개</b></div>", unsafe_allow_html=True)
        sf = pd.read_csv(NSCH_ART / "selected_features.csv")["column"].tolist()
        variable_df = pd.DataFrame([[idx + 1, NSCH_LABELS.get(x, "기타 분석 항목")] for idx, x in enumerate(sf)], columns=["순서", "최종 분석 항목"])
        select_left, select_right = st.columns([.58, .42], gap="large")
        with select_left:
            st.dataframe(variable_df, hide_index=True, width="stretch", height=330)
        with select_right:
            st.markdown("<div class='explain-card'><b>선정 기준</b><br>① 생활환경과의 통계적 관계 확인<br>② 랜덤 포레스트가 분류에 중요하게 사용한 정도 확인<br>③ 일반 사용자가 이해할 수 있는 생활환경 의미인지 검토<br><br><b>제외한 항목</b><br>ASD·ADHD·불안·우울 등 건강·진단 관련 항목은 연관성 확인용으로만 분리했습니다.</div>", unsafe_allow_html=True)
        st.subheader("비교한 머신러닝 모델")
        model_method_df = pd.DataFrame([
            ["로지스틱 회귀", "여러 생활환경 항목을 함께 보고 위험 그룹일 가능성을 계산", "해석이 쉽고 기준 모델로 적합"],
            ["K-최근접 이웃", "응답 패턴이 비슷한 사례를 찾아 분류", "가까운 사례 기반 분류와 비교"],
            ["의사결정나무", "조건을 순서대로 나누어 분류", "직관적인 규칙 기반 분류와 비교"],
            ["랜덤 포레스트", "여러 의사결정나무의 결과를 합산", "복잡한 관계와 변수 중요도 확인"],
        ], columns=["모델", "어떻게 분류하나", "비교한 이유"])
        left_table(model_method_df, .94, 230)
        st.markdown("<div class='explain-card'><b>실제로 적용한 입력 처리</b><br>가족 형태·식품 안정성·주거비 납부 곤란·보호자 학력·학교 및 또래 관계처럼 선택지로 응답한 8개 항목은 <b>원-핫 인코딩</b>으로 처리했습니다. 가구 빈곤수준·아동 연령·화면 사용 시간·수면 시간 4개 수치형 항목은 <b>학습자료 기준 표준화</b>를 적용했습니다.<br><br>이후 네 모델을 같은 학습·검증 자료에 적용해 비교했습니다. 이 설명은 실제 학습 파이프라인과 일치합니다.</div>", unsafe_allow_html=True)
        cm = pd.read_csv(NSCH_ART / "model_validation_comparison.csv"); cm_display = cm.copy(); cm_display["model"] = cm_display["model"].replace({"LogisticRegression": "로지스틱 회귀", "KNN": "K-최근접 이웃", "DecisionTree": "의사결정나무", "RandomForest": "랜덤 포레스트"}); cm_display.columns = ["모델", "구분", "정확도", "정밀도", "재현율", "F1 점수", "ROC-AUC"]
    with nt[5]:
        st.subheader("최종 12개 설문 응답을 머신러닝 입력값으로 변환")
        st.markdown("<div class='explain-card'><b>무엇을 했나?</b><br>최종 12개 설문 응답은 숫자형과 범주형으로 나누어 변환한 뒤 네 모델에 같은 형태로 입력했습니다.<br><br><b>왜 변환했나?</b><br>선택지형 응답의 숫자 1·2·3·4는 크기가 아니라 답변 종류를 뜻합니다. 그대로 숫자로 넣으면 잘못된 크기 관계를 만들 수 있기 때문에 별도 상태로 변환했습니다.</div>", unsafe_allow_html=True)
        st.markdown("<div class='overview-flow'><div class='overview-flow-node'>최종 12개 설문</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>숫자형<br>표준화</div><div class='overview-flow-arrow'>+</div><div class='overview-flow-node'>범주형<br>원-핫 인코딩</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>4개 모델 입력</div></div>", unsafe_allow_html=True)
        transform_df = pd.DataFrame([
            ["숫자형 4개", "가구 빈곤수준·아동 연령·화면 사용 시간·수면 시간", "StandardScaler 표준화", "단위가 다른 숫자들의 크기를 비슷한 기준으로 맞춤"],
            ["범주형 8개", "가족 형태·식품 안정성·주거비 납부·보호자 학력·학교·또래 관계", "원-핫 인코딩", "선택지를 단순한 점수 크기로 취급하지 않음"],
        ], columns=["항목", "포함 내용", "변환 방법", "이유"])
        left_table(transform_df, .94, 190)
        st.markdown("<div class='formula-box'><b>원-핫 인코딩 예시</b><br>항상 → [1, 0, 0, 0] &nbsp; / &nbsp; 대부분 → [0, 1, 0, 0] &nbsp; / &nbsp; 때때로 → [0, 0, 1, 0] &nbsp; / &nbsp; 전혀 아님 → [0, 0, 0, 1]<br><br>따라서 ‘전혀 아님=4’가 ‘항상=1’보다 4배 위험하다고 계산하는 방식이 아닙니다.</div>", unsafe_allow_html=True)
    with nt[6]:
        st.subheader("4개 모델이 생활환경 위험 수준(낮음 / 높음)을 얼마나 잘 구분하는지 비교")
        st.markdown("<div class='explain-card'><b>무엇을 했나?</b><br>동일한 학습자료를 로지스틱 회귀, K-최근접 이웃, 의사결정나무, 랜덤 포레스트에 <b>각각 별도로</b> 학습시켰습니다. 모델들이 순서대로 연결된 것이 아니라, 같은 문제를 네 가지 방법으로 풀어 성능을 비교한 것입니다.<br><br><b>왜 검증자료에서 모델을 골랐나?</b><br>학습자료 성능만 보면 모델이 이미 본 응답을 외운 것처럼 높게 나올 수 있습니다. 그래서 학습에 쓰지 않은 검증자료에서 네 모델의 성능을 비교했습니다.<br><br><b>선택 기준</b><br>검증자료의 ROC-AUC를 먼저 비교하고, 동률 또는 근접한 경우 F1 점수를 함께 봤습니다. 실제 비교 결과 로지스틱 회귀가 ROC-AUC와 F1 점수 모두 가장 높아 최종 모델로 선택되었습니다.</div>", unsafe_allow_html=True)
        st.markdown("<div class='overview-flow'><div class='overview-flow-node'>동일한 학습자료</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>로지스틱 회귀</div><div class='overview-flow-node'>K-최근접 이웃</div><div class='overview-flow-node'>의사결정나무</div><div class='overview-flow-node'>랜덤 포레스트</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>검증 성능 비교</div><div class='overview-flow-arrow'>→</div><div class='overview-flow-node'>로지스틱 회귀 선택</div></div>", unsafe_allow_html=True)
        st.dataframe(cm_display, hide_index=True, width="stretch")
        st.caption("정확도는 전체적으로 맞힌 비율, 정밀도는 위험으로 분류한 경우의 정확성, 재현율은 실제 위험 그룹을 찾아낸 비율, F1 점수는 정밀도와 재현율의 균형, ROC-AUC는 두 그룹을 구분하는 전반적 능력입니다.")
        mini_note("이번 분석은 0/1 분류 문제이므로 연속값 예측에 쓰는 회귀 평가 지표 R²는 사용하지 않았습니다.")
        metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["정확도", "정밀도", "재현율", "F1 점수", "ROC-AUC"]
        graph_df = cm.copy(); graph_df["model"] = graph_df["model"].replace({"LogisticRegression": "로지스틱 회귀", "KNN": "K-최근접 이웃", "DecisionTree": "의사결정나무", "RandomForest": "랜덤 포레스트"})
        score_matrix = graph_df[metric_cols].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(3.3, 2.0))
        image = ax.imshow(score_matrix, cmap="Blues", vmin=0, vmax=1)
        ax.set_title("모델별 성능 요약", fontsize=10)
        ax.set_xticks(np.arange(len(metric_labels))); ax.set_xticklabels(metric_labels, rotation=32, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(graph_df))); ax.set_yticklabels(graph_df["model"], fontsize=7)
        for row in range(score_matrix.shape[0]):
            for col in range(score_matrix.shape[1]):
                ax.text(col, row, f"{score_matrix[row, col]:.2f}", ha="center", va="center", fontsize=6, color="#17364b" if score_matrix[row, col] < .72 else "white")
        fig.colorbar(image, ax=ax, fraction=.05, pad=.04).ax.tick_params(labelsize=6)
        fig.tight_layout(); center_plot(fig, .38)
        test_metrics = pd.read_csv(NSCH_ART / "final_test_metrics.csv").iloc[0]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5, gap="small")
        for col, label, key in zip([mc1, mc2, mc3, mc4, mc5], metric_labels, metric_cols): col.metric(label, f"{float(test_metrics[key]):.3f}")
        st.success("최종 선택 모델: 로지스틱 회귀 — 검증자료에서 ROC-AUC 0.882, F1 점수 0.588로 네 모델 중 가장 높은 종합 성능을 보였습니다.")
    with nt[7]:
        st.subheader("최종 선택 모델: 로지스틱 회귀")
        st.markdown("<div class='explain-card'><b>무엇을 했나?</b><br>검증자료에서 가장 좋은 성능을 보인 로지스틱 회귀를 선택한 뒤, 학습자료와 검증자료를 합쳐 다시 학습했습니다. 그 다음 모델 선택에 사용하지 않은 최종평가 자료 9,609명으로 성능을 한 번만 확인했습니다.<br><br><b>왜 로지스틱 회귀를 선택했나?</b><br>검증자료에서 ROC-AUC 0.882, F1 점수 0.588로 네 모델 중 가장 높은 종합 성능을 보였습니다. 특히 생활환경 위험 그룹을 놓치지 않는 것이 중요하므로 재현율도 함께 확인했습니다.</div>", unsafe_allow_html=True)
        final_metric_df = pd.DataFrame([
            ["정확도", "81.89%", "전체 응답 중 위험 수준을 맞게 구분한 비율"],
            ["정밀도", "47.73%", "위험 높음으로 분류한 응답 중 실제 위험 높음인 비율"],
            ["재현율", "82.16%", "실제 위험 높음 응답 중 모델이 찾아낸 비율"],
            ["F1 점수", "60.38%", "정밀도와 재현율의 균형"],
            ["ROC-AUC", "88.66%", "낮음·높음 두 그룹을 전반적으로 구분하는 능력"],
        ], columns=["평가 지표", "최종평가 결과", "의미"])
        left_table(final_metric_df, .88, 250)
        result_line("<b>결과:</b> 최종 모델은 위험 그룹을 실제로 찾아내는 재현율이 82.16%였습니다. 다만 이 결과는 생활환경 추가관찰을 위한 분류 성능이며, 의료 진단이나 실제 위험 확률을 뜻하지 않습니다.")
    with nt[8]:
        st.subheader("랜덤 포레스트가 위험 수준 분류에 중요하게 사용한 최종 12개 항목")
        st.markdown("<div class='explain-card'><b>무엇을 했나?</b><br>랜덤 포레스트 500개 나무가 20개 생활환경 항목을 검토할 때, 생활환경 위험 그룹을 나누는 데 상대적으로 많이 참고한 정도를 계산했습니다. 선택지형 항목은 여러 응답칸으로 나눈 뒤, 각 응답칸의 중요도를 다시 원래 항목 단위로 합산했습니다.<br><br><b>무엇을 의미하나?</b><br>변수 중요도는 어떤 항목이 랜덤 포레스트 분류 과정에서 상대적으로 많이 사용되었는지를 뜻합니다. 중요도가 높다고 해서 해당 항목이 생활환경 위험의 원인이라는 의미는 아닙니다.<br><br><b>결과 활용</b><br>이 중요도를 점수에 직접 더하지 않고, 통계적 관계와 함께 최종 12개 항목을 고르는 근거로 사용했습니다. 최종 생활환경 점수는 12개 항목을 넣은 <b>로지스틱 회귀 모델</b>으로 계산합니다.</div>", unsafe_allow_html=True)
        step1, step2, step3 = st.columns(3, gap="small")
        step1.metric("1차 분석 항목", "27개")
        step2.metric("건강·진단 항목 분리", "7개")
        step3.metric("최종 분석 항목", "12개")
        result_line("27개 1차 분석 항목 → 건강·진단 관련 7개는 연관성 확인용으로 분리 → 생활환경 20개에 통계검정과 랜덤 포레스트 적용 → 두 순위를 합산 → 최종 12개 선택")
        im = pd.read_csv(NSCH_ART / "rf_feature_importance.csv"); im = im[im["selected_final"].eq(True)].sort_values("rf_importance")
        labels = im["column"].map(lambda x: NSCH_LABELS.get(x, "기타 분석 항목"))
        fig, ax = plt.subplots(figsize=(3.6, 2.5))
        bars = ax.barh(labels, im["rf_importance"], color="#7ea6c4")
        for bar, value in zip(bars, im["rf_importance"]):
            ax.text(float(value) + .002, bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center", fontsize=7)
        ax.set_title("생활환경 및 아동 생활특성 분류에 중요하게 사용된 항목", fontsize=9)
        ax.set_xlabel("랜덤 포레스트 변수 중요도", fontsize=7); ax.set_ylabel("분석 항목", fontsize=7); ax.tick_params(labelsize=6)
        fig.tight_layout(); center_plot(fig, .45)
    st.stop()
    page_title("8. NSCH 외부데이터 분석")
    t = st.tabs(["분석 개요", "통계검정", "모델 비교", "변수 중요도"])
    with t[0]: st.dataframe(pd.DataFrame([["원본", "51,375행 × 457열"], ["ACE_HIGH 사용 가능", "48,042행"], ["후보 → 최종", "27개 → 12개"]], columns=["항목", "내용"]), hide_index=True)
    with t[1]: st.dataframe(pd.read_csv(NSCH_ART / "statistical_tests.csv"), hide_index=True, width="stretch")
    with t[2]: st.dataframe(pd.read_csv(NSCH_ART / "model_validation_comparison.csv"), hide_index=True, width="stretch"); st.dataframe(pd.read_csv(NSCH_ART / "final_test_metrics.csv"), hide_index=True, width="stretch")
    with t[3]:
        i = pd.read_csv(NSCH_ART / "rf_feature_importance.csv").sort_values("rf_importance"); fig, ax = plt.subplots(figsize=(7, 4)); ax.barh(i["column"], i["rf_importance"]); st.pyplot(fig, width="stretch"); plt.close(fig)
elif menu.startswith("9."):
    page_title("9. 생활환경 위험 설문")
    with (NSCH_ART / "final_model_nsch.pkl").open("rb") as f: m = pickle.load(f)
    v = {}
    for j, x in enumerate(NSCH_FEATURES):
        if x in NSCH_CODE_OPTIONS:
            o = NSCH_CODE_OPTIONS[x]; v[x] = st.selectbox(NSCH_LABELS[x], o, format_func=lambda z: z[1], key=f"n_{x}")[0]
        elif x == "fpl_i1": v[x] = st.number_input(NSCH_LABELS[x], min_value=50.0, max_value=400.0, value=100.0, step=1.0, key=f"n_{x}")
        elif x == "sc_age_years": v[x] = st.number_input(NSCH_LABELS[x], min_value=0, max_value=17, value=8, step=1, key=f"n_{x}")
        else: v[x] = st.number_input(NSCH_LABELS[x], min_value=0.0, value=8.0, step=.5, key=f"n_{x}")
    if st.button("생활환경 관찰점수 계산"): st.session_state["nsch_environment_score"] = float(m.predict_proba(pd.DataFrame([v], columns=NSCH_FEATURES))[0, 1] * 100); st.metric("생활환경 관찰점수", f"{st.session_state['nsch_environment_score']:.1f}점"); st.caption("진단 확률이 아닌 관찰용 지표입니다.")
elif menu.startswith("10."):
    page_title("10. 종합 결과")
    a = sum(int(r["points"]) for _, r in weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)].head(10).iterrows() if st.session_state.get(f"teacher_check_{r['feature']}", False)) if not weighted_checklist.empty else 0; e = float(st.session_state.get("nsch_environment_score", 0)); ah = a >= 65; eh = e >= 50
    st.metric("ASD 행동 관찰점수", a); st.metric("생활환경 관찰점수", f"{e:.1f}"); st.success("복합 추가관찰" if ah and eh else "ASD 행동 집중 관찰" if ah else "생활환경 집중 관찰" if eh else "일반 관찰")
elif menu.startswith("7."):
    page_title("7. 체크리스트")
    pipeline(6)
    st.markdown("### 체크리스트")
    survey_tabs = st.tabs(["ASD 행동 설문", "생활환경 설문", "종합 결과"])
    with survey_tabs[0]:
        asd_checklist_slot = st.empty()
    with survey_tabs[1]:
        env_form_col, env_result_col = st.columns([.57, .43], gap="large")
        with env_form_col:
            st.markdown('<span class="environment-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown('<div class="environment-title">생활환경 설문</div>', unsafe_allow_html=True)
            st.caption("아동센터 선생님이 관찰하거나 알고 있는 범위에서 답하는 8개 질문입니다. 각 선택지는 NSCH 2024 원래 응답 범주를 사용합니다.")
            with (NSCH_ASD_ART / "final_model_nsch_asd.pkl").open("rb") as f:
                nsch_ui_model = pickle.load(f)
            with st.form("integrated_nsch_form"):
                nsch_values = {"fpl_i1": np.nan, "makefriend": np.nan}
                visible_features = [x for x in NSCH_ASD_FEATURES if x not in {"fpl_i1", "makefriend"}]
                for number, x in enumerate(visible_features, start=1):
                    question = NSCH_SURVEY_QUESTIONS.get(x, NSCH_LABELS[x])
                    if x in NSCH_CODE_OPTIONS:
                        options = NSCH_CODE_OPTIONS[x]
                        nsch_values[x] = st.selectbox(
                            f"{number}. {question}", options,
                            format_func=lambda value: value[1], key=f"integrated_{x}",
                        )[0]
                submitted = st.form_submit_button("생활환경 설문 결과 확인", use_container_width=True)
            if submitted:
                st.session_state["nsch_environment_score"] = float(
                    nsch_ui_model.predict_proba(pd.DataFrame([nsch_values], columns=NSCH_ASD_FEATURES))[0, 1] * 100
                )
        with env_result_col:
            env_now = st.session_state.get("nsch_environment_score")
            st.markdown('<span class="environment-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown('<div class="environment-title">생활환경 설문 결과</div>', unsafe_allow_html=True)
            if env_now is not None:
                env_threshold = float(NSCH_ASD_META.get("validation_threshold", .5)) * 100
                env_level = "높음" if float(env_now) >= env_threshold else "낮음"
                st.markdown(f"<div class='combined-card env'><div class='combined-card-label'>🏡 환경·생활 특성 점수</div><div class='combined-card-score'>{float(env_now):.1f} / 100</div><div class='combined-card-band'>{env_level}</div></div>", unsafe_allow_html=True)
                st.caption(f"검증자료 기준 {env_threshold:.0f}점 이상은 ‘높음’으로 표시합니다.")
            else:
                st.markdown("<div class='explain-card'><b>결과 안내</b><br>왼쪽 8개 질문에 응답한 뒤 결과 확인 버튼을 누르면 점수가 표시됩니다.</div>", unsafe_allow_html=True)
    with survey_tabs[2]:
        env = float(st.session_state.get("nsch_environment_score", 0.0)); asd = sum(int(r["points"]) for _, r in weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)].head(10).iterrows() if st.session_state.get(f"teacher_check_{r['feature']}", False)) if not weighted_checklist.empty else 0; asd_pct = min(100.0, float(asd)); avg = (asd_pct + env) / 2
        high = asd >= int(meta.get("weighted_checklist", {}).get("high_cutoff", 65)); env_high = env >= float(NSCH_ASD_META.get("validation_threshold", .5)) * 100
        verdict = "복합 추가관찰" if high and env_high else "ASD 행동 집중 관찰" if high else "생활환경 집중 관찰" if env_high else "일반 관찰"
        asd_band = "높음" if high else "낮음"; env_band = "높음" if env_high else "낮음"
        verdict_text = "ASD 행동 특성과 환경·생활 특성이 모두 높게 관찰되었습니다." if high and env_high else "ASD 행동 특성은 높고 환경·생활 특성은 낮게 관찰되었습니다." if high else "환경·생활 특성은 높고 ASD 행동 특성은 낮게 관찰되었습니다." if env_high else "ASD 행동 특성과 환경·생활 특성이 모두 낮게 관찰되었습니다."
        observe_threshold = int(meta.get("weighted_checklist", {}).get("observe_cutoff", 50))
        asd_threshold = int(meta.get("weighted_checklist", {}).get("high_cutoff", 65))
        very_high_threshold = int(meta.get("weighted_checklist", {}).get("very_high_cutoff", 80))
        threshold_text = float(NSCH_ASD_META.get("validation_threshold", .5)) * 100
        environment_score_bands = pd.read_csv(NSCH_ASD_ART / "environment_score_bands.csv")
        env_row_classes = ["low", "mid", "high", "vhigh"]
        asd_current_index = 0 if asd < observe_threshold else 1 if asd < asd_threshold else 2 if asd < very_high_threshold else 3
        env_current_index = next(
            (i for i, (_, row) in enumerate(environment_score_bands.iterrows()) if env < float(row["upper_score"]) * 100 or i == len(environment_score_bands) - 1),
            len(environment_score_bands) - 1,
        )
        environment_band_rows = "".join(
            f"<div class='criteria-row {env_row_classes[i]}{' current' if i == env_current_index else ''}'><b>{float(row['lower_score'])*100:.1f}~{float(row['upper_score'])*100:.1f}점 · {row['score_band']}</b><br>검증자료 {int(row['validation_count']):,}명 · 현재 ASD 있음 {float(row['observed_asd_rate'])*100:.1f}%</div>"
            for i, (_, row) in enumerate(environment_score_bands.iterrows())
        )
        asd_criteria = f"{asd_threshold}점 이상" if high else f"{asd_threshold}점 미만"
        env_criteria = f"{threshold_text:.0f}점 이상" if env_high else f"{threshold_text:.0f}점 미만"
        card1, card2 = st.columns(2, gap="large")
        with card1:
            st.markdown(f"<div class='combined-card asd'><div class='combined-card-label'>🧩 ASD 행동점수</div><div class='combined-card-score'>{asd_pct:.0f} / 100</div><div class='combined-card-band'>{asd_band}</div></div>", unsafe_allow_html=True)
        with card2:
            st.markdown(f"<div class='combined-card env'><div class='combined-card-label'>🏡 환경·생활 특성 점수</div><div class='combined-card-score'>{env:.0f} / 100</div><div class='combined-card-band'>{env_band}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:1.4rem;color:#7f9db6;margin:.35rem 0'>↓</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='combined-result'><div class='combined-result-title'>종합 관찰 결과</div><div class='combined-result-value'>🟠 {verdict}</div><div class='combined-result-text'>{verdict_text}<br><br><b>{verdict} 기준</b><br>ASD 행동점수 {asd_pct:.0f}점 ({asd_criteria}) + 환경·생활 특성 점수 {env:.0f}점 ({env_criteria})</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-line'><b>종합 참고점수: {avg:.1f} / 100</b><br>ASD 행동점수와 환경·생활 특성 점수를 같은 비중으로 단순 평균한 학습용 참고 수치입니다.</div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class='criteria-wrap'>
            <div class='criteria-heading'>점수 구간과 종합 판정 기준</div>
            <div style='display:grid;grid-template-columns:1fr 1fr;gap:.55rem'>
              <div class='criteria-panel'>
                <div class='criteria-panel-title'>🧩 ASD 행동점수 구간</div>
                <div class='criteria-row low{' current' if asd_current_index == 0 else ''}'><b>0~{observe_threshold-1}점 · 일반 관찰</b><br>일상 행동 변화를 지속적으로 관찰</div>
                <div class='criteria-row mid{' current' if asd_current_index == 1 else ''}'><b>{observe_threshold}~{asd_threshold-1}점 · 추가 관찰</b><br>반복 행동의 빈도·상황·지속 여부를 기록</div>
                <div class='criteria-row high{' current' if asd_current_index == 2 else ''}'><b>{asd_threshold}~{very_high_threshold-1}점 · ASD 선별 고관찰</b><br>관찰 내용을 아이의 보호자와 공유</div>
                <div class='criteria-row vhigh{' current' if asd_current_index == 3 else ''}'><b>{very_high_threshold}~100점 · ASD 선별 매우 고관찰</b><br>행동 특성을 조금 더 자세히 기록</div>
              </div>
              <div class='criteria-panel'>
                <div class='criteria-panel-title'>🏡 환경·생활 특성 점수 구간</div>
                {environment_band_rows}
              </div>
            </div>
            <div class='combine-algorithm'>
              <div class='combine-algorithm-title'>두 설문을 합치는 공식과 근거</div>
              <div style='font-size:.76rem;line-height:1.65;color:#4b606f;text-align:left'>
                <b>종합 판정 = 2×2(ASD 행동점수 {asd_threshold}점 기준, 환경·생활 특성 점수 {threshold_text:.0f}점 기준)</b>입니다. 두 점수를 평균해 하나의 모델로 다시 계산하지 않고, 각 점수가 기준 이상인지 확인한 뒤 네 가지 조합으로 판정합니다.<br><br>
                ASD 행동점수는 기존 A1~A10 <b>로지스틱 회귀 계수 기반 가중점수</b>를 사용합니다. 환경·생활 특성 점수는 4개 모델을 비교해 선택한 <b>로지스틱 회귀</b>의 학습 계수로 계산합니다. 환경 점수의 {threshold_text:.0f}점 기준은 검증자료에서 F1 점수가 가장 높았던 지점입니다.<br><br>
                이 방식은 서로 다른 자료와 알고리즘으로 만든 두 점수 중 하나가 높을 때 평균 때문에 의미가 약해지는 것을 막고, 각 영역을 독립적으로 확인할 수 있어 프로젝트의 종합 관찰 목적에 적합합니다.
              </div>
            </div></div>""",
            unsafe_allow_html=True,
        )

    asd_checklist_container = asd_checklist_slot.container()
    asd_checklist_container.__enter__()
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

        st.subheader("ASD 행동 체크리스트")
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
                "관찰한 행동이 어떤 상황에서 반복되는지 계속 기록해 보세요."
            )
        elif total_score >= high_cutoff:
            result_title = "ASD 선별 고관찰 구간"
            result_comment = (
                "아동에게 여러 ASD 관련 행동 특성이 함께 관찰됩니다.<br>"
                "관찰 내용을 아이의 보호자와 공유하고 변화 양상을 함께 살펴보세요."
            )
        elif total_score >= observe_cutoff:
            result_title = "추가 관찰 구간"
            result_comment = (
                "아동에게 일부 ASD 관련 행동 특성이 관찰됩니다.<br>"
                "같은 행동이 특정 상황에서 반복되는지 기록하고, 변화가 지속되는지 "
                "조금 더 주의 깊게 살펴보는 것이 좋습니다."
            )
        else:
            result_title = ""
            result_comment = (
                "현재 설문에서 ASD 관련 행동 특성이 낮은 수준으로 관찰됩니다.<br>"
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
        st.markdown("이 점수 구간은 <b>프로젝트 데이터의 가중점수 분포를 바탕으로 만든 관찰 단계</b>입니다.", unsafe_allow_html=True)

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
    asd_checklist_container.__exit__(None, None, None)
