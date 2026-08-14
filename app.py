from pathlib import Path
import json
import pickle
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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
        max-width: 1340px;
        margin-left:0 !important;
        margin-right:auto !important;
        padding-left:4.60rem !important;
        padding-right:1.85rem !important;
        padding-top: 4.20rem !important;
        padding-bottom: 2.5rem;
    }
    section[data-testid="stSidebar"] {width: 16.5rem !important;}
    section[data-testid="stSidebar"] > div {width: 16.5rem !important;}

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
    .nsch-preprocess-result {
        background:#f4f8fc; border-left:3px solid #5a83ad; border-radius:6px;
        padding:.72rem 1rem; margin:.45rem 0 .75rem 0; color:#284b69;
        font-size:.93rem; line-height:1.55; width:96%; max-width:none;
        box-sizing:border-box;
    }
    .nsch-conclusion-result {font-size:.80rem; line-height:1.48; padding:.60rem .82rem;}
    .nsch-final-flow {max-width:610px; margin:.20rem auto .48rem; padding:.50rem .66rem; font-size:.78rem; line-height:1.42;}
    .nsch-final-flow .formula-arrow {font-size:.90rem; margin:.06rem 0;}
    .nsch-reduction-flow {
        width:96%; background:#f8fafc; border:1px solid #d9e4ee; border-radius:9px;
        padding:.65rem .9rem; margin:.45rem 0 .75rem 0; box-sizing:border-box;
        color:#315b78; font-size:.84rem; line-height:1.65;
    }
    .nsch-reduction-flow b {color:#24445c;}
    .nsch-learning-tree {
        position:sticky; top:4.15rem; background:#f7f9fb; border:1px solid #d6e0e8;
        border-radius:10px; padding:.88rem .86rem; margin:.15rem 0 .7rem; color:#556674;
        font-size:.72rem; line-height:1.48; box-sizing:border-box;
    }
    .nsch-learning-tree-title {font-size:.84rem; font-weight:850; color:#294b64; padding:.05rem .10rem .58rem; border-bottom:1px solid #dbe4eb; margin-bottom:.48rem;}
    .nsch-tree-stage {border-left:3px solid #d7e0e7; border-radius:0 7px 7px 0; padding:.34rem .38rem .38rem .52rem; margin:.22rem 0; transition:all .16s ease;}
    .nsch-tree-stage.active {background:#eaf3fb; border-left-color:#5d91b8; box-shadow:inset 0 0 0 1px #c7ddec;}
    .nsch-tree-main {font-size:.76rem; font-weight:800; color:#3a5264;}
    .nsch-tree-stage.active .nsch-tree-main {color:#1e5d8b;}
    .nsch-tree-sub {font-size:.68rem; color:#71808c; padding-left:.18rem; margin-top:.12rem;}
    .nsch-tree-stage.active .nsch-tree-sub {color:#4c6d84;}
    .nsch-tree-foot {font-size:.66rem; color:#82909a; border-top:1px solid #dbe4eb; padding:.50rem .10rem 0; margin-top:.55rem; line-height:1.45;}
    .nsch-final-heading {font-size:.94rem; font-weight:850; color:#2f485c; margin:.58rem 0 .40rem;}
    .nsch-final-table {width:100%; table-layout:fixed; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid #d7e3ec; border-radius:9px; font-size:.74rem; color:#425766; margin:.18rem 0 .48rem;}
    .nsch-final-table th {background:#f2f6f9; color:#496679; font-weight:650; text-align:center; padding:.52rem .38rem; border-bottom:1px solid #d7e3ec; white-space:normal; line-height:1.35;}
    .nsch-final-table td {padding:.48rem .45rem; border-bottom:1px solid #e7edf1; vertical-align:middle; line-height:1.42; word-break:keep-all;}
    .nsch-final-table tr:last-child td {border-bottom:0;}
    .nsch-final-table tr:nth-child(even) td {background:#fbfcfd;}
    .nsch-final-num {color:#6b879b; font-weight:500; text-align:center; width:5%;}
    .nsch-final-question {font-weight:400; color:#354f61; width:33%;}
    .nsch-final-response {color:#607785; width:18%;}
    .nsch-final-rate {text-align:center; color:#47677d; font-weight:500; width:12%; white-space:nowrap;}
    .nsch-final-gap {text-align:center; color:#4e8064; font-weight:500; width:10%; white-space:nowrap;}
    .nsch-final-weight {text-align:center; color:#365f79; font-weight:600; width:10%; white-space:nowrap;}
    .behavior-weight-table {width:100%; border-collapse:separate; border-spacing:0; border:1px solid #d7e3ec; border-radius:8px; overflow:hidden; font-size:.78rem; color:#425766;}
    .behavior-weight-table th, .behavior-weight-table td {text-align:center !important; vertical-align:middle; padding:.48rem .35rem; border-bottom:1px solid #e7edf1;}
    .behavior-weight-table th {background:#f2f6f9; color:#496679; font-weight:700; line-height:1.35;}
    .behavior-weight-table tr:last-child td {border-bottom:0;}
    .behavior-weight-table tr:nth-child(even) td {background:#fbfcfd;}
    @media (max-width: 900px) {
        .nsch-learning-tree {position:static; margin-bottom:.75rem;}
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
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        min-height:28px; align-items:center !important; margin:0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        margin:0 !important; line-height:1.25 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:nth-of-type(8) {
        border-top:1px solid #cfd7df; margin-top:1.44rem !important; padding-top:1.44rem !important;
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
    .project-flow-wrap {
        display:flex; align-items:stretch; gap:.34rem; overflow-x:auto;
        padding:.18rem .10rem .78rem; margin:.04rem 0 .42rem;
        scrollbar-color:#b9c8d5 transparent;
    }
    .project-flow-card {
        flex:1 0 122px; min-width:122px; min-height:112px;
        background:#f8fafc; border:1px solid #d9e3eb; border-top:4px solid #7ea6c4;
        border-radius:10px; padding:.72rem .70rem .62rem; box-sizing:border-box;
        color:#374957;
    }
    .project-flow-card.data {flex-basis:168px; min-width:168px;}
    .project-flow-card.combine {background:#f1f7fb; border-color:#c7dce9; border-top-color:#5d91b8;}
    .project-flow-card.final {background:#f4f8fc; border-color:#c7dce9; border-top-color:#4f7d9d;}
    .project-flow-kicker {font-size:.65rem; font-weight:800; letter-spacing:.045em; color:#6a8da6; margin-bottom:.28rem;}
    .project-flow-title {font-size:.83rem; font-weight:850; color:#2f526a; line-height:1.35; margin-bottom:.38rem; word-break:keep-all;}
    .project-flow-text {font-size:.70rem; color:#5d6f7d; line-height:1.48; word-break:keep-all;}
    .project-flow-arrow {flex:0 0 13px; align-self:center; color:#80a0b6; font-size:1.13rem; text-align:center; margin-top:-.05rem;}
    .project-flow-summary {font-size:.74rem; color:#617180; text-align:center; margin:-.10rem 0 .50rem; letter-spacing:.01em;}
    .project-flow-heading {font-size:.98rem; font-weight:850; color:#315b78; margin:.10rem 0 .30rem;}
    .analysis-stage-heading {font-size:1.02rem; font-weight:850; color:#2f485c; margin:.45rem 0 .48rem;}
    @media (max-width: 760px) {
        .project-flow-card {flex-basis:145px; min-width:145px;}
        .project-flow-card.data {flex-basis:170px; min-width:170px;}
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
    .combined-result, .result-line, .criteria-wrap, .references-box {width:100%; max-width:720px; box-sizing:border-box; margin-left:auto; margin-right:auto;}
    .combined-result {background:#f5f8fb; border:1px solid #d5e2ee; border-top:5px solid #7f9db6; border-radius:12px; padding:1.1rem 1.2rem; text-align:center; margin-top:.8rem; margin-bottom:.2rem; color:#2f485c;}
    .combined-result-title {font-size:.82rem; font-weight:800; color:#647a8d; margin-bottom:.40rem;}
    .combined-result-value {font-size:1.35rem; font-weight:900; margin-bottom:.42rem;}
    .combined-result-text {font-size:.86rem; line-height:1.55;}
    .criteria-wrap {margin-top:.90rem; margin-bottom:.30rem;}
    .criteria-heading {font-size:.98rem; font-weight:850; color:#2f485c; margin:0 0 .48rem;}
    .criteria-panel {background:#fff; border:1px solid #dce4ec; border-radius:10px; padding:.72rem .78rem; height:100%;}
    .criteria-panel-title {font-size:.80rem; font-weight:850; color:#3e627b; margin-bottom:.45rem;}
    .criteria-row {border:1px solid #e1e7ec; border-radius:7px; padding:.34rem .44rem; margin:.24rem 0; font-size:.72rem; line-height:1.38; color:#4b606f;}
    .criteria-row b {color:#2f485c;}
    .criteria-row.low, .criteria-row.mid, .criteria-row.high, .criteria-row.vhigh {background:#ffffff; border-color:#e1e7ec;}
    .criteria-row.current {background:#eef8f1; border-color:#a8d3b4;}
    .environment-criteria .criteria-panel {background:#f8fbfe; border-color:#c9ddeb;}
    .environment-criteria .criteria-panel-title {color:#365f79;}
    .environment-criteria .criteria-row.current {background:#e8f3fb; border-color:#8fb4cd; color:#2f5c78;}
    .environment-criteria .criteria-row.current b {color:#2f5c78;}
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
.survey-section-spacer {height:1.15rem;}
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
    font-size:.88rem; line-height:1.65; text-align:center;
}
.formula-arrow {display:block; text-align:center; color:#6f92aa; font-size:1.04rem; line-height:1.05; margin:.12rem 0;}
.nsch-section-subheading {font-size:.98rem; font-weight:750; color:#344f62; margin:.72rem 0 .32rem;}


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
    .environment-report .report-kicker, .environment-report .report-name {color:#365f79;}
    .environment-report .report-score {color:#3f7191;}
    .environment-report .report-band {background:#e8f3fb; border-color:#b8d3e6; color:#2f5c78;}
    .environment-report .report-comment {background:#f4f9fd; border-color:#d3e3ef; color:#36566d;}
    .environment-report .report-section-title {color:#365f79;}
    .environment-report .action-list-wrap {background:#f8fbfe; border-color:#d3e3ef;}
    .environment-report .action-list-title {color:#365f79;}
    .environment-report .action-row.active {background:#e8f3fb; border-color:#b8d3e6; color:#2f5c78;}
    .references-box {border-top:1px solid #e0e7ed; margin-top:1.1rem; padding-top:.62rem; color:#667582; font-size:.72rem; line-height:1.7;}
    .references-box b {color:#405462;}
    .references-box a {color:#3f7191; text-decoration:none;}
    .references-box a:hover {text-decoration:underline;}

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
    # 코드북의 9개 학력 응답을 설문용 5단계로 단순화했다.
    # 모델은 범주형으로 학습됐으므로 각 단계의 대표 원래 코드(1·2·3·7·8)를 그대로 전달한다.
    "a1_grade": [(1, "초등학교 이하"), (2, "중학교"), (3, "고등학교"), (7, "대학교"), (8, "대학원")],
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

# 4~11세 ACE 4개 이상 위험신호 분석 산출물.
# 기존 NSCH-ASD 산출물은 보존하고, 결과 화면은 이 새 모델만 사용한다.
NSCH_RISK_ART = APP_DIR / "model_artifacts" / "nsch_ace4_8q"
NSCH_RISK_META = (json.loads((NSCH_RISK_ART / "final_model_metadata.json").read_text(encoding="utf-8")) if (NSCH_RISK_ART / "final_model_metadata.json").exists() else {})
NSCH_RISK_FEATURES = NSCH_RISK_META.get("final_features", [])

NSCH_LABELS.update({
    "family_r": "가족 형태", "a1_employed_r": "주 보호자의 현재 고용 상태",
    "a1_grade": "주 보호자의 최종 학력", "foodsit": "최근 12개월 가구 식품 상황",
    "missmortgage": "주거비·임대료 납부 곤란", "k10q40_r": "아이가 느끼는 동네 안전",
    "k10q22": "동네의 노후·불량 주택", "k10q23": "동네의 기물파손",
    "moves": "이사 횟수", "hoursleep": "평균 수면 시간", "screentime": "화면 사용 시간",
    "k10q41_r": "아이가 느끼는 학교 안전", "k7q04r_r": "학교가 가정에 문제로 연락한 횟수",
    "k7q82_r": "학교에서 잘하고 싶어 하는 마음",
    "a1_relation": "주 보호자와 아이의 관계",
    "a1_marital": "주 보호자의 혼인 상태", "a2_marital": "두 번째 보호자의 혼인 상태",
    "everhomeless": "거주지 없이 지낸 경험", "homeevic": "주거 퇴거 걱정",
    "bullied_r": "최근 따돌림·괴롭힘 경험",
})
NSCH_CODE_OPTIONS.update({
    "k10q22": [(1, "예"), (2, "아니오")],
    "k10q23": [(1, "예"), (2, "아니오")],
    "moves": [(1, "0회"), (2, "1회"), (3, "2회 이상")],
    "a1_relation": [(1, "친부모·입양부모"), (2, "새부모"), (3, "조부모"), (4, "위탁부모"), (6, "기타 친족"), (7, "비친족 보호자")],
    "a1_marital": [(1, "기혼"), (2, "동거 중이나 미혼"), (3, "미혼"), (4, "이혼"), (5, "별거"), (6, "사별")],
    "a2_marital": [(1, "기혼"), (2, "동거 중이나 미혼"), (3, "미혼"), (4, "이혼"), (5, "별거"), (6, "사별")],
    "everhomeless": [(2, "아니오"), (3, "모름"), (1, "예")],
    "homeevic": [(5, "전혀 없음"), (4, "드물게"), (3, "가끔"), (2, "대부분"), (1, "항상")],
    "bullied_r": [(1, "최근 12개월 동안 없음"), (2, "1~2회"), (3, "한 달에 1~2회"), (4, "일주일에 1~2회"), (5, "거의 매일")],
})
NSCH_SURVEY_QUESTIONS.update({
    "family_r": "아이와 함께 사는 가족의 형태는 무엇입니까?",
    "a1_employed_r": "아이의 주 보호자의 현재 고용 상태는 무엇입니까?",
    "a1_grade": "아이의 주 보호자의 최종 학력은 어디에 해당합니까?",
    "foodsit": "최근 12개월 동안 아이 가정의 식품 상황은 어떠했습니까?",
    "missmortgage": "최근 12개월 동안 주거비나 임대료를 제때 내기 어려웠던 적이 있습니까?",
    "moves": "아이는 지금 주소로 오기 전까지 이사를 몇 번 했습니까?",
    "k10q22": "아이의 동네에 관리가 잘 되지 않은 낡은 주택이 있습니까?",
    "k10q23": "아이의 동네에 기물파손이나 낙서가 있습니까?",
    "screentime": "평일에 아이가 TV·휴대폰·컴퓨터 화면을 사용하는 시간은 어느 정도입니까?",
    "hoursleep": "지난 일주일 동안 아이의 하루 평균 수면 시간은 어느 정도입니까?",
    "k10q40_r": "아이는 사는 동네가 안전하다고 생각합니까?",
    "k10q41_r": "아이는 학교가 안전하다고 생각합니까?",
    "k7q04r_r": "최근 12개월 동안 아이의 문제로 학교에서 가정에 연락한 적이 있습니까?",
    "k7q82_r": "아이는 학교에서 잘하고 싶어 하는 모습을 얼마나 보입니까?",
    "a1_relation": "아이의 주 보호자는 아이와 어떤 관계입니까?",
    "a1_marital": "아이의 주 보호자의 현재 혼인 상태는 무엇입니까?",
    "a2_marital": "아이의 두 번째 보호자의 현재 혼인 상태는 무엇입니까?",
    "everhomeless": "아이가 거주지 없이 지낸 경험이 있습니까?",
    "homeevic": "현재 사는 집에서 나가야 할까 걱정되는 일이 얼마나 있습니까?",
    "bullied_r": "최근 12개월 동안 아이가 또래에게 따돌림이나 괴롭힘을 당한 적이 있습니까?",
})

final_row = final_result.iloc[0] if not final_result.empty else pd.Series(dtype="object")


# ============================================================
# 3. 데이터 정리 / 표시 이름
# ============================================================
def normalize_raw(df):
    out = df.copy()
    # pandas 3.x에서는 문자열 dtype이 object와 분리될 수 있으므로 둘 다 명시한다.
    for c in out.select_dtypes(include=["object", "str"]).columns:
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
    "결과",
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


def nsch_learning_tree(active_stage):
    """NSCH 5개 화면 옆에서 수업 흐름 전체와 현재 단계를 함께 보여 준다."""
    stages = [
        (0, "① 데이터 불러오기", "└─ pandas.read_stata() · NSCH 2024 원자료"),
        (0, "② 데이터 확인", "└─ 행·열 · 4~11세 · ACE 유효응답 · 결측 확인"),
        (0, "③ 독립변수 X / 종속변수 y 분리", "└─ X=환경·생활 항목 · y=ACE 4개 이상 여부"),
        (0, "④ 전처리", "└─ 최빈값 대체 · One-Hot Encoding · Pipeline"),
        (1, "⑤ 통계적 관계 확인", "└─ ACE 4개 이상 vs 0~3개 · 카이제곱 · Cramér's V"),
        (1, "⑥ 최종 질문 선정", "└─ 관계 크기 + Random Forest 변수 중요도"),
        (2, "⑦ 훈련·검증·최종평가 분리", "└─ Train / Validation / Test = 60% / 20% / 20%"),
        (2, "⑧ 지도학습 분류모델 비교", "└─ 로지스틱 회귀 · KNN · 의사결정나무 · 랜덤 포레스트"),
        (2, "⑨ 분류 성능 평가", "└─ 정확도 · 정밀도 · 재현율 · F1 · ROC-AUC"),
        (3, "⑩ 새 설문 점수 계산", "└─ RF 문항 가중치 + 응답별 위험군 차이 합산"),
        (4, "⑪ 최종 결론·설문 문항", "└─ 통계 관계 + RF 중요도 + 모델 검증 → 8문항 확정"),
    ]
    tree_html = []
    for stage, title, detail in stages:
        active = " active" if stage == active_stage else ""
        tree_html.append(
            f'<div class="nsch-tree-stage nsch-tree-stage-{stage}{active}">'
            f'<div class="nsch-tree-main">{title}</div><div class="nsch-tree-sub">{detail}</div></div>'
        )
    st.markdown(
        "<div class='nsch-learning-tree'><div class='nsch-learning-tree-title'>NSCH 분석 순서</div>"
        + "".join(tree_html)
        + "<div class='nsch-tree-foot'>파란색 영역은 현재 보고 있는 분석 단계입니다.</div></div>",
        unsafe_allow_html=True,
    )


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

    page_title("NSCH 외부데이터 분석", "NSCH 2024 자료에서 현재 ASD 여부와 함께 나타나는 환경·생활 특성을 분석했습니다.")
    tabs = st.tabs(["1. 데이터 확인 및 전처리", "2. 연관성 분석", "3. 환경·생활 요인 선정", "4. 머신러닝 모델 비교", "5. 모델 성능 평가", "6. 생활환경 점수 산출"])

    with tabs[0]:
        nsch_pipeline(0)
        st.subheader("1. 데이터 확인 및 전처리")
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

    with tabs[1]:
        nsch_pipeline(1)
        st.subheader("2. ASD 여부와 환경·생활 특성의 연관성 확인")
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
        st.subheader("3. 최종 환경·생활 요인 선정")
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
        st.subheader("4. 머신러닝 모델 비교")
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
        st.subheader("5. 최종 모델 성능 평가")
        st.markdown(f'''<div class="split-diagram">
            <div class="split-root-row"><div class="split-box root"><strong>전체 분석 데이터 {int(target['usable_current_asd']):,}명</strong><br>현재 ASD 여부 유효응답</div></div>
            <div class="split-branch-arrows"><span>↙</span><span>↘</span></div>
            <div class="split-branches">
                <div class="split-box train"><strong>Train · {int(split_counts.get('train', 0)):,}명</strong><br>모델 학습</div>
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
        st.subheader("6. 생활환경 관찰점수 산출")
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


def render_nsch_ace4_page_legacy():
    """4~11세 ACE 4개 이상 위험신호 분석 결과를 기존 NSCH 화면 형식으로 보여 준다."""
    target = pd.read_csv(NSCH_RISK_ART / "target_summary.csv").set_index("metric")["value"]
    candidates = pd.read_csv(NSCH_RISK_ART / "candidate_features.csv")
    imputation = pd.read_csv(NSCH_RISK_ART / "imputation_summary.csv")
    stats = pd.read_csv(NSCH_RISK_ART / "statistical_tests.csv")
    selection = pd.read_csv(NSCH_RISK_ART / "final_selection.csv")
    comparison = pd.read_csv(NSCH_RISK_ART / "model_validation_comparison.csv")
    group_response_comparison = pd.read_csv(NSCH_RISK_ART / "group_response_comparison.csv")
    final = pd.read_csv(NSCH_RISK_ART / "final_test_metrics.csv").iloc[0]
    split_summary = pd.read_csv(NSCH_RISK_ART / "split_summary.csv")
    severity_summary = pd.read_csv(NSCH_RISK_ART / "severity_summary.csv")
    severity_comparison = pd.read_csv(NSCH_RISK_ART / "severity_validation_comparison.csv")
    severity_final = pd.read_csv(NSCH_RISK_ART / "severity_test_metrics.csv").iloc[0]
    selected = selection[selection["selected_final"]].copy()
    model_names = {"LogisticRegression": "로지스틱 회귀", "KNN": "K-최근접 이웃", "DecisionTree": "의사결정나무", "RandomForest": "랜덤 포레스트"}
    chosen_model = str(final["model"])
    split_counts = split_summary.groupby("split")["count"].sum().to_dict()

    page_title("NSCH 외부데이터 분석", "4~11세 아동의 ACE 4개 이상 경험과 함께 나타나는 환경·생활 요인을 분석해 생활환경 설문을 구성했습니다.")
    tabs = st.tabs(["1. 데이터 확인 및 전처리", "2. 관련성 분석", "3. 최종 질문 선정", "4. 머신러닝 모델 비교", "5. 모델 성능 평가", "6. 설문·점수 연결"])

    with tabs[0]:
        nsch_pipeline(0)
        st.subheader("1. 4~11세 분석 자료 만들기")
        learning_cards("NSCH 2024 원자료를 확인", "행동 설문과 같은 연령대에 맞추기 위해", "4~11세와 ACE 9개 유효 응답만 남김", f"{int(target['ace_complete_4_11_count']):,}명 분석")
        cols = st.columns(4, gap="small")
        cols[0].metric("NSCH 전체", f"{int(target['raw_rows']):,}명")
        cols[1].metric("4~11세 아동", f"{int(target['age_4_11_count']):,}명")
        cols[2].metric("ACE 분석 가능", f"{int(target['ace_complete_4_11_count']):,}명")
        cols[3].metric("ACE 4개 이상", f"{int(target['ace4_high_count']):,}명")
        severity_total = severity_summary.groupby("severity_group")["count"].sum()
        s1, s2, s3 = st.columns(3, gap="small")
        s1.metric("ACE 0~3개", f"{int(severity_total.get('ACE 0~3개', 0)):,}명")
        s2.metric("ACE 4~5개", f"{int(severity_total.get('ACE 4~5개', 0)):,}명")
        s3.metric("ACE 6개 이상", f"{int(severity_total.get('ACE 6개 이상', 0)):,}명")
        st.markdown("<div class='formula-box'><b>분석 흐름</b><br>4~11세 아동 → ACE 9개 문항 유효 응답 확인 → ACE 4개 이상 / 0~3개 구분 → 환경·생활 요인 분석</div>", unsafe_allow_html=True)
        prep = pd.DataFrame([
            ["원본 자료", f"{int(target['raw_rows']):,}명 × {int(target['raw_columns']):,}개 항목", "NSCH 2024 전체 아동 설문"],
            ["연령 기준", f"{int(target['age_4_11_count']):,}명", "행동 설문과 동일한 4~11세로 제한"],
            ["분석 대상", f"{int(target['ace_complete_4_11_count']):,}명", "ACE 9개 문항 응답이 모두 유효한 아동"],
            ["예측할 결과", "ACE 4개 이상 여부", "4개 이상=1, 0~3개=0"],
        ], columns=["항목", "결과", "왜 사용했나"])
        left_table(prep, .92)
        st.subheader("수업에서 배운 방법 중 무엇을 사용했나")
        lesson_map = pd.DataFrame([
            ["문제 유형", "지도학습 · 분류", "결과가 연속 숫자가 아니라 ACE 4개 이상 여부이기 때문", "사용"],
            ["독립변수 X / 종속변수 y", "X=환경·생활 14개 후보, y=ACE 4개 이상 여부", "입력과 정답을 분리해 학습하기 위해", "사용"],
            ["상관계수·p-value", "카이제곱·Cramér's V·Spearman", "후보와 결과의 관계 및 우연 가능성 확인", "사용"],
            ["범주형 전처리", "최빈값 대치 + OneHotEncoder", "응답코드를 연속적인 크기로 오해하지 않게 하기 위해", "사용"],
            ["데이터 분리", "Train 60% / Validation 20% / Test 20%", "학습·모델선택·최종평가를 분리하기 위해", "사용"],
            ["분류모델", "로지스틱·KNN·의사결정나무·랜덤 포레스트", "같은 입력으로 성능을 비교하기 위해", "비교"],
            ["랜덤 포레스트", "원-핫 더미 중요도를 원래 질문별로 합산", "최종 질문 선정 근거를 만들기 위해", "사용"],
            ["K-means 군집화", "정답 없이 유형을 묶는 비지도학습", "이번에는 ACE 단계라는 정답이 있어 최종 예측에는 부적합", "미사용"],
            ["PCA", "여러 변수를 소수 주성분으로 축소", "8개 질문의 의미와 설명력을 유지해야 하므로 부적합", "미사용"],
            ["회귀", "연속값 예측", "이번 결과는 위험군 범주이므로 분류가 더 적합", "미사용"],
        ], columns=["수업 내용", "이번 분석에서의 의미", "선택·제외 이유", "적용"])
        left_table(lesson_map, .98, 430)

    with tabs[1]:
        nsch_pipeline(1)
        st.subheader("2. ACE 4개 이상 여부와 환경·생활 요인의 관련성 확인")
        learning_cards("후보별로 두 집단 차이 확인", "단순히 자주 나온 항목이 아닌, ACE 4개 이상과 구분되는 항목을 찾기 위해", "범주형은 카이제곱·Cramér's V, 순서형은 Spearman", f"{int(stats['significant_0_05'].sum())}개 항목 유의")
        mini_note("여기서 비교한 대상은 ‘ACE 4개 이상’ 집단과 ‘ACE 0~3개’ 집단입니다. p-value는 우연한 차이인지, 관계 크기는 차이가 얼마나 큰지 보여 줍니다.")
        view = stats.copy()
        view["관계 정도"] = view["effect_size"].map(lambda x: f"{float(x):.3f}")
        view["p-value"] = view["p_value"].map(lambda x: "< 0.001" if float(x) < .001 else f"{float(x):.3f}")
        view["결과"] = np.where(view["significant_0_05"], "통계적으로 유의", "유의하지 않음")
        view = view[["korean_name", "domain", "test_method", "관계 정도", "p-value", "결과"]]
        view.columns = ["분석 항목", "영역", "사용한 방법", "관계 정도", "p-value", "결과"]
        left_table(view, .96, 440)
        result_line("표본이 크면 작은 차이도 유의하게 나올 수 있으므로, 다음 단계에서 관계 크기와 랜덤 포레스트 변수 중요도를 함께 확인했습니다.")

    with tabs[2]:
        nsch_pipeline(2)
        st.subheader("3. 14개 후보에서 최종 10개 설문 질문 선정")
        learning_cards("환경·생활 후보 14개 검토", "ACE 문항을 직접 묻지 않는 10문항 설문을 만들기 위해", "통계 관계 순위 + 누적 ACE 가중 RF 중요도 순위를 합산", f"최종 {len(selected)}개 질문 선정")
        st.markdown("<div class='formula-box'><b>14개 후보</b> → 통계적 관계 크기 확인 → 원-핫 인코딩 → ACE 4개=1.0, 5개=1.25, 6개=1.5, 7개=1.75, 8개=2.0 가중 Random Forest → 두 순위 합산 → <b>최종 10개 질문</b></div>", unsafe_allow_html=True)
        plot_data = selected.sort_values("rf_importance")
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        bars = ax.barh(plot_data["korean_name"], plot_data["rf_importance"], color="#7ea6c4")
        for bar, value in zip(bars, plot_data["rf_importance"]):
            ax.text(float(value) + .002, bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center", fontsize=7)
        ax.set_xlabel("랜덤 포레스트 변수 중요도", fontsize=8)
        ax.set_ylabel("생활환경 설문 항목", fontsize=8)
        ax.set_title("ACE 4개 이상 분류에 중요하게 사용된 항목", fontsize=10)
        ax.tick_params(labelsize=7); fig.tight_layout(); left_plot(fig, .56)
        table = selected.sort_values("selection_rank")[["korean_name", "domain", "reason", "effect_size", "rf_importance"]].copy()
        table["effect_size"] = table["effect_size"].map(lambda x: f"{float(x):.3f}")
        table["rf_importance"] = table["rf_importance"].map(lambda x: f"{float(x):.3f}")
        table.columns = ["최종 질문", "영역", "후보 선정 이유", "관계 정도", "RF 중요도"]
        left_table(table, .96, 360)

    with tabs[3]:
        nsch_pipeline(3)
        st.subheader("4. 같은 10개 질문으로 4개 모델 비교")
        learning_cards("네 가지 분류모델 학습", "특정 알고리즘을 미리 정하지 않고 검증자료에서 비교하기 위해", "Train 학습 → Validation 비교", f"최종 선택: {model_names[chosen_model]}")
        compare = comparison.copy(); compare["model"] = compare["model"].map(model_names)
        compare = compare[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "severity_weighted_f1", "ace6plus_recall"]]
        for col in compare.columns[1:]: compare[col] = compare[col].map(lambda x: f"{float(x):.3f}")
        compare.columns = ["모델", "정확도", "정밀도", "재현율", "F1", "ROC-AUC", "누적가중 F1", "ACE 6개+ 재현율"]
        left_table(compare, .98)
        result_line("ACE 4개 이상 집단의 비율이 낮으므로 정확도만 보지 않고, 실제 위험신호 집단을 놓치지 않는 재현율·F1·ROC-AUC를 함께 비교했습니다.")
        st.subheader("계층형 2단계 분류")
        st.markdown("<div class='formula-box'><b>1차 분류</b> · ACE 0~3개 / 4개 이상 → 위험신호 점수<br><br><b>2차 분류</b> · 1차 위험군 안에서 ACE 4~5개 / 6개 이상 → 누적 단계 패턴</div>", unsafe_allow_html=True)
        stage2_view = severity_comparison.copy(); stage2_view["model"] = stage2_view["model"].map(model_names)
        stage2_view = stage2_view[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
        for col in stage2_view.columns[1:]: stage2_view[col] = stage2_view[col].map(lambda x: f"{float(x):.3f}")
        stage2_view.columns = ["모델", "정확도", "정밀도", "재현율", "F1", "ROC-AUC"]
        left_table(stage2_view, .90)
        mini_note("한 번에 3개 집단을 분류하면 0~3개 집단이 지나치게 커서 작은 집단 학습이 약해집니다. 그래서 먼저 4개 이상을 찾고, 그 안에서 6개 이상 패턴을 한 번 더 구분했습니다.")

    with tabs[4]:
        nsch_pipeline(4)
        st.subheader("5. 최종 모델 성능 평가")
        learning_cards("Test 자료로 마지막 평가", "모델 선택 과정에 쓰지 않은 자료에서 일반화 성능을 확인하기 위해", f"Train {int(split_counts['train']):,}명 + Validation {int(split_counts['validation']):,}명으로 재학습", f"Test {int(split_counts['test']):,}명 평가")
        metrics_df = pd.DataFrame([
            ["정확도", final["accuracy"], "전체 분류가 맞은 비율"], ["정밀도", final["precision"], "높은 위험신호로 표시한 아동 중 실제 ACE 4개 이상 비율"],
            ["재현율", final["recall"], "실제 ACE 4개 이상 아동 중 모델이 찾아낸 비율"], ["F1 점수", final["f1"], "정밀도와 재현율의 균형"],
            ["ROC-AUC", final["roc_auc"], "두 집단을 전반적으로 구분하는 정도"],
            ["누적가중 F1", final["severity_weighted_f1"], "ACE 개수가 많은 사례를 더 크게 반영한 F1"],
            ["ACE 6개 이상 재현율", final["ace6plus_recall"], "ACE 6개 이상 사례 중 1차 모델이 찾아낸 비율"],
        ], columns=["평가 지표", "Test 결과", "의미"])
        metrics_df["Test 결과"] = metrics_df["Test 결과"].map(lambda x: f"{float(x):.3f}")
        left_table(metrics_df, .92)
        fig, ax = plt.subplots(figsize=(4.2, 2.6)); keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        bars = ax.bar(["정확도", "정밀도", "재현율", "F1", "ROC-AUC"], [float(final[k]) for k in keys], color="#7ea6c4")
        ax.set_ylim(0, 1.05); ax.set_ylabel("Test 성능", fontsize=8); ax.tick_params(labelsize=8)
        for bar, value in zip(bars, [float(final[k]) for k in keys]): ax.text(bar.get_x()+bar.get_width()/2, value+.02, f"{value:.3f}", ha="center", fontsize=7)
        fig.tight_layout(); left_plot(fig, .50)
        st.subheader("2차 누적단계 모델 Test 결과")
        stage2_test = pd.DataFrame([
            ["선택 모델", model_names[str(severity_final["model"])]], ["정확도", f"{float(severity_final['accuracy']):.3f}"],
            ["정밀도", f"{float(severity_final['precision']):.3f}"], ["재현율", f"{float(severity_final['recall']):.3f}"],
            ["F1", f"{float(severity_final['f1']):.3f}"], ["ROC-AUC", f"{float(severity_final['roc_auc']):.3f}"],
        ], columns=["항목", "결과"])
        left_table(stage2_test, .64)
        mini_note("2차 모델은 표본이 더 작아 1차 점수를 대체하지 않으며, ACE 4~5개 패턴과 6개 이상 패턴을 구분하는 보조 결과로만 사용합니다.")

    with tabs[5]:
        nsch_pipeline(5)
        st.subheader("6. 최종 8문항을 생활환경 설문 점수로 연결")
        learning_cards("선정된 8문항을 설문으로 구성", "센터에서 생활환경 위험신호를 같은 방식으로 확인하기 위해", "응답 범주를 원-핫 변환 후 1차·2차 모델 입력", "0~100 위험신호 점수 + 누적 단계 패턴")
        st.markdown(f"<div class='formula-box'><b>1차 점수</b><br>8문항 응답 → {model_names[chosen_model]} → predict_proba × 100 → ACE 4개 이상 위험신호 점수<br><br><b>2차 보조 결과</b><br>같은 8문항 → {model_names[str(severity_final['model'])]} → ACE 4~5개 / 6개 이상 누적 패턴 구분</div>", unsafe_allow_html=True)
        input_view = selected[["korean_name", "domain"]].copy(); input_view.columns = ["설문 질문", "영역"]
        left_table(input_view, .76, 330)
        result_line("랜덤 포레스트 중요도는 질문을 고르는 근거로 사용했고, ACE 개수가 많을수록 학습 가중치를 높였습니다. 실제 0~100점과 누적 단계는 각각 Validation에서 선택된 로지스틱 회귀 모델이 계산합니다.")


def render_nsch_ace4_stage_content(target, candidates, imputation, stats, selection, comparison,
                                   final, broad_candidate_count, selected, split_counts,
                                   model_names, chosen_model, group_response_comparison, current_stage):
    """선택된 NSCH 단계의 상세 내용을 오른쪽 본문 열에 렌더링한다."""
    if current_stage == 0:
        st.subheader("1. 데이터 확인 및 전처리")
        summary_cards = st.columns(4, gap="small")
        with summary_cards[0]:
            metric_card("행(Row)", f"{int(target['raw_rows']):,}명")
        with summary_cards[1]:
            metric_card("열(Column)", f"{int(target['raw_columns']):,}개")
        with summary_cards[2]:
            metric_card("중복 응답", "0건")
        with summary_cards[3]:
            metric_card("결측 응답(8개 입력)", f"{int(imputation['analysis_missing_count'].sum()):,}건")
        st.markdown(
            f"<div class='nsch-reduction-flow'><b>전처리 규모 변화</b><br>"
            f"분석 데이터: {int(target['raw_rows']):,}명 → {int(target['age_4_11_count']):,}명(4~11세) → {int(target['ace_complete_4_11_count']):,}명(ACE 응답 유효)<br>"
            f"분석 항목: {int(target['raw_columns']):,}개 → {broad_candidate_count}개(전수 탐색 후보) → {len(candidates)}개(설문 가능 환경·생활 후보) → {len(selected)}개(최종 설문 입력)</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### 전처리 내용")
        preprocess_tbl = pd.DataFrame([
            ["연령 기준 적용", f"전체 {int(target['raw_rows']):,}명", f"4~11세 {int(target['age_4_11_count']):,}명", "행동패턴 설문과 같은 4~11세만 남김"],
            ["ACE 유효 응답 확인", f"4~11세 {int(target['age_4_11_count']):,}명", f"분석 가능 {int(target['ace_complete_4_11_count']):,}명", f"ACE 9개 응답이 부족한\n{int(target['age_4_11_count']) - int(target['ace_complete_4_11_count']):,}명 제외"],
            ["결과값 생성", f"분석 가능 {int(target['ace_complete_4_11_count']):,}명", f"0~3개 {int(target['ace_complete_4_11_count']) - int(target['ace4_high_count']):,}명\n4개 이상 {int(target['ace4_high_count']):,}명", "ACE 개수를 두 위험신호 그룹으로 변환"],
            ["컬럼 개수", f"원본 {int(target['raw_columns']):,}개 → 전수 탐색 {broad_candidate_count}개", f"설문 가능 후보 {len(candidates)}개 → 최종 {len(selected)}개", "기술·정답 누수 항목 제외 후\n설문 가능 영역만 다시 정리"],
            ["결측값 처리", "최종 8개 컬럼", f"결측이 있는 {len(imputation)}개 컬럼", "Train 자료의 최빈 응답으로 대체"],
            ["범주형 변환", "최종 8개 선택지", "OneHotEncoder 입력값", "각 선택지를 별도 0/1 입력값으로 변환"],
        ], columns=["전처리 종류", "처리 전", "처리 후", "처리 방법"])
        preprocess_style = (
            preprocess_tbl.style
            .set_properties(**{"text-align": "center", "white-space": "pre-wrap"})
            .set_table_styles([
                {"selector": "th", "props": [("text-align", "center")]},
                {"selector": "td", "props": [("white-space", "pre-wrap")]},
            ])
        )
        left_table(preprocess_style, .96)
        st.markdown(
            f"<div class='nsch-preprocess-result'>전처리 결과: <b>{int(target['raw_rows']):,}명 · {int(target['raw_columns']):,}개 항목</b>에서 시작해, <b>4~11세 ACE 유효응답 {int(target['ace_complete_4_11_count']):,}명 · 전수 탐색 {broad_candidate_count}개 · 설문 가능 후보 {len(candidates)}개</b>로 분석 범위를 정리했습니다.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### 결측값 처리 상세")
        imputation_view = imputation.copy()
        def mode_label(row):
            code = int(float(row["train_mode_code"]))
            options = NSCH_CODE_OPTIONS.get(row["column"], [])
            return next((label for value, label in options if value == code), f"응답 코드 {code}")
        imputation_view["Train 최빈 응답"] = imputation_view.apply(mode_label, axis=1)
        imputation_view["분석자료 결측 수"] = imputation_view["analysis_missing_count"].map(lambda value: f"{int(value):,}개")
        imputation_view["Train 결측 수"] = imputation_view["train_missing_count"].map(lambda value: f"{int(value):,}개")
        imputation_view = imputation_view[["korean_name", "분석자료 결측 수", "Train 결측 수", "Train 최빈 응답", "method"]]
        imputation_view.columns = ["결측이 있었던 항목", "분석자료 결측 수", "Train 결측 수", "대체한 응답", "처리 방법"]
        left_table(imputation_view, .96, 315)
        mini_note("결측값은 Validation·Test에서 따로 계산하지 않고, Train 자료에서 정한 최빈 응답만 적용했습니다. 그래야 평가 자료의 정보가 학습 과정에 섞이지 않습니다.")


    if current_stage == 1:
        st.subheader("2. 관련성 확인과 최종 8문항 선정")
        st.markdown(
            f"<div class='nsch-preprocess-result'><b>무엇과 무엇을 비교했나?</b> 4~11세 ACE 유효응답 {int(target['ace_complete_4_11_count']):,}명에서, <b>ACE 4개 이상 위험신호 {int(target['ace4_high_count']):,}명</b>과 <b>ACE 0~3개 비교집단 {int(target['ace4_low_count']):,}명</b>의 환경·생활 응답 비율을 비교했습니다. 비교집단이 있어야 어떤 항목이 전체 아동에게 흔한 특성인지, 위험신호 그룹에서 상대적으로 더 많이 나타나는 특성인지 구분할 수 있습니다.</div>",
            unsafe_allow_html=True,
        )
        left, right = st.columns([.48, .52], gap="large")
        with left:
            st.markdown(f"""<div class='formula-box'><b>질문 선정 흐름</b><br><br>
            전수 탐색 {broad_candidate_count}개 → 설문 가능 후보 {len(candidates)}개
            <div class='formula-arrow'>↓</div>
            카이제곱 검정·Cramér's V
            <div class='formula-arrow'>↓</div>
            Random Forest 변수 중요도
            <div class='formula-arrow'>↓</div>
            통계 관계와 중요도 순위 결합
            <div class='formula-arrow'>↓</div>
            최종 생활환경 설문 {len(selected)}문항</div>""", unsafe_allow_html=True)
            mini_note("전수 탐색에서는 각 범주형 응답을 카이제곱 검정과 Cramér's V로 ACE 4개 이상 여부와 비교했습니다. 그 뒤 설문으로 확인 가능한 환경·생활 항목만 남겨 모델 입력 후보로 사용했습니다.")
            stat_view = stats.copy()
            stat_view["관계 정도"] = stat_view["effect_size"].map(lambda x: f"{float(x):.3f}")
            stat_view["p-value"] = stat_view["p_value"].map(lambda x: "< 0.001" if float(x) < .001 else f"{float(x):.3f}")
            stat_view = stat_view[["korean_name", "test_method", "관계 정도", "p-value"]]
            stat_view.columns = ["후보 항목", "사용한 방법", "관계 정도", "p-value"]
            with st.expander(f"2차 후보 {len(candidates)}개 통계 결과 보기"):
                st.dataframe(stat_view, hide_index=True, width="stretch")
        with right:
            plot_data = selected.sort_values("rf_importance")
            fig, ax = plt.subplots(figsize=(8.2, 6.0))
            bars = ax.barh(plot_data["korean_name"], plot_data["rf_importance"], color="#7ea6c4")
            for bar, value in zip(bars, plot_data["rf_importance"]):
                ax.text(float(value) + .002, bar.get_y() + bar.get_height() / 2, f"{float(value):.3f}", va="center", fontsize=10)
            ax.set_title("최종 8문항 변수 중요도", fontsize=14, pad=10)
            ax.set_xlabel("Random Forest 변수 중요도", fontsize=11)
            ax.set_ylabel("생활환경 설문 항목", fontsize=11)
            ax.tick_params(labelsize=10); fig.tight_layout(pad=1.1)
            st.pyplot(fig, width="stretch")
            plt.close(fig)
        st.markdown("<div class='nsch-section-subheading'>최종 설문 질문별 응답 차이</div>", unsafe_allow_html=True)
        comparison_rows = []
        for column in selected.sort_values("selection_rank")["column"]:
            item = group_response_comparison[group_response_comparison["column"].eq(column)].copy()
            item["응답"] = item["response_code"].map(
                lambda code: next((label for value, label in NSCH_CODE_OPTIONS.get(column, []) if value == int(float(code))), f"응답 {int(float(code))}")
            )
            # ACE 4개 이상 그룹에서 비율이 가장 크게 높았던 응답을 같은 기준으로 비교한다.
            strongest = item.loc[item["difference_pp"].idxmax()]
            comparison_rows.append([
                NSCH_LABELS.get(column, str(strongest["korean_name"])),
                str(strongest["응답"]),
                f"{float(strongest['low_rate']) * 100:.1f}%",
                f"{float(strongest['high_rate']) * 100:.1f}%",
                f"+{float(strongest['difference_pp']):.1f}%p",
            ])
        group_summary = pd.DataFrame(comparison_rows, columns=[
            "최종 설문 항목", "4개 이상 그룹에서 더 많았던 응답", "ACE 0~3개", "ACE 4개 이상", "차이",
        ])
        mini_note("각 항목에서 ACE 4개 이상 그룹의 비율이 가장 크게 높았던 응답을 비교했습니다. ‘차이’는 두 그룹의 응답 비율 차이(%p)이며, 원인을 뜻하지는 않습니다.")
        # 고정 높이를 크게 주면 실제 문항보다 아래에 빈 행처럼 보이는 영역이 생긴다.
        # 현재 최종 문항 수에 맞춰 표 높이를 계산해 실제 행까지만 표시한다.
        group_summary_height = 44 + len(group_summary) * 34
        left_table(group_summary, 1.0, group_summary_height)
        with st.expander("문항별 전체 응답 분포 보기"):
            detail = group_response_comparison.copy()
            detail["응답"] = detail.apply(
                lambda row: next((label for value, label in NSCH_CODE_OPTIONS.get(row["column"], []) if value == int(float(row["response_code"]))), f"응답 {int(float(row['response_code']))}"), axis=1,
            )
            detail["ACE 0~3개 비율"] = detail["low_rate"].map(lambda value: f"{float(value) * 100:.1f}%")
            detail["ACE 4개 이상 비율"] = detail["high_rate"].map(lambda value: f"{float(value) * 100:.1f}%")
            detail["차이(%p)"] = detail["difference_pp"].map(lambda value: f"{float(value):+.1f}%p")
            detail = detail[["korean_name", "응답", "ACE 0~3개 비율", "ACE 4개 이상 비율", "차이(%p)"]]
            detail.columns = ["최종 설문 항목", "응답", "ACE 0~3개", "ACE 4개 이상", "차이"]
            st.dataframe(detail, hide_index=True, width="stretch", height=420)

    if current_stage == 2:
        st.subheader("3. 머신러닝 모델 비교")
        mini_note("여기서 머신러닝은 ‘정답을 새로 만드는 단계’가 아니라, 통계적으로 확인한 최종 8문항 조합이 두 그룹을 함께 구분할 때 어떤 항목을 중요하게 쓰는지와 조합 성능을 검증하는 단계입니다.")
        left, right = st.columns([.47, .53], gap="large")
        with left:
            st.markdown(f"""<div class='formula-box'><b>학습과 평가 흐름</b><br><br>
            최종 질문 8개
            <div class='formula-arrow'>↓</div>
            Train {int(split_counts['train']):,}명 · Validation {int(split_counts['validation']):,}명 · Test {int(split_counts['test']):,}명
            <div class='formula-arrow'>↓</div>
            로지스틱 회귀 / K-최근접 이웃<br>의사결정나무 / 랜덤 포레스트
            <div class='formula-arrow'>↓</div>
            Validation 성능 비교
            <div class='formula-arrow'>↓</div>
            <b>최종 선택: {model_names[chosen_model]}</b></div>""", unsafe_allow_html=True)
            plain_list([
                ("정확도", "전체 분류가 맞은 비율"), ("정밀도", "높은 위험신호로 표시한 결과의 정확성"),
                ("재현율", "실제 높은 위험신호를 놓치지 않고 찾은 비율"),
                ("F1", "정밀도와 재현율의 균형"), ("ROC-AUC", "두 그룹을 전반적으로 구분하는 능력"),
            ])
        with right:
            compare = comparison.copy(); compare["model"] = compare["model"].map(model_names)
            graph_metrics = ["recall", "f1", "roc_auc"]
            x = np.arange(len(compare)); width = .23
            fig, ax = plt.subplots(figsize=(4.7, 2.8))
            colors = ["#7ea6c4", "#a8c9a5", "#e4b985"]
            for idx, (metric, color) in enumerate(zip(graph_metrics, colors)):
                ax.bar(x + (idx - 1) * width, compare[metric], width, label={"recall": "재현율", "f1": "F1", "roc_auc": "ROC-AUC"}[metric], color=color)
            ax.set_xticks(x, compare["model"], rotation=0); ax.set_ylim(0, 1.05)
            ax.set_ylabel("Validation 성능", fontsize=8); ax.set_title("머신러닝 모델 성능 비교", fontsize=10)
            ax.legend(fontsize=7, ncol=3, loc="upper center"); ax.tick_params(labelsize=7); fig.tight_layout()
            left_plot(fig, 1.0)
            compare_view = compare[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
            for col in compare_view.columns[1:]: compare_view[col] = compare_view[col].map(lambda value: f"{float(value):.3f}")
            compare_view.columns = ["모델", "정확도", "정밀도", "재현율", "F1", "ROC-AUC"]
            left_table(compare_view, 1.0)

    if current_stage == 3:
        st.subheader("4. 최종 결과와 생활환경 설문 연결")
        survey_final = pd.read_csv(NSCH_RISK_ART / "weighted_score_test_metrics.csv").iloc[0]
        test_split = pd.read_csv(NSCH_RISK_ART / "split_summary.csv")
        test_high = int(test_split[(test_split["split"].eq("test")) & (test_split["ace4_high"].eq(1))]["count"].iloc[0])
        test_low = int(test_split[(test_split["split"].eq("test")) & (test_split["ace4_high"].eq(0))]["count"].iloc[0])
        test_total = test_high + test_low
        found_high = int(round(test_high * float(survey_final["recall"])))
        left, right = st.columns([.47, .53], gap="large")
        with left:
            st.markdown(f"""<div class='formula-box'><b>설문 점수 계산</b><br><br>
            생활환경 설문 8문항
            <div class='formula-arrow'>↓</div>
            Random Forest 문항 가중치
            <div class='formula-arrow'>↓</div>
            선택 응답의 위험군 반영점수
            <div class='formula-arrow'>↓</div>
            8문항 반영점수 합산
            <div class='formula-arrow'>↓</div>
            <b>생활환경 위험신호 점수 0~100</b></div>""", unsafe_allow_html=True)
            result_line("문항 가중치는 Random Forest로 계산하고, 선택 응답은 ACE 4개 이상 위험군에서 더 많이 나타난 정도로 반영합니다. 그래서 모든 문항을 바꾸면 해당 문항 점수가 즉시 달라집니다.")
        with right:
            st.markdown(
                f"<div class='explain-card'><b>무엇을 비교해 평가했나?</b><br>Test 자료 {test_total:,}명에서 ACE 4개 이상 {test_high:,}명(약 {test_high / test_total * 100:.1f}%)과 ACE 0~3개 {test_low:,}명을 구분한 결과입니다.<br><br>"
                f"<b>왜 F1·정밀도가 낮게 보이나?</b><br>ACE 4개 이상 사례가 매우 적은 불균형 자료라서, 실제 {test_high:,}명 중 약 {found_high:,}명을 찾은 재현율 {float(survey_final['recall']):.3f}과 달리 정밀도·F1은 낮아질 수 있습니다. 계산 오류가 아니라 두 집단의 규모 차이에서 나타나는 평가 특성입니다.<br><br>"
                f"<b>어떻게 해석하나?</b><br>ROC-AUC {float(survey_final['roc_auc']):.3f}은 8문항 가중점수가 두 집단을 전반적으로 순서대로 구분하는 능력입니다. 이 점수는 위험신호 요인을 정리하고 설문 문항을 검증하는 용도이며, 개별 결과를 단정하는 도구로 해석하지 않습니다.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='nsch-section-subheading'>Test 최종 성능 평가</div>", unsafe_allow_html=True)
        metrics_view = pd.DataFrame([
            ["점수 산출", "8문항 가중합", "RF 문항 가중치와 응답별 반영점수를 합산", "각 응답이 독립적으로 반영돼 설문 점수로 사용 가능"],
            ["정확도", f"{float(survey_final['accuracy']):.3f}", "전체 {0:,}명 중 맞게 구분한 비율".format(test_total), "높아 보이지만 ACE 4개 이상이 4.0%뿐이라 단독 판단 근거로는 부족"],
            ["정밀도", f"{float(survey_final['precision']):.3f}", "위험신호로 표시한 응답 중 실제 ACE 4개 이상인 비율", "낮음: 위험신호 표시 중 비교집단도 많음. 개별 위험군 확정에는 부족"],
            ["재현율", f"{float(survey_final['recall']):.3f}", f"실제 ACE 4개 이상 {test_high:,}명 중 약 {int(round(test_high * float(survey_final['recall']))):,}명을 찾아낸 비율", "보통 이하: 단독 선별 도구로는 충분하지 않음"],
            ["F1 점수", f"{float(survey_final['f1']):.3f}", "정밀도와 재현율을 함께 반영한 균형 지표", "낮음: 불균형 자료에서 위험군을 정확히 찾아내는 성능은 제한적"],
            ["ROC-AUC", f"{float(survey_final['roc_auc']):.3f}", "두 그룹을 전반적으로 구분하는 순위 성능", "양호: 관련 요인의 순위 구분과 설문 문항 검증 근거로는 활용 가능"],
        ], columns=["평가 지표", "Test 결과", "해석", "타당성 판단"])
        st.dataframe(metrics_view, hide_index=True, width="stretch")

    if current_stage == 4:
        st.subheader("5. 최종 결론과 생활환경 설문 문항")
        st.markdown(
            f"<div class='nsch-preprocess-result nsch-conclusion-result'><b>최종 결론이 나온 과정</b><br>"
            f"ACE 4개 이상 그룹 {int(target['ace4_high_count']):,}명과 ACE 0~3개 비교집단 {int(target['ace4_low_count']):,}명의 환경·생활 응답을 비교했습니다. "
            f"그중 설문 가능한 {len(candidates)}개 후보에 통계적 관계 크기와 Random Forest 변수 중요도를 적용하고, 최종 {len(selected)}개 조합을 4개 분류모델로 검증했습니다. "
            f"최종 모델은 <b>{model_names[chosen_model]}</b>이며, 이 결과를 바탕으로 아래 {len(selected)}개 문항을 간접 생활환경 설문으로 확정했습니다.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='nsch-final-heading'>최종 생활환경 문항 {len(selected)}개</div>", unsafe_allow_html=True)
        final_rows = []
        for number, column in enumerate(selected.sort_values("selection_rank")["column"], start=1):
            question_weight = float(
                selected.loc[selected["column"].eq(column), "rf_importance"].iloc[0]
                / selected["rf_importance"].sum() * 100
            )
            response_rows = group_response_comparison[group_response_comparison["column"].eq(column)].copy()
            response_rows["response_label"] = response_rows["response_code"].map(
                lambda code: next((label for value, label in NSCH_CODE_OPTIONS.get(column, []) if value == int(float(code))), f"응답 {int(float(code))}")
            )
            high_response = response_rows.loc[response_rows["difference_pp"].idxmax()]
            final_rows.append([
                number,
                NSCH_SURVEY_QUESTIONS.get(column, NSCH_LABELS.get(column, column)),
                str(high_response["response_label"]),
                f"{float(high_response['high_rate']) * 100:.1f}%",
                f"{float(high_response['low_rate']) * 100:.1f}%",
                f"+{float(high_response['difference_pp']):.1f}%p",
                f"{question_weight:.1f}점",
            ])
        final_table_rows = []
        for number, question, response, high_rate, low_rate, gap, question_weight in final_rows:
            final_table_rows.append(
                f"<tr><td class='nsch-final-num'>{number}</td><td class='nsch-final-question'>{question}</td>"
                f"<td class='nsch-final-response'>{response}</td><td class='nsch-final-rate'>{high_rate}</td>"
                f"<td class='nsch-final-rate'>{low_rate}</td><td class='nsch-final-gap'>{gap}</td>"
                f"<td class='nsch-final-weight'>{question_weight}</td></tr>"
            )
        st.markdown(
            "<table class='nsch-final-table'><thead><tr>"
            "<th style='width:5%'>번호</th><th style='width:33%'>최종 생활환경 문항</th><th style='width:18%'>위험군에서 더 많았던 응답</th>"
            "<th style='width:12%'>ACE 4개 이상<br>위험군 응답</th><th style='width:12%'>ACE 0~3개<br>비교집단 응답</th>"
            "<th style='width:10%'>위험군 응답 차이</th><th style='width:10%'>문항 가중치</th></tr></thead><tbody>"
            + "".join(final_table_rows) + "</tbody></table>",
            unsafe_allow_html=True,
        )
        mini_note("‘문항 가중치’는 Train 자료에서 Random Forest가 각 문항을 분류에 사용한 중요도를 100점으로 환산한 값입니다. ‘위험군 응답 차이’는 비교집단보다 해당 응답이 얼마나 더 많이 나타났는지 보여 주는 비율 차이(%p)입니다.")
        st.markdown("<div class='nsch-final-heading'>최종 선정 기준</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='formula-box nsch-final-flow'><b>최종 흐름</b><br><br>"
            f"ACE 4개 이상 그룹과 0~3개 그룹의 응답 비율 비교<div class='formula-arrow'>↓</div>"
            f"카이제곱 검정·Cramér’s V로 관계 크기 확인<div class='formula-arrow'>↓</div>"
            f"Random Forest로 분류에 중요하게 사용된 항목 확인<div class='formula-arrow'>↓</div>"
            f"로지스틱 회귀·KNN·의사결정나무·랜덤 포레스트 비교<div class='formula-arrow'>↓</div>"
            f"최종 {len(selected)}문항 확정</div>",
            unsafe_allow_html=True,
        )
        mini_note("이 문항들은 ACE 문항을 직접 묻지 않고, ACE 4개 이상 그룹과 관련성이 확인된 가족·주거·경제·학교·생활 특성을 확인하기 위한 간접 설문 항목입니다. 관련성은 함께 나타난 정도를 뜻하며 원인이나 개인의 결과를 단정하지 않습니다.")


def render_nsch_ace4_page():
    """왼쪽 분석 순서와 오른쪽 단계별 내용을 같은 행에 배치한다."""
    target = pd.read_csv(NSCH_RISK_ART / "target_summary.csv").set_index("metric")["value"]
    candidates = pd.read_csv(NSCH_RISK_ART / "candidate_features.csv")
    imputation = pd.read_csv(NSCH_RISK_ART / "imputation_summary.csv")
    stats = pd.read_csv(NSCH_RISK_ART / "statistical_tests.csv")
    selection = pd.read_csv(NSCH_RISK_ART / "final_selection.csv")
    comparison = pd.read_csv(NSCH_RISK_ART / "model_validation_comparison.csv")
    group_response_comparison = pd.read_csv(NSCH_RISK_ART / "group_response_comparison.csv")
    final = pd.read_csv(NSCH_RISK_ART / "final_test_metrics.csv").iloc[0]
    split_summary = pd.read_csv(NSCH_RISK_ART / "split_summary.csv")
    exploration_summary_path = APP_DIR / "model_artifacts" / "nsch_ace4_exploration" / "screen_summary.json"
    exploration_summary = json.loads(exploration_summary_path.read_text(encoding="utf-8")) if exploration_summary_path.exists() else {}
    broad_candidate_count = int(exploration_summary.get("screened_candidate_count", len(candidates)))
    selected = selection[selection["selected_final"]].copy()
    split_counts = split_summary.groupby("split")["count"].sum().to_dict()
    model_names = {"LogisticRegression": "로지스틱 회귀", "KNN": "K-최근접 이웃", "DecisionTree": "의사결정나무", "RandomForest": "랜덤 포레스트"}
    chosen_model = str(final["model"])
    stage_labels = ["1. 데이터 확인 및 전처리", "2. 관련성 확인·질문 선정", "3. 머신러닝 모델 비교", "4. 결과·설문 연결", "5. 최종 결론·설문 문항"]
    current_label = st.session_state.get("nsch_stage_selector", stage_labels[0])
    current_stage = stage_labels.index(current_label) if current_label in stage_labels else 0

    page_title("NSCH 외부데이터 분석", "ACE(아동기 부정적 경험) 4개 이상 고위험군과 관련된 가족·주거·경제·학교 요인을 찾아\n간접 생활환경 설문 문항을 제작하는 과정입니다.")
    tree_column, content_column = st.columns([.20, .80], gap="medium")
    with tree_column:
        nsch_learning_tree(current_stage)
    with content_column:
        selected_label = st.radio("NSCH 분석 단계", stage_labels, horizontal=True, label_visibility="collapsed", key="nsch_stage_selector")
        selected_stage = stage_labels.index(selected_label)
        render_nsch_ace4_stage_content(
            target, candidates, imputation, stats, selection, comparison, final,
            broad_candidate_count, selected, split_counts, model_names, chosen_model,
            group_response_comparison, selected_stage,
        )


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


def load_nsch_response_weights():
    """Train 자료에서 계산한 응답별 가중 반영표를 불러온다."""
    return pd.read_csv(NSCH_RISK_ART / "response_weight_summary.csv")


@st.cache_resource
def load_pickle_model(model_path, modified_time):
    """같은 모델 파일을 설문 입력 때마다 다시 읽지 않고 메모리에서 재사용한다."""
    del modified_time  # 모델 파일이 바뀌면 캐시 키만 갱신하는 용도다.
    with Path(model_path).open("rb") as model_file:
        return pickle.load(model_file)


def load_nsch_score_calibration():
    """Validation 가중 설문점수 분포를 불러와 현재 점수의 상대 위치를 계산한다."""
    return pd.read_csv(NSCH_RISK_ART / "environment_score_calibration.csv")["survey_score"].to_numpy()


def nsch_environment_percentile(survey_score):
    """새 가중 설문점수가 Validation 자료 안에서 어느 백분위인지 0~100으로 계산한다."""
    reference = load_nsch_score_calibration()
    return float(np.searchsorted(np.sort(reference), float(survey_score), side="right") / len(reference) * 100)


def nsch_weighted_response_summary(answers):
    """선택한 응답의 위험군 차이×RF 중요도 반영 정도를 0~100으로 요약한다."""
    weights = load_nsch_response_weights()
    rows = []
    for column, response_code in answers.items():
        if response_code is None or pd.isna(response_code):
            continue
        matched = weights[
            weights["column"].eq(column)
            & np.isclose(weights["response_code"].astype(float), float(response_code))
        ]
        if matched.empty:
            continue
        row = matched.iloc[0]
        rows.append({
            "문항": NSCH_LABELS.get(column, column),
            "문항 가중치": float(row["question_weight"]),
            "선택 응답 반영점수": float(row["max_contribution"]),
        })
    detail = pd.DataFrame(rows)
    score = float(detail["선택 응답 반영점수"].sum()) if not detail.empty else 0.0
    return min(100.0, score), detail


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
    comment_html = "<br>".join(str(comment).replace("<br>", "\n").splitlines())
    st.markdown(
        f'''<div class="report-kicker">개별 관찰 결과</div>
        <div class="report-score">{int(score)} / 100점</div>''',
        unsafe_allow_html=True,
    )
    if title:
        st.markdown(f'<div class="report-band">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-comment">{comment_html}</div>', unsafe_allow_html=True)


def environment_result_report_header(student_name, score, title, comment):
    """생활환경 설문 결과를 ASD 설문 결과 카드와 같은 구조로 표시한다."""
    # 문장 데이터에는 HTML 태그를 넣지 않고, 화면 출력용 줄바꿈만 여기서 만든다.
    comment_html = "<br>".join(str(comment).splitlines())
    st.markdown(
        f'''<div class="report-kicker">개별 관찰 결과</div>
        <div class="report-score">{float(score):.1f} / 100점</div>
        <div class="report-comment">{comment_html}</div>''',
        unsafe_allow_html=True,
    )


def environment_result_state(score, score_bands):
    """ACE 4개 이상 위험신호 점수의 검증 구간에 맞는 제목과 설명을 고른다."""
    index = next(
        (i for i, (_, row) in enumerate(score_bands.iterrows()) if float(score) < float(row["upper_score"]) or i == len(score_bands) - 1),
        len(score_bands) - 1,
    )
    states = [
        (
            "낮은 위험신호 구간",
            "현재 응답에서 가족·주거·학교생활 관련 항목은 ACE 4개 이상 그룹에서 두드러졌던 생활환경 패턴과 비교적 거리가 있습니다.\n현재 확인한 생활환경 특성을 평소와 같이 관찰해 주세요.",
        ),
        (
            "주의 위험신호 구간",
            "현재 응답에서 일부 생활환경 항목이 ACE 4개 이상 그룹에서 함께 나타난 응답 패턴과 유사하게 확인됩니다.\n생활 리듬과 가정·학교에서의 변화가 반복되는지 조금 더 기록해 주세요.",
        ),
        (
            "높은 위험신호 구간",
            "현재 응답에서 여러 생활환경 항목이 ACE 4개 이상 그룹에서 상대적으로 많이 나타난 응답 패턴과 유사하게 확인됩니다.\n응답 내용을 다시 확인하고, 관찰한 생활환경 정보를 보호자와 함께 살펴보세요.",
        ),
        (
            "매우 높은 위험신호 구간",
            "현재 응답에서 가족·주거·학교생활 관련 위험신호가 함께 나타나는 패턴이 비교적 뚜렷하게 확인됩니다.\n관찰한 내용과 생활환경 정보를 보호자 및 담당자와 우선 공유해 필요한 지원을 함께 확인해 주세요.",
        ),
    ]
    title, comment = states[min(index, len(states) - 1)]
    return index, title, comment


def environment_action_list(score, score_bands):
    """현재 생활환경 가중점수를 0~100의 네 점수 구간으로 표시한다."""
    labels = ["낮은 위험신호", "주의 위험신호", "높은 위험신호", "매우 높은 위험신호"]
    positions = []
    actions = []
    for i, (_, row) in enumerate(score_bands.iterrows()):
        lower = float(row["lower_score"])
        upper = float(row["upper_score"])
        is_last = i == len(score_bands) - 1
        positions.append(f"{lower:.0f}~{upper:.0f}점 {'이하' if is_last else '미만'}")
        actions.append(f"생활환경 가중점수가 {lower:.0f}점 이상 {upper:.0f}점 {'이하' if is_last else '미만'}인 구간입니다.")
    rows = ['<div class="action-list-wrap"><div class="action-list-title">생활환경 점수 구간</div>']
    for i, (_, row) in enumerate(score_bands.iterrows()):
        lower = float(row["lower_score"])
        upper = float(row["upper_score"])
        is_last = i == len(score_bands) - 1
        active = " active" if (lower <= float(score) <= upper if is_last else lower <= float(score) < upper) else ""
        rows.append(f"<div class='action-row{active}'><b>{positions[min(i, len(positions)-1)]} · {labels[min(i, len(labels)-1)]}</b><br>{actions[min(i, len(actions)-1)]}</div>")
    rows.append('</div>')
    st.markdown(''.join(rows), unsafe_allow_html=True)


def environment_score_position_chart(score, score_bands):
    """현재 생활환경 가중점수가 0~100 중 어느 구간인지 보여 주는 그래프다."""
    colors = ["#edf7ef", "#dcefe0", "#c8e5ce", "#b3d9bc"]
    fig, ax = plt.subplots(figsize=(2.30, 1.92))
    for i, (_, row) in enumerate(score_bands.iterrows()):
        lower = float(row["lower_score"])
        upper = float(row["upper_score"])
        ax.barh(0, upper - lower, left=lower, height=.42, color=colors[i], edgecolor="#8ebc98", linewidth=.55)
        ax.text((lower + upper) / 2, 0, str(i + 1), ha="center", va="center", fontsize=6.5, color="#356047", fontweight="bold")
    ax.axvline(float(score), color="#3f7750", linewidth=1.6)
    ax.scatter([float(score)], [0], s=22, color="#3f7750", zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-.42, .42)
    ax.set_yticks([])
    score_ticks = sorted(set(score_bands["lower_score"].astype(float).tolist() + [float(score_bands["upper_score"].iloc[-1])]))
    ax.set_xticks(score_ticks)
    ax.set_xlabel("생활환경 위험신호 점수", fontsize=6.5, color="#526b5a")
    ax.tick_params(axis="x", labelsize=5.8, colors="#687c6e")
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#cfe4d5")
    fig.tight_layout(pad=.7)
    st.pyplot(fig, width="content")
    plt.close(fig)


def make_summary_pdf(student_name, asd_score, env_score, total_score, verdict, summary_label):
    """종합 결과 화면에 표시되는 내용을 같은 순서로 2페이지 PDF에 담는다."""
    output = BytesIO()
    with PdfPages(output) as pdf:
        who = student_name.strip() if student_name and student_name.strip() else "해당 아동"
        def heading(fig, title, subtitle):
            fig.text(.10, .94, title, fontsize=20, fontweight="bold", color="#243746")
            fig.text(.10, .912, subtitle, fontsize=10.5, color="#52616d")

        # 1페이지: 화면 상단의 이름·두 점수·종합 결과를 그대로 배치한다.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
        heading(fig, "아동 관찰 결과", f"아이 이름: {who}")
        fig.text(.10, .855, "종합 결과", fontsize=14, fontweight="bold", color="#315b78")
        fig.add_artist(FancyBboxPatch((.10, .765), .80, .065, boxstyle="round,pad=.014,rounding_size=.012", facecolor="#fff3e4", edgecolor="#f0d2ad", transform=fig.transFigure))
        fig.text(.15, .802, f"ASD 행동점수    {asd_score:.1f} / 100", fontsize=13, fontweight="bold", color="#8b5a22", va="center")
        fig.add_artist(FancyBboxPatch((.10, .675), .80, .065, boxstyle="round,pad=.014,rounding_size=.012", facecolor="#edf5fb", edgecolor="#c9ddeb", transform=fig.transFigure))
        fig.text(.15, .712, f"생활환경 위험신호 점수    {env_score:.1f} / 100", fontsize=13, fontweight="bold", color="#3f7750", va="center")
        fig.add_artist(FancyBboxPatch((.10, .525), .80, .105, boxstyle="round,pad=.014,rounding_size=.012", facecolor="#f5f8fb", edgecolor="#7f9db6", linewidth=1.4, transform=fig.transFigure))
        fig.text(.50, .595, "종합 관찰 결과", fontsize=11, fontweight="bold", color="#647a8d", ha="center")
        fig.text(.50, .558, summary_label, fontsize=18, fontweight="bold", color="#c47722", ha="center")
        fig.text(.50, .535, f"영역별 확인: {verdict}", fontsize=10.5, color="#52616d", ha="center")
        fig.text(.10, .43, f"최종 종합점수: {total_score:.1f} / 100", fontsize=15, fontweight="bold", color="#315b78")
        fig.text(.10, .34, "최종 점수 구간별 관찰 안내", fontsize=14, fontweight="bold", color="#315b78")
        guidance = {
            "일반 관찰": ("0점 이상 ~ 25점 미만", "현재 관찰된 특성이 낮은 수준입니다. 평소와 같이 관찰합니다."),
            "관찰 강화": ("25점 이상 ~ 50점 미만", "행동이 나타난 상황·빈도·지속시간을 기록하고 보호자와 공유합니다."),
            "추가 확인 권고": ("50점 이상 ~ 75점 미만", "기록 내용을 보호자와 담당자에게 공유하고 추가 확인을 권합니다."),
            "전문상담 우선 권고": ("75점 이상 ~ 100점 이하", "관찰 내용을 공유하고 관련 전문가 또는 전문기관 상담을 안내합니다."),
        }
        y = .275
        for label, (range_text, detail) in guidance.items():
            active = label == summary_label
            fig.add_artist(FancyBboxPatch((.10, y), .80, .045, boxstyle="round,pad=.006,rounding_size=.006", facecolor="#eaf6ed" if active else "#fbfcfd", edgecolor="#a8d3b4" if active else "#dce4ec", transform=fig.transFigure))
            fig.text(.12, y + .029, f"{range_text} · {label}", fontsize=7.2, fontweight="bold", color="#315742", va="center")
            fig.text(.12, y + .011, detail, fontsize=6.1, color="#53626d", va="center")
            y -= .052
        fig.text(.10, .055, "종합점수 계산 방법", fontsize=10, fontweight="bold", color="#315b78")
        fig.text(.10, .035, "종합점수 = (ASD 행동점수 + 생활환경 위험신호 점수) ÷ 2 · 두 영역을 같은 비중으로 정리한 프로젝트 참고점수입니다.", fontsize=6.3, color="#405462")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    output.seek(0)
    return output.getvalue()


def preserve_checklist_scroll_position():
    """실시간 입력으로 화면이 재실행돼도 체크리스트를 보던 높이로 되돌린다."""
    components.html(
        """<script>
        const key = 'tp1-checklist-scroll-position';
        const parentWindow = window.parent;
        const saved = parentWindow.sessionStorage.getItem(key);
        if (saved !== null) {
            parentWindow.requestAnimationFrame(() => parentWindow.scrollTo(0, Number(saved)));
        }
        if (!parentWindow.__tp1ChecklistScrollListener) {
            parentWindow.addEventListener('scroll', () => {
                parentWindow.sessionStorage.setItem(key, String(parentWindow.scrollY || 0));
            }, {passive: true});
            parentWindow.__tp1ChecklistScrollListener = true;
        }
        </script>""",
        height=0,
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
sidebar_options = [
    "1. 프로젝트 개요",
    "2. 데이터 전처리",
    "3. 연관성 확인",
    "4. 머신러닝 모델",
    "5. 모델 성능 평가",
    "6. 가중치 산출",
    "7. 결과",
    "NSCH 외부데이터 분석",
]
sidebar_current = st.session_state.get("sidebar_current", "1. 프로젝트 개요")
sidebar_index = sidebar_options.index(sidebar_current) if sidebar_current in sidebar_options else 0
sidebar_choice = st.sidebar.radio(
    "메뉴",
    sidebar_options,
    index=sidebar_index,
    label_visibility="collapsed",
    key="sidebar_menu_radio",
)
st.session_state["sidebar_current"] = sidebar_choice
menu = "8. NSCH 외부데이터 분석" if sidebar_choice == "NSCH 외부데이터 분석" else sidebar_choice


# ============================================================
# 6. Page 1 - 프로젝트 개요
# ============================================================
if menu.startswith("1."):
    page_title("아동학대 사전 예측 솔루션", "아동학대를 조기에 발견하여 사전에 방지하는 시스템")
    st.markdown("<div class='project-flow-heading'>프로젝트 플로우</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="project-flow-wrap">
          <div class="project-flow-card"><div class="project-flow-title">문제 정의</div><div class="project-flow-text">아동학대 위험신호 조기 확인과 관찰 우선도 지원 목적 설정</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card data"><div class="project-flow-title">데이터 취합</div><div class="project-flow-text">UCI 행동 데이터와 NSCH 2024 생활환경 데이터 취합</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card"><div class="project-flow-title">데이터 전처리</div><div class="project-flow-text">분석 대상 정리 · 결측값 처리 · 범주형 입력 변환</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card"><div class="project-flow-title">관계·유의성 분석</div><div class="project-flow-text">카이제곱 검정 · Cramér’s V · p-value로 관계 크기와 유의성 확인</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card"><div class="project-flow-title">분류모델 비교</div><div class="project-flow-text">로지스틱 회귀 · K-최근접 이웃 · 의사결정나무 · 랜덤 포레스트 비교</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card"><div class="project-flow-title">성능평가·모델선택</div><div class="project-flow-text">정확도 · 재현율 · F1 점수 · ROC-AUC 기준 최종모델 선택</div></div>
          <div class="project-flow-arrow">→</div>
          <div class="project-flow-card final"><div class="project-flow-title">종합 관찰 결과·웹 서비스</div><div class="project-flow-text">행동·생활환경 점수 기반 종합 관찰 결과와 대응 방향 제공</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='analysis-stage-heading'>데이터 분석 단계</div>", unsafe_allow_html=True)

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
    page_title("4. 머신러닝 모델", "학습자료로 4개 분류 모델을 학습하고, 검증자료 성능을 비교해 최종 모델을 선택한다.")
    pipeline(3)

    st.subheader("모델 학습과 선택 방법")
    plain_list(
        [
            ("독립변수 X", "A1~A10 행동 응답 — 모델이 ASD 선별 YES/NO를 구분할 때 사용하는 입력값"),
            ("종속변수 y", "ASD 선별 결과 — YES와 NO 두 범주를 예측하므로 회귀가 아닌 분류 문제로 설정"),
            ("학습자료", f"{int(meta.get('selection_train_rows', 174))}명 — 각 모델이 행동 응답과 선별 결과의 패턴을 학습"),
            ("검증자료", f"{int(meta.get('validation_rows', 58))}명 — 학습에 사용하지 않은 동일 자료로 4개 모델을 공정하게 비교"),
        ]
    )

    st.subheader("4개 분류 모델의 검증자료 성능 비교")
    st.caption("로지스틱 회귀 · K-최근접 이웃 · 의사결정나무 · 랜덤 포레스트를 같은 입력 항목과 같은 검증자료로 비교했습니다.")
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
        plot_df = model_compare.sort_values("f1", ascending=True)
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        labels = plot_df["model"].replace(
            {"Logistic Regression": "Logistic", "Decision Tree": "Tree", "Random Forest": "RF"}
        )
        colors = ["#88b895" if m == "Logistic Regression" else "#c8d1d8" for m in plot_df["model"]]
        bars = ax.barh(labels, plot_df["f1"], color=colors)
        ax.set_xlim(0.75, 1.03)
        ax.set_xlabel("검증자료 F1 점수")
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

        best = model_compare.sort_values(["selection_rank", "f1", "recall"], ascending=[True, False, False]).iloc[0]
        st.subheader("최종 모델 선택")
        st.markdown(
            f'<div class="selected-model-note"><b>✓ {model_name(best["model"])}</b> · '
            '검증자료에서 정확도·정밀도·재현율·F1 점수·ROC-AUC를 함께 비교한 결과 1순위로 선택</div>',
            unsafe_allow_html=True,
        )
        result_line(
            f"검증자료 기준 최종 선택 모델: <b>{model_name(best['model'])}</b> &nbsp; | &nbsp; "
            f"F1 {fmt(best['f1'])} / Recall {fmt(best['recall'])} / ROC-AUC {fmt(best['roc_auc'])}"
        )


# ============================================================
# 10. Page 5 - 모델 성능 평가
# ============================================================
elif menu.startswith("5."):
    page_title("5. 모델 성능 평가", "4번에서 선택한 최종 모델을 시험자료에 한 번만 적용해 일반화 성능을 확인한다.")
    pipeline(4)

    st.subheader("최종 시험자료 평가")
    selection_train_rows = int(meta.get("selection_train_rows", 174))
    validation_rows = int(meta.get("validation_rows", 58))
    final_test_rows = int(meta.get("final_test_rows", 58))
    st.markdown(
        f'''<div class="split-diagram">
            <div class="split-root-row"><div class="split-box root"><strong>분석 가능 데이터 {len(analysis)}명</strong><br>중복 제거·전처리 완료</div></div>
            <div class="split-branch-arrows"><span>↓</span></div>
            <div class="split-branches" style="grid-template-columns:repeat(3, 1fr); width:82%;">
                <div class="split-box train"><strong>학습자료 · {selection_train_rows}명</strong><br>4개 모델 학습</div>
                <div class="split-box train"><strong>검증자료 · {validation_rows}명</strong><br>모델 비교·선택</div>
                <div class="split-box test"><strong>시험자료 · {final_test_rows}명</strong><br>선택 모델 최종 평가</div>
            </div>
        </div>''',
        unsafe_allow_html=True,
    )
    st.caption("시험자료는 모델 학습이나 선택에 사용하지 않고, 최종 모델을 정한 뒤 마지막 평가에만 사용했습니다.")

    st.markdown('<div style="height:.70rem"></div>', unsafe_allow_html=True)
    selected_model_label = model_name(meta.get("selected_model", "Logistic Regression"))
    st.subheader(f"{selected_model_label} 최종 성능")

    if not final_row.empty:
        metric_table = pd.DataFrame(
            [
                ["정확도(Accuracy)", final_row.get("accuracy"), "전체 중 맞힌 비율"],
                ["정밀도(Precision)", final_row.get("precision"), "YES라고 분류한 것 중 실제 YES 비율"],
                ["재현율(Recall)", final_row.get("recall"), "실제 YES 중 찾아낸 비율"],
                ["F1 점수(F1-score)", final_row.get("f1"), "Precision과 Recall의 균형"],
                ["ROC-AUC", final_row.get("roc_auc"), "YES와 NO를 전반적으로 구분하는 능력"],
            ],
            columns=["평가 지표", "시험자료 결과", "설명"],
        )
        metric_table["시험자료 결과"] = metric_table["시험자료 결과"].map(lambda x: fmt(x))
        left_table(metric_table, .82)
        result_line(
            f"시험자료 기준 <b>{selected_model_label}</b> 성능: "
            f"정확도 {fmt(final_row.get('accuracy'))} / 재현율 {fmt(final_row.get('recall'))} / "
            f"F1 {fmt(final_row.get('f1'))} / ROC-AUC {fmt(final_row.get('roc_auc'))}"
        )

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
                f"<b>결과:</b> 시험자료 {tp+tn+fp+fn}명 중 {tp+tn}명을 정확히 분류했다."
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
                    f"<b>현재 로지스틱 회귀 시험자료 결과: ROC-AUC {auc:.3f}</b>"
                )

        st.subheader("평가 결과 해석")
        explain_card(
            "시험자료에서도 모든 평가지표가 높아, 선택된 모델이 이 데이터의 ASD 선별 규칙을 안정적으로 재현했습니다.<br>"
            "다만 원래 ASD 선별 결과가 A1~A10 응답 합계 규칙으로 만들어졌고 모델도 같은 A1~A10을 입력으로 사용하므로, "
            "높은 성능은 새로운 독립 요인을 발견했다기보다 기존 선별 규칙을 정확히 학습한 결과로 해석해야 합니다."
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
        weight_table_html = wtbl.to_html(index=False, classes="behavior-weight-table", border=0)
        weight_table_cols = st.columns([.74, .26], gap="small")
        with weight_table_cols[0]:
            st.markdown(weight_table_html, unsafe_allow_html=True)

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
# 12. Page 7 - 결과
# ============================================================
elif menu.startswith("8."):
    render_nsch_ace4_page()
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
    page_title("7. 결과", "ASD 행동 특성과 ACE 4개 이상 경험에 관련된 생활환경 위험신호를 각각 확인하고 종합 결과를 봅니다.")
    # 최초 1회만 스크롤 복원 리스너를 붙여 입력 때마다 iframe이 다시 그려지는 깜빡임을 줄인다.
    if not st.session_state.get("checklist_scroll_listener_ready", False):
        preserve_checklist_scroll_position()
        st.session_state["checklist_scroll_listener_ready"] = True
    pipeline(6)
    st.markdown("### 체크리스트")
    pending_tab = st.session_state.pop("checklist_target_tab", None)
    checklist_tab_labels = ["1. ASD 행동설문", "2. 생활환경 설문", "3. 종합 결과"]
    # Streamlit 탭 상태를 session_state로 직접 관리한다.
    # 버튼에서 이 값을 바꾼 뒤 rerun하면 다음 탭이 서버 상태에 맞춰 확실히 열린다.
    if pending_tab is not None:
        requested_tab = max(0, min(int(pending_tab), len(checklist_tab_labels) - 1))
        st.session_state["checklist_active_tab"] = checklist_tab_labels[requested_tab]
    elif "checklist_active_tab" not in st.session_state:
        st.session_state["checklist_active_tab"] = checklist_tab_labels[0]
    survey_tabs = st.tabs(
        checklist_tab_labels,
        key="checklist_active_tab",
        on_change="rerun",
    )
    with survey_tabs[0]:
        # st.empty()는 재실행 시작과 동시에 기존 설문을 지워 화면이 꺼졌다 켜지는 것처럼 보인다.
        # 일반 컨테이너는 기존 화면을 유지한 채 변경된 점수와 그래프만 교체해 깜빡임을 줄인다.
        asd_checklist_slot = st.container()
    with survey_tabs[1]:
        st.subheader("생활환경 설문")
        st.markdown('<div class="survey-section-spacer"></div>', unsafe_allow_html=True)
        env_form_col, env_result_col = st.columns([.57, .43], gap="large")
        with env_form_col:
            st.markdown('<span class="environment-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown('<div class="environment-title">아동학대 의심 설문조사 (생활환경)</div>', unsafe_allow_html=True)
            nsch_model_path = NSCH_RISK_ART / "final_model_nsch_ace4.pkl"
            nsch_ui_model = load_pickle_model(str(nsch_model_path), nsch_model_path.stat().st_mtime_ns)
            # 첫 화면에서는 어떤 응답도 미리 선택하지 않는다.
            # 선택한 문항만 즉시 점수에 반영하고, 8개를 모두 선택하면 최종 결과를 볼 수 있다.
            nsch_values = {}
            visible_features = NSCH_RISK_FEATURES
            for number, x in enumerate(visible_features, start=1):
                question = NSCH_SURVEY_QUESTIONS.get(x, NSCH_LABELS[x])
                if x in NSCH_CODE_OPTIONS:
                    options = [None, *NSCH_CODE_OPTIONS[x]]
                    selected_option = st.selectbox(
                        f"{number}. {question}", options,
                        format_func=lambda value: "선택하세요" if value is None else value[1],
                        key=f"integrated_{x}_v4",
                    )
                    nsch_values[x] = np.nan if selected_option is None else selected_option[0]
            st.session_state["nsch_environment_answers"] = nsch_values.copy()
            answered_count = sum(pd.notna(value) for value in nsch_values.values())
            all_environment_answers = answered_count == len(visible_features)
            st.session_state["nsch_environment_answer_count"] = answered_count
            # 저장 파이프라인이 학습 때 받은 전체 열을 만들고, 실제 선택 항목 외 열은 NaN으로 둔다.
            nsch_model_input = pd.DataFrame(
                [{column: nsch_values.get(column, np.nan) for column in nsch_ui_model.feature_names_in_}]
            )
            nsch_raw_probability = float(nsch_ui_model.predict_proba(nsch_model_input)[0, 1])
            st.session_state["nsch_environment_raw_probability"] = nsch_raw_probability
            weighted_response_score, weighted_response_detail = nsch_weighted_response_summary(nsch_values)
            # 주 점수는 각 문항 가중치와 선택 응답 반영점수를 더한 값이다.
            # 따라서 어떤 문항을 바꿔도 해당 문항의 반영점수만큼 항상 점수가 변한다.
            st.session_state["nsch_environment_score"] = weighted_response_score
            st.session_state["nsch_environment_percentile"] = nsch_environment_percentile(weighted_response_score)
            st.session_state["nsch_environment_weighted_score"] = weighted_response_score
            st.session_state["nsch_environment_weighted_detail"] = weighted_response_detail.to_dict("records")
            env_back_col, env_submit_col = st.columns(2, gap="small")
            with env_back_col:
                return_to_asd = st.button("이전 설문으로 돌아가기", use_container_width=True, key="environment_back")
            with env_submit_col:
                submitted = st.button("제출 및 결과보기", type="primary", use_container_width=True, key="environment_submit", disabled=not all_environment_answers)
            if submitted:
                # 실시간으로 계산된 현재 점수를 그대로 종합 결과 탭에 사용한다.
                st.session_state["environment_survey_saved"] = True
                st.session_state["checklist_target_tab"] = 2
                st.rerun()
            elif return_to_asd:
                st.session_state["checklist_target_tab"] = 0
                st.rerun()
        with env_result_col:
            env_now = st.session_state.get("nsch_environment_score")
            answered_count = int(st.session_state.get("nsch_environment_answer_count", 0))
            all_environment_answers = answered_count == len(NSCH_RISK_FEATURES)
            # 결과 영역은 ASD 행동패턴 결과와 같은 흰색 카드·녹색 강조 형식을 사용한다.
            st.markdown('<span class="result-panel-marker"></span>', unsafe_allow_html=True)
            if answered_count:
                environment_score_bands = pd.read_csv(NSCH_RISK_ART / "environment_score_bands.csv")
                env_score = float(env_now)
                _, env_title, env_comment = environment_result_state(env_score, environment_score_bands)
                if not all_environment_answers:
                    env_title = ""
                    env_comment = f"현재 {answered_count} / {len(NSCH_RISK_FEATURES)}문항 응답 기준의 중간 점수입니다.\n나머지 문항을 모두 선택하면 최종 생활환경 결과가 표시됩니다."
                environment_result_report_header(
                    st.session_state.get("student_name", ""), env_now, env_title, env_comment
                )
                if all_environment_answers:
                    environment_action_list(env_score, environment_score_bands)
                weighted_now = float(st.session_state.get("nsch_environment_weighted_score", 0.0))
                st.markdown(
                    f"<div class='result-line'><span style='font-weight:500'>문항 가중 반영도</span> · {weighted_now:.1f} / 100<br>"
                    "각 문항의 Random Forest 가중치와 선택 응답의 위험군 차이를 합산한 생활환경 점수입니다.</div>",
                    unsafe_allow_html=True,
                )
                mini_note("100점은 8개 문항 모두에서 학습자료상 위험군 응답 차이가 가장 컸던 선택지를 고른 경우에만 나옵니다. 각 문항의 선택지가 겉으로 비슷하게 우려되어 보여도 실제 두 집단의 응답 차이가 작으면 해당 문항 점수는 낮게 반영됩니다.")
                with st.expander("문항별 가중 반영 보기"):
                    weighted_detail = pd.DataFrame(st.session_state.get("nsch_environment_weighted_detail", []))
                    if not weighted_detail.empty:
                        weighted_detail["문항 가중치"] = weighted_detail["문항 가중치"].map(lambda value: f"{float(value):.1f}점")
                        weighted_detail["선택 응답 반영점수"] = weighted_detail["선택 응답 반영점수"].map(lambda value: f"{float(value):.1f}점")
                        st.dataframe(weighted_detail, hide_index=True, width="stretch")
                    mini_note("문항 가중치는 Train 자료의 Random Forest 중요도를 100점으로 환산했습니다. 선택 응답 반영점수는 ACE 4개 이상 위험군에서 비교집단보다 더 많이 나타난 정도를 적용했으며, 식품 상황·학교 연락·주거 퇴거 걱정·따돌림처럼 응답 순서가 있는 문항은 응답 강도가 높을수록 점수가 커지도록 반영했습니다.")
                if all_environment_answers:
                    st.markdown('<div class="report-section-title">생활환경 점수 위치</div>', unsafe_allow_html=True)
                    environment_score_position_chart(env_score, environment_score_bands)
                    st.markdown('<div class="radar-note">그래프의 표시점은 위에 나온 생활환경 점수와 동일한 값입니다. 네 구간 중 높은 위험신호의 시작점은 검증자료에서 F1 점수가 가장 높았던 기준을 사용합니다.</div>', unsafe_allow_html=True)
                with st.expander("검증자료 구간 기준"):
                    criteria_view = environment_score_bands.copy()
                    criteria_view["점수 구간"] = criteria_view.apply(
                        lambda row: f"{float(row['lower_score']):.0f}점 이상~{float(row['upper_score']):.0f}점 {'이하' if float(row['upper_score']) == 100 else '미만'}",
                        axis=1,
                    )
                    criteria_view["검증자료 인원"] = criteria_view["validation_count"].map(lambda value: f"{int(value):,}명")
                    criteria_view["ACE 4개 이상 비율"] = criteria_view["observed_ace4_rate"].map(lambda value: f"{float(value)*100:.1f}%")
                    criteria_view["평균 ACE 개수"] = criteria_view["observed_mean_ace_count"].map(lambda value: f"{float(value):.2f}개")
                    st.dataframe(criteria_view[["점수 구간", "score_band", "검증자료 인원", "ACE 4개 이상 비율", "평균 ACE 개수"]].rename(columns={"score_band": "위험신호 구간"}), hide_index=True, width="stretch")
            else:
                st.markdown("<div class='explain-card'><b>결과 안내</b><br>왼쪽 8개 질문에 응답하면 점수가 바로 표시됩니다.</div>", unsafe_allow_html=True)
    with survey_tabs[2]:
        # ASD는 기존 가중 체크리스트 합계, 환경 점수는 RF 문항 가중치와 응답별 위험신호 반영점수를 더한 값이다.
        # 동일한 아동을 두 데이터에서 함께 관찰한 자료가 없으므로, 별도 결합 모델 없이 두 점수를 같은 비중으로 평균한다.
        env = float(st.session_state.get("nsch_environment_score", 0.0))
        # 저장하고 다음 설문하기를 누른 뒤에는 확정한 ASD 점수를 사용한다.
        # 저장 전이거나 이전 세션이면 기존 체크박스 합산값을 안전한 기본값으로 사용한다.
        asd_fallback = sum(
            int(r["points"])
            for _, r in weighted_checklist[weighted_checklist["feature"].isin(BEHAVIOR)].head(10).iterrows()
            if st.session_state.get(f"teacher_check_{r['feature']}", False)
        ) if not weighted_checklist.empty else 0
        asd = float(st.session_state.get("asd_behavior_score", asd_fallback))
        asd_pct = min(100.0, asd)
        avg = (asd_pct + env) / 2
        environment_score_bands = pd.read_csv(NSCH_RISK_ART / "environment_score_bands.csv")
        survey_cutoff = float(environment_score_bands.iloc[2]["lower_score"])
        high = asd >= int(meta.get("weighted_checklist", {}).get("high_cutoff", 65)); env_high = env >= survey_cutoff
        verdict = "복합 추가확인" if high and env_high else "ASD 행동 집중 관찰" if high else "생활환경 위험신호 추가확인" if env_high else "일반 관찰"
        asd_observe_cutoff = int(meta.get("weighted_checklist", {}).get("observe_cutoff", 50))
        asd_high_cutoff = int(meta.get("weighted_checklist", {}).get("high_cutoff", 65))
        asd_very_high_cutoff = int(meta.get("weighted_checklist", {}).get("very_high_cutoff", 80))
        asd_band = (
            "ASD 선별 매우 고관찰 구간" if asd >= asd_very_high_cutoff else
            "ASD 선별 고관찰 구간" if asd >= asd_high_cutoff else
            "추가 관찰 구간" if asd >= asd_observe_cutoff else
            "일반 관찰 구간"
        )
        _, env_band, _ = environment_result_state(env, environment_score_bands)
        verdict_text = "ASD 행동 특성과 ACE 4개 이상 위험신호가 모두 높게 나타났습니다." if high and env_high else "ASD 행동 특성은 높고 ACE 4개 이상 위험신호는 낮게 나타났습니다." if high else "ACE 4개 이상 위험신호는 높고 ASD 행동 특성은 낮게 나타났습니다." if env_high else "ASD 행동 특성과 ACE 4개 이상 위험신호가 모두 낮게 나타났습니다."
        if avg < 25:
            summary_label = "일반 관찰"
            summary_action = "현재 ASD 행동 특성과 ACE 4개 이상 경험에 관련된 생활환경 위험신호가 모두 낮게 나타났습니다. 평소와 같이 아이의 놀이, 의사소통, 또래관계와 일상 적응을 관찰해 주세요."
        elif avg < 50:
            summary_label = "관찰 강화"
            summary_action = "특정 행동이 반복되는지 조금 더 자세히 확인하고, 약 2~4주 동안 나타난 상황·빈도·지속시간을 객관적으로 기록해 보호자와 관찰 사실을 공유해 주세요."
        elif avg < 75:
            summary_label = "추가 확인 권고"
            summary_action = "기록한 내용을 보호자와 담당자에게 공유하고, 필요한 경우 훈련받은 담당자의 표준화된 발달 선별검사 또는 소아청소년 발달 전문가와의 상담을 권합니다."
        else:
            summary_label = "전문상담 우선 권고"
            summary_action = "보호자에게 관찰 내용을 신속하게 공유하고 소아청소년과, 발달 전문가 또는 관련 전문기관에 상담할 수 있도록 안내해 주세요. 이 결과만으로 아동을 ASD라고 판단하거나 낙인찍지 않도록 합니다."

        # 좌우 여백을 두어 기존보다 약 15% 좁은 두 카드를 가운데에 배치한다.
        _, card1, card2, _ = st.columns([.15, .35, .35, .15], gap="large")
        with card1:
            st.markdown(f"<div class='combined-card asd'><div class='combined-card-label'>🧩 ASD 행동점수</div><div class='combined-card-score'>{asd_pct:.0f} / 100</div><div class='combined-card-band'>{asd_band}</div></div>", unsafe_allow_html=True)
        with card2:
            st.markdown(f"<div class='combined-card env'><div class='combined-card-label'>🏡 생활환경 위험신호 점수</div><div class='combined-card-score'>{env:.0f} / 100</div><div class='combined-card-band'>{env_band}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;font-size:1.4rem;color:#7f9db6;margin:.35rem 0'>↓</div>", unsafe_allow_html=True)
        child_name = st.session_state.get("student_name", "").strip() or "미입력"
        st.markdown(f"<div class='combined-result'><div class='combined-result-title'>종합 관찰 결과</div><div class='combined-result-value'>🟠 {summary_label}</div><div class='combined-result-text' style='width:60%;margin-left:auto;margin-right:auto;line-height:1.75'><b>아이 이름: {child_name}</b><br><span style='display:inline-block;margin-top:.55rem;font-size:1.08rem;font-weight:800;color:#315b78'>최종 종합점수: {avg:.1f} / 100</span><br><br>{summary_action}<br><br><b>영역별 확인: {verdict}</b><br>{verdict_text}</div></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class='criteria-wrap'>
            <div class='criteria-heading'>최종 점수 구간별 관찰 안내</div>
            <div class='criteria-panel'>
              <div class='criteria-row{' current' if avg < 25 else ''}'><b>0점 이상 ~ 25점 미만 · 일반 관찰</b><br>현재 관찰된 특성이 낮은 수준입니다. 평소와 같이 놀이, 의사소통, 또래관계와 일상 적응을 관찰하고, 이전과 다른 변화가 반복되면 날짜와 상황을 간단히 기록합니다.</div>
              <div class='criteria-row{' current' if 25 <= avg < 50 else ''}'><b>25점 이상 ~ 50점 미만 · 관찰 강화</b><br>약 2~4주 동안 행동이 나타난 상황, 빈도와 지속시간을 객관적으로 기록하고 보호자와 관찰 사실을 공유합니다.</div>
              <div class='criteria-row{' current' if 50 <= avg < 75 else ''}'><b>50점 이상 ~ 75점 미만 · 추가 확인 권고</b><br>기록한 내용을 보호자와 담당자에게 공유하고, 필요한 경우 표준화된 발달 선별검사 또는 발달 전문가와의 상담을 권합니다.</div>
              <div class='criteria-row{' current' if 75 <= avg <= 100 else ''}'><b>75점 이상 ~ 100점 이하 · 전문상담 우선 권고</b><br>보호자에게 관찰 내용을 신속하게 공유하고 관련 전문가 또는 전문기관 상담을 안내합니다. 이 결과만으로 아동을 ASD라고 판단하거나 낙인찍지 않도록 합니다.</div>
            </div>
            <div class='combine-algorithm'>
              <div class='combine-algorithm-title'>종합점수 계산 방법</div>
              <div style='font-size:.76rem;line-height:1.65;color:#4b606f;text-align:left'>
                ASD 행동점수는 A1~A10 문항의 기존 가중치를 더해 계산합니다. 생활환경 위험신호 점수는 4~11세 NSCH 자료에서 ACE 4개 이상 여부를 비교한 뒤, Random Forest 중요도로 계산한 문항 가중치와 선택 응답의 위험군 차이를 더해 계산합니다. 따라서 각 문항 응답은 점수에 독립적으로 반영됩니다.<br><br>두 원자료는 같은 아동을 함께 조사한 자료가 아니므로, 두 예측값과 실제 정답으로 학습한 결합·스태킹 모델은 현재 존재하지 않습니다.<br><br>
                <b>종합점수 = (ASD 행동점수 + 생활환경 위험신호 점수) ÷ 2</b><br><br>
                따라서 최종 종합점수는 두 영역을 같은 비중으로 정리한 프로젝트 참고점수이며, 두 영역이 각각 높은지 여부는 위의 ‘영역별 확인’ 2×2 결과로 별도로 표시합니다.
              </div>
            </div>
            <div class='warning-line'>이 결과는 관찰과 상담의 우선순위를 돕기 위한 참고정보이며, 의학적 진단 결과가 아닙니다. 점수가 낮더라도 발달 퇴행이나 뚜렷한 우려가 관찰되면 보호자와 상의하여 전문가 상담을 권장합니다.</div>
            </div></div>""",
            unsafe_allow_html=True,
        )
        # 현재 종합 결과 값을 그대로 1페이지 PDF 보고서로 만들어 다운로드한다.
        summary_pdf = make_summary_pdf(
            st.session_state.get("student_name", ""),
            float(asd_pct),
            float(env),
            float(avg),
            verdict,
            summary_label,
        )
        pdf_left, pdf_button, pdf_right = st.columns([.30, .40, .30], gap="small")
        with pdf_button:
            st.download_button(
                "결과 출력하기 (PDF)",
                data=summary_pdf,
                file_name="아동_관찰_결과.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
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
        asd_next = False

        with check_col:
            st.markdown('<span class="survey-panel-marker"></span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="survey-title">아동학대 의심 설문조사 (행동패턴)</div>',
                unsafe_allow_html=True,
            )
            # 일반 체크박스로 두어 문항을 누를 때마다 오른쪽 점수와 결과가 바로 갱신되게 한다.
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
            st.markdown('<div style="height:.45rem"></div>', unsafe_allow_html=True)
            next_left, next_button, next_right = st.columns([.35, .30, .35], gap="small")
            with next_button:
                asd_next = st.button("저장하고 다음설문하기", type="primary", use_container_width=True, key="asd_next")

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

        if asd_next:
            # 생활환경 설문으로 이동하기 전에 행동 설문의 현재 점수와 선택 문항을 확정해 둔다.
            # 종합 결과는 이 저장값을 우선 사용하므로 탭을 이동해도 ASD 점수가 사라지지 않는다.
            st.session_state["asd_behavior_score"] = float(total_score)
            st.session_state["asd_behavior_answers"] = checked_items.copy()
            st.session_state["asd_survey_saved"] = True
            st.session_state["checklist_target_tab"] = 1
            st.rerun()

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
