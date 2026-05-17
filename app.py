import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from datetime import datetime, date
import os
 
# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="North",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
}
 
/* ── App background ── */
.stApp { background: #0c0b10; color: #e8e0d0; }
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111015 !important;
    border-right: 1px solid #2a2535;
}
section[data-testid="stSidebar"] * {
    font-size: 15px !important;
    color: #d8d0c2 !important;
}
section[data-testid="stSidebar"] input {
    color: #f0e8d8 !important;
    background: #1a1820 !important;
    border: 1px solid #3a3545 !important;
    font-size: 15px !important;
}
 
/* ── Main container ── */
.main .block-container { padding: 2rem 2.5rem; max-width: 1500px;margin: auto; }
 
/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20,18,25,0.95), rgba(28,24,35,0.95));
    border: 1px solid rgba(201,169,110,0.2);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    border-color: rgba(201,169,110,0.4);
}
[data-testid="stMetricLabel"] p {
    font-size: 13px !important;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #9a9080 !important;
    font-weight: 500 !important;
    margin-bottom: 0.8rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 2.6rem !important;
    font-weight: 600 !important;
    color: #f5ead7 !important;
    letter-spacing: -1px;
}
[data-testid="stMetricDelta"] svg { display: none !important; }
[data-testid="stMetricDelta"] > div {
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    margin-top: 4px;
}
 
/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stDateInput > div > div > input {
    background: #1a1820 !important;
    border: 1px solid #3a3545 !important;
    border-radius: 8px !important;
    color: #f0e8d8 !important;
    font-size: 15px !important;
    padding: 0.55rem 0.9rem !important;
}
.stSelectbox > div > div {
    background: #1a1820 !important;
    border: 1px solid #3a3545 !important;
    color: #f0e8d8 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
}
 
/* ── Buttons ── */
.stButton > button {
    background: rgba(201,169,110,0.12);
    border: 1px solid #c9a96e;
    color: #c9a96e;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
    padding: 0.5rem 1.4rem;
}
.stButton > button:hover { background: rgba(201,169,110,0.25); }
[data-testid="stFormSubmitButton"] > button {
    background: #c9a96e !important;
    color: #0c0b10 !important;
    border-color: #c9a96e !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    width: 100%;
    padding: 0.75rem;
    border-radius: 10px !important;
}
 
/* ── DataFrame / Table ── */
[data-testid="stDataFrame"] { border: 1px solid #2a2535; border-radius: 12px; overflow: hidden; }
[data-testid="stDataFrame"] thead th {
    background: #18151e !important;
    color: #b0a898 !important;
    font-size: 13px !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 12px 16px !important;
}
[data-testid="stDataFrame"] tbody td {
    font-size: 15px !important;
    color: #e8e0d0 !important;
    padding: 10px 16px !important;
}
 
/* ── Misc ── */
hr { border-color: #2a2535 !important; margin: 1.5rem 0 !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0c0b10; }
::-webkit-scrollbar-thumb { background: #4a4550; border-radius: 3px; }
.stNumberInput button { background: #1a1820 !important; border-color: #3a3545 !important; color: #c9a96e !important; font-size: 16px !important; }
 
/* ── Dark card (budget/savings) ── */
.dark-card {
    background: linear-gradient(145deg, rgba(20,18,25,0.95), rgba(28,24,35,0.95));
    border: 1px solid rgba(201,169,110,0.15);
    border-radius: 18px;
    padding: 1.8rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    transition: all 0.25s ease;
}
.dark-card:hover {
    transform: translateY(-3px);
    border-color: rgba(201,169,110,0.3);
}
details summary { font-size: 15px !important; color: #c9a96e !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p { font-size: 15px !important; color: #d8d0c2 !important; }
</style>
""", unsafe_allow_html=True)
 
# ── CONSTANTS ──────────────────────────────────────────────────────────────────
CAT_COLORS = {
    "Food": "#c9a96e", "Transport": "#7eb8c9", "Groceries": "#8fbc8f",
    "Shopping": "#c9a0c9", "Education": "#e8b86d", "Entertainment": "#7ea8c9",
    "Health": "#85c985", "Bills": "#c97e7e", "Salary": "#a8c9a8",
    "Freelance": "#c9c47e", "Bonus": "#e8c87e", "Other": "#888888",
}
 
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#6a6060", size=11),
    margin=dict(t=10, b=10, l=10, r=10),
)
 
# ── HELPERS ────────────────────────────────────────────────────────────────────
def fmt(n):
    return "₹{:,.0f}".format(abs(n))
 
def fmts(n):
    if n >= 100000: return "₹{:.1f}L".format(n / 100000)
    if n >= 1000:   return "₹{:.1f}K".format(n / 1000)
    return "₹{:.0f}".format(n)
 
def bar_html(pct, color="#c9a96e", h=5):
    pct = min(100, max(0, pct))
    return (
        '<div style="height:{}px;background:#1a1820;border-radius:3px;overflow:hidden;margin:5px 0 3px">'
        '<div style="height:100%;width:{:.1f}%;background:{};border-radius:3px"></div>'
        '</div>'
    ).format(h, pct, color)
 
def section_title(txt):
    st.markdown(
        '<p style="font-family:Cormorant Garamond,Georgia,serif;font-size:24px;'
        'font-weight:400;color:#e8e0d0;letter-spacing:0.5px;margin:0 0 0.8rem">{}</p>'.format(txt),
        unsafe_allow_html=True,
    )
 
def page_header(title, sub=""):
    st.markdown(
        '<h2 style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;'
        'font-weight:300;color:#e8e0d0;margin-bottom:4px">{}</h2>'
        '<p style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;'
        'color:#c8beb0;margin-bottom:1.5rem">{}</p>'.format(title, sub),
        unsafe_allow_html=True,
    )
 
# ── DATA ───────────────────────────────────────────────────────────────────────
CSV_PATH = "transactions.csv"
 
@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df["date"] = pd.to_datetime(df["date"], format="mixed")
        return df
    return pd.DataFrame(columns=["date", "description", "amount", "type", "category"])
 
def save_data(df):
    df.to_csv(CSV_PATH, index=False)
    st.cache_data.clear()
 
if "df" not in st.session_state:
    st.session_state.df = load_data()
 
df = st.session_state.df
 
# ── AI MODEL ───────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(n):
    exp = df[df["type"] == "expense"].copy()
    if len(exp) < 3:
        return None, None
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vec.fit_transform(exp["description"])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X, exp["category"])
    return vec, clf
 
vec, clf = train_model(len(df))
 
def predict_cat(desc):
    if clf is None or not desc.strip():
        return "Other"
    return clf.predict(vec.transform([desc]))[0]
 
# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0 1.5rem">'
        '<div style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;'
        'font-weight:300;color:#c9a96e;font-style:italic;letter-spacing:2px">North</div>'
        '<div style="font-size:16px;letter-spacing:2px;text-transform:uppercase;'
        'color:#c0b8a8;margin-top:2px">Finance Tracker</div>'
        '</div>',
        unsafe_allow_html=True,
    )
 
    page = st.radio(
        "Nav",
        ["◈  Dashboard", "≡  Transactions", "◉  Analytics", "+  Add Entry", "◎  Settings"],
        label_visibility="collapsed",
    )
 
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;'
        'color:#c0b8a8;margin-bottom:12px">Financial Targets</p>',
        unsafe_allow_html=True,
    )
 
    target_income  = st.number_input("Monthly Income Target (₹)",  min_value=0, value=50000, step=500)
    monthly_budget = st.number_input("Monthly Expense Budget (₹)", min_value=0, value=15000, step=500)
    savings_goal   = st.number_input("Savings Goal (₹)",           min_value=0, value=10000, step=500)
 
    st.markdown("<hr>", unsafe_allow_html=True)
    name = st.text_input("Your name", value="Shruti")
 
    st.markdown(
        '<div style="padding:1rem 0;font-size:16px;letter-spacing:1px;text-transform:uppercase;'
        'color:#2a2530;text-align:center">Built for next semester ◈</div>',
        unsafe_allow_html=True,
    )
 
# ── DERIVED METRICS ────────────────────────────────────────────────────────────
exp_df     = df[df["type"] == "expense"]
inc_df     = df[df["type"] == "income"]
total_exp  = exp_df["amount"].sum()
total_inc  = inc_df["amount"].sum()
savings    = total_inc - total_exp
budget_pct = min(100.0, (total_exp / monthly_budget * 100)) if monthly_budget else 0.0
sav_pct    = min(100.0, (savings   / savings_goal   * 100)) if savings_goal   else 0.0
health     = min(100, int(savings  / savings_goal   * 100)) if savings_goal   else (100 if savings > 0 else 0)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
 
    hour  = datetime.now().hour
    greet = "Good morning" if hour < 12 else ("Good evening" if hour >= 18 else "Good afternoon")
 
    st.markdown(
        '<div style="margin-bottom:1.8rem">'
        '<div style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;'
        'font-weight:300;color:#e8e0d0;letter-spacing:1px">'
        '{}, <span style="color:#c9a96e;font-style:italic">{}</span></div>'
        '<div style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;'
        'color:#c8beb0;margin-top:6px">{} · Your financial overview</div>'
        '</div>'.format(greet, name, datetime.now().strftime("%A, %d %B %Y")),
        unsafe_allow_html=True,
    )
 
    # — KPI cards —
    k1, k2, k3 = st.columns(3)
    with k1:
        d = total_inc - target_income
        st.metric("TOTAL INCOME", fmt(total_inc),
                  delta="{} {} target".format(fmts(abs(d)), "above" if d >= 0 else "below"),
                  delta_color="normal" if d >= 0 else "inverse")
    with k2:
        r = monthly_budget - total_exp
        st.metric("TOTAL EXPENSES", fmt(total_exp),
                  delta="{} {}".format(fmts(abs(r)), "remaining" if r >= 0 else "over budget"),
                  delta_color="normal" if r >= 0 else "inverse")
    with k3:
        ds = savings - savings_goal
        st.metric("NET SAVINGS", fmt(savings),
                  delta="{} {} goal".format(fmts(abs(ds)), "above" if ds >= 0 else "below"),
                  delta_color="normal" if ds >= 0 else "inverse")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # — Monthly flow + Health score —
    c1, c2 = st.columns([3, 1])
 
    with c1:
        section_title("Monthly Flow")
        if not df.empty:
            m = df.copy()
            m["month"] = m["date"].dt.to_period("M").astype(str)
            mg = m.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0).reset_index()
            if "income"  not in mg.columns: mg["income"]  = 0
            if "expense" not in mg.columns: mg["expense"] = 0
            mg["label"] = pd.to_datetime(mg["month"]).dt.strftime("%b")
 
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mg["label"], y=mg["income"], name="Income",
                fill="tozeroy", line=dict(color="#8fbc8f", width=1.5),
                fillcolor="rgba(143,188,143,0.07)", mode="lines",
            ))
            fig.add_trace(go.Scatter(
                x=mg["label"], y=mg["expense"], name="Expenses",
                fill="tozeroy", line=dict(color="#c9a96e", width=1.5),
                fillcolor="rgba(201,169,110,0.07)", mode="lines",
            ))
            fig.update_layout(
                **PLOTLY_BASE,
                height=420,
                showlegend=True,
                legend=dict(font=dict(size=11, color="#6a6060"), bgcolor="rgba(0,0,0,0)", orientation="h", x=0, y=1.15),
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#1a1820", zeroline=False, tickprefix="₹", tickformat=",.0f"),
            )
            st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        section_title("Health Score")
        sc = "#8fbc8f" if health >= 80 else ("#c9c47e" if health >= 50 else "#c97e7e")
        lb = "Excellent" if health >= 80 else ("Good" if health >= 60 else ("Fair" if health >= 40 else "Needs work"))
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=9, color="#3a3530")),
                bar=dict(color=sc, thickness=0.25),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  40], color="#1a1820"),
                    dict(range=[40, 70], color="#16141a"),
                    dict(range=[70,100], color="#14121a"),
                ],
            ),
            number=dict(font=dict(family="Cormorant Garamond,Georgia,serif", size=36, color="#e8e0d0")),
        ))
        fig2.update_layout(**PLOTLY_BASE, showlegend=False, height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            '<p style="text-align:center;font-size:16px;color:{};margin-top:-14px">{}</p>'.format(sc, lb),
            unsafe_allow_html=True,
        )
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    # — Donut + Category breakdown —
    d1, d2 = st.columns(2)
 
    with d1:
        section_title("Spending by Category")
        if not exp_df.empty:
            cd     = exp_df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            colors = [CAT_COLORS.get(c, "#888") for c in cd["category"]]
            fp = go.Figure(go.Pie(
                labels=cd["category"], values=cd["amount"], hole=0.62,
                marker=dict(colors=colors, line=dict(color="#0d0d0f", width=2)),
                textinfo="none",
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            ))
            fp.update_layout(height=460)
            fp.add_annotation(
                text="<b>{}</b>".format(fmts(total_exp)), x=0.5, y=0.56, showarrow=False,
                font=dict(family="Cormorant Garamond,Georgia,serif", size=25, color="#e8e0d0"),
            )
            fp.add_annotation(
                text="total spend", x=0.5, y=0.42, showarrow=False,
                font=dict(size=25, color="#4a4440"),
            )
            fp.update_layout(**PLOTLY_BASE, showlegend=False, height=460)
            st.plotly_chart(fp, use_container_width=True)
 
    with d2:
        section_title("Category Breakdown")
        if not exp_df.empty:
            cd = exp_df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            for _, row in cd.iterrows():
                pct   = (row["amount"] / total_exp * 100) if total_exp else 0
                color = CAT_COLORS.get(row["category"], "#888")
                b     = bar_html(pct, color=color)
                st.markdown(
                    '<div style="margin-bottom:11px">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px">'
                    '<span style="font-size:16px;color:#e0d8c8">{cat}</span>'
                    '<span style="font-size:16px;color:#e8e0d0">{amt}'
                    '&nbsp;<span style="color:#c8beb0;font-size:16px">{pct:.0f}%</span>'
                    '</span></div>{bar}</div>'.format(
                        cat=row["category"], amt=fmt(row["amount"]), pct=pct, bar=b
                    ),
                    unsafe_allow_html=True,
                )
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    # — Budget & Savings progress —
    section_title("Budget & Savings Progress")
    b1, b2 = st.columns(2)
 
    with b1:
        bc  = "#c97e7e" if budget_pct > 90 else ("#c9c47e" if budget_pct > 70 else "#8fbc8f")
        bar = bar_html(budget_pct, color=bc, h=6)
        st.markdown(
            '<div style="class="dark-card" style="">'
            '<p style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#c8beb0;margin:0 0 8px">Monthly Expense Budget</p>'
            '<p style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;color:#e8e0d0;font-weight:300;margin:0">'
            '{spent} <span style="font-size:24px;color:#c8beb0">/ {limit}</span></p>'
            '{bar}'
            '<div style="display:flex;justify-content:space-between;margin-top:4px">'
            '<span style="font-size:16px;color:{bc}">{pct:.0f}% used</span>'
            '<span style="font-size:16px;color:#c8beb0">{rem} remaining</span>'
            '</div></div>'.format(
                spent=fmt(total_exp), limit=fmt(monthly_budget), bar=bar,
                bc=bc, pct=budget_pct, rem=fmt(monthly_budget - total_exp),
            ),
            unsafe_allow_html=True,
        )
 
    with b2:
        bar2 = bar_html(sav_pct, color="#c9a96e", h=6)
        st.markdown(
            '<div style="class="dark-card" style="">'
            '<p style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#c8beb0;margin:0 0 8px">Savings Goal</p>'
            '<p style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;color:#e8e0d0;font-weight:300;margin:0">'
            '{sav} <span style="font-size:24px;color:#c8beb0">/ {goal}</span></p>'
            '{bar}'
            '<div style="display:flex;justify-content:space-between;margin-top:4px">'
            '<span style="font-size:16px;color:#c9a96e">{pct:.0f}% achieved</span>'
            '<span style="font-size:16px;color:#c8beb0">{rem} to go</span>'
            '</div></div>'.format(
                sav=fmt(savings), goal=fmt(savings_goal), bar=bar2,
                pct=sav_pct, rem=fmt(max(0, savings_goal - savings)),
            ),
            unsafe_allow_html=True,
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # — Top 5 Expenses —
    section_title("Top 5 Expenses")
    if not exp_df.empty:
        top5 = exp_df.nlargest(5, "amount")[["date", "description", "category", "amount"]].copy()
        top5["date"]   = top5["date"].dt.strftime("%d %b")
        top5["amount"] = top5["amount"].apply(fmt)
        st.dataframe(
            top5.rename(columns={"date": "Date", "description": "Description",
                                  "category": "Category", "amount": "Amount"}),
            use_container_width=True, hide_index=True,
        )
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    # — Smart Insights —
    section_title("◈  Smart Insights")
    insights = []
 
    if not exp_df.empty:
        tc  = exp_df.groupby("category")["amount"].sum().idxmax()
        tcv = exp_df.groupby("category")["amount"].sum().max()
        insights.append(("◈", "Highest spending: <b>{}</b> at {} ({:.0f}% of expenses)".format(tc, fmt(tcv), tcv / total_exp * 100)))
 
    if total_exp > monthly_budget:
        insights.append(("△", "Budget <b>exceeded</b> by {}. Consider cutting discretionary spend.".format(fmt(total_exp - monthly_budget))))
    else:
        insights.append(("✓", "Budget on track — <b>{}</b> remaining this month.".format(fmt(monthly_budget - total_exp))))
 
    if savings >= savings_goal:
        insights.append(("★", "Savings goal <b>achieved!</b> You saved {}, target was {}.".format(fmt(savings), fmt(savings_goal))))
    else:
        insights.append(("○", "<b>{}</b> away from your savings goal. Keep going!".format(fmt(savings_goal - savings))))
 
    if not exp_df.empty:
        avg_e = exp_df["amount"].mean()
        max_e = exp_df["amount"].max()
        if max_e > avg_e * 3:
            big = exp_df.loc[exp_df["amount"].idxmax()]
            insights.append(("◉", "Unusual transaction: <b>{}</b> — {} ({:.1f}x your average)".format(
                big["description"], fmt(big["amount"]), big["amount"] / avg_e)))
 
    for icon, msg in insights:
        st.markdown(
            '<div style="display:flex;align-items:flex-start;gap:12px;padding:12px 16px;'
            'background:rgba(201,169,110,0.04);border:0.5px solid #2a2018;'
            'border-radius:8px;margin-bottom:8px">'
            '<span style="color:#c9a96e;font-size:16px;margin-top:2px;flex-shrink:0">{icon}</span>'
            '<span style="font-size:16px;color:#e0d8c8;line-height:1.7">{msg}</span>'
            '</div>'.format(icon=icon, msg=msg),
            unsafe_allow_html=True,
        )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSACTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Transactions" in page:
    page_header("Transactions", "{} entries total".format(len(df)))
 
    f1, f2, f3 = st.columns([3, 1.5, 1.5])
    with f1: search = st.text_input("Search", placeholder="Search description...", label_visibility="collapsed")
    with f2: tf = st.selectbox("Type", ["All", "expense", "income"], label_visibility="collapsed")
    with f3:
        all_cats = ["All"] + sorted(df["category"].unique().tolist())
        cf = st.selectbox("Category", all_cats, label_visibility="collapsed")
 
    fdf = df.copy()
    if search: fdf = fdf[fdf["description"].str.contains(search, case=False, na=False)]
    if tf != "All": fdf = fdf[fdf["type"] == tf]
    if cf != "All": fdf = fdf[fdf["category"] == cf]
    fdf = fdf.sort_values("date", ascending=False)
 
    st.markdown(
        '<p style="font-size:16px;letter-spacing:1px;color:#c0b8a8;margin-bottom:1rem">'
        '{} results</p>'.format(len(fdf)),
        unsafe_allow_html=True,
    )
 
    for _, row in fdf.iterrows():
        sign      = "+" if row["type"] == "income" else "−"
        amt_color = "#8fbc8f" if row["type"] == "income" else "#c9a96e"
        cat_color = CAT_COLORS.get(row["category"], "#888")
        r_int = int(cat_color[1:3], 16)
        g_int = int(cat_color[3:5], 16)
        b_int = int(cat_color[5:7], 16)
        st.markdown(
            '<div style="display:flex;align-items:center;gap:14px;padding:11px 14px;'
            'background:#16141a;border:0.5px solid #1e1c22;border-radius:8px;margin-bottom:6px">'
            '<div style="width:34px;height:34px;border-radius:50%;background:rgba(201,169,110,0.08);'
            'display:flex;align-items:center;justify-content:center;font-size:16px;'
            'color:{ac};flex-shrink:0">{sign}</div>'
            '<div style="flex:1;min-width:0">'
            '<div style="font-size:16px;color:#e8e0d0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{desc}</div>'
            '<div style="font-size:16px;color:#c8beb0;margin-top:2px">{dt}</div>'
            '</div>'
            '<span style="font-size:16px;padding:3px 10px;border-radius:20px;white-space:nowrap;flex-shrink:0;'
            'background:rgba({r},{g},{b},0.1);color:{cc};border:0.5px solid rgba({r},{g},{b},0.25)">'
            '{cat}</span>'
            '<div style="font-family:Cormorant Garamond,Georgia,serif;font-size:24px;'
            'color:{ac};min-width:90px;text-align:right;flex-shrink:0">{amt}</div>'
            '</div>'.format(
                ac=amt_color, sign=sign,
                desc=row["description"],
                dt=pd.to_datetime(row["date"]).strftime("%d %b %Y"),
                r=r_int, g=g_int, b=b_int,
                cc=cat_color, cat=row["category"],
                amt=fmt(row["amount"]),
            ),
            unsafe_allow_html=True,
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙  Edit / Delete rows"):
        edited = st.data_editor(
            df.sort_values("date", ascending=False),
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "amount": st.column_config.NumberColumn("Amount (₹)", format="₹%.2f"),
                "date":   st.column_config.DateColumn("Date"),
            },
        )
        if st.button("💾  Save changes"):
            save_data(edited)
            st.session_state.df = edited
            st.success("Changes saved!")
            st.rerun()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "Analytics" in page:
    page_header("Analytics", "Spending patterns & trends")
 
    avg_daily = total_exp / max(exp_df["date"].nunique(), 1)
    avg_txn   = exp_df["amount"].mean() if not exp_df.empty else 0
    max_e     = exp_df["amount"].max()  if not exp_df.empty else 0
    sav_rate  = (savings / total_inc * 100) if total_inc else 0
 
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("AVG DAILY SPEND",  fmts(avg_daily))
    s2.metric("AVG TRANSACTION",  fmts(avg_txn))
    s3.metric("LARGEST EXPENSE",  fmts(max_e))
    s4.metric("SAVINGS RATE",     "{:.0f}%".format(sav_rate))
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    if not exp_df.empty:
        daily = exp_df.groupby("date")["amount"].sum().reset_index().sort_values("date")
        daily["label"] = pd.to_datetime(daily["date"]).dt.strftime("%d %b")
 
        section_title("Daily Expenditure")
        fb = go.Figure(go.Bar(
            x=daily["label"], y=daily["amount"],
            marker=dict(color="#c9a96e", opacity=0.7, line=dict(width=0)),
        ))
        fb.update_layout(
            **PLOTLY_BASE, showlegend=False, height=420,
            xaxis=dict(showgrid=False, tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="#1a1820", tickprefix="₹", tickformat=",.0f"),
        )
        st.plotly_chart(fb, use_container_width=True)
 
    st.markdown("<hr>", unsafe_allow_html=True)
 
    a1, a2 = st.columns(2)
 
    with a1:
        section_title("Category Breakdown")
        if not exp_df.empty:
            cd = exp_df.groupby("category")["amount"].sum().reset_index().sort_values("amount")
            fh = go.Figure(go.Bar(
                x=cd["amount"], y=cd["category"], orientation="h",
                marker=dict(color=[CAT_COLORS.get(c, "#888") for c in cd["category"]], opacity=0.8, line=dict(width=0)),
                text=[fmts(v) for v in cd["amount"]], textposition="outside",
                textfont=dict(size=11, color="#6a6060"),
            ))
            fh.update_layout(
                **PLOTLY_BASE, showlegend=False, height=420,
                xaxis=dict(showgrid=True, gridcolor="#1a1820", tickprefix="₹", tickformat=",.0f"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fh, use_container_width=True)
 
    with a2:
        section_title("Income Sources")
        if not inc_df.empty:
            sd = inc_df.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
            for _, r in sd.iterrows():
                pct = (r["amount"] / total_inc * 100) if total_inc else 0
                b   = bar_html(pct, color="#8fbc8f")
                st.markdown(
                    '<div style="margin-bottom:14px">'
                    '<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                    '<span style="font-size:16px;color:#e0d8c8">{cat}</span>'
                    '<span style="font-size:16px;color:#8fbc8f">{amt}'
                    '&nbsp;<span style="color:#c0b8a8;font-size:16px">{pct:.0f}%</span>'
                    '</span></div>{bar}</div>'.format(
                        cat=r["category"], amt=fmt(r["amount"]), pct=pct, bar=b
                    ),
                    unsafe_allow_html=True,
                )
 
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center">'
                '<span style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#c8beb0">Savings rate</span>'
                '<span style="font-family:Cormorant Garamond,Georgia,serif;font-size:42px;font-weight:300;color:#c9a96e">'
                '{:.0f}%</span></div>'.format(sav_rate),
                unsafe_allow_html=True,
            )
 
    if not exp_df.empty and len(daily) >= 3:
        st.markdown("<hr>", unsafe_allow_html=True)
        section_title("3-Day Spending Trend")
        trend = daily.copy()
        trend["rolling"] = trend["amount"].rolling(3, min_periods=1).mean()
        fm = go.Figure()
        fm.add_trace(go.Bar(
            x=trend["label"], y=trend["amount"],
            marker=dict(color="#c9a96e", opacity=0.2, line=dict(width=0)), name="Daily",
        ))
        fm.add_trace(go.Scatter(
            x=trend["label"], y=trend["rolling"],
            line=dict(color="#c9a96e", width=2), mode="lines", name="3-day avg",
        ))
        fm.update_layout(
            **PLOTLY_BASE, showlegend=False, height=420,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#1a1820", tickprefix="₹"),
        )
        st.plotly_chart(fm, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ADD ENTRY
# ══════════════════════════════════════════════════════════════════════════════
elif "Add" in page:
    page_header("Record Transaction", "AI auto-detects category from your description")
 
    cf1, cf2 = st.columns([3, 2])
 
    with cf1:
        with st.form("add_form", clear_on_submit=True):
            txn_type = st.selectbox("Type", ["expense", "income"])
            desc     = st.text_input("Description", placeholder="e.g. Zomato order, Uber ride, Salary...")
 
            ac, dc = st.columns(2)
            with ac: amt = st.number_input("Amount (₹)", min_value=0.0, step=10.0)
            with dc: dt  = st.date_input("Date", value=date.today())
 
            if txn_type == "expense":
                cats     = ["Food", "Transport", "Groceries", "Shopping", "Education", "Entertainment", "Health", "Bills", "Other"]
                auto_cat = predict_cat(desc) if desc else "Food"
                idx      = cats.index(auto_cat) if auto_cat in cats else 0
                cat      = st.selectbox("Category (auto-detected)", cats, index=idx)
            else:
                cat = st.selectbox("Source", ["Salary", "Freelance", "Bonus", "Other"])
 
            submitted = st.form_submit_button("Record Transaction", use_container_width=True)
 
            if submitted and desc and amt > 0:
                new_row = pd.DataFrame([{
                    "date": pd.to_datetime(dt),
                    "description": desc,
                    "amount": amt,
                    "type": txn_type,
                    "category": cat,
                }])
                updated = pd.concat([df, new_row], ignore_index=True)
                save_data(updated)
                st.session_state.df = updated
                st.success("✓ {} of {} recorded under {}".format(txn_type.capitalize(), fmt(amt), cat))
                st.rerun()
            elif submitted:
                st.warning("Please fill in description and amount.")
 
    with cf2:
        section_title("Recent Entries")
        recent = df.sort_values("date", ascending=False).head(8)
        for _, row in recent.iterrows():
            ac  = "#8fbc8f" if row["type"] == "income" else "#c9a96e"
            sg  = "+" if row["type"] == "income" else "−"
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:9px 0;border-bottom:0.5px solid #1a1820">'
                '<div>'
                '<div style="font-size:16px;color:#e0d8c8">{desc}</div>'
                '<div style="font-size:16px;color:#c0b8a8;margin-top:1px">{dt} · {cat}</div>'
                '</div>'
                '<span style="font-family:Cormorant Garamond,Georgia,serif;font-size:24px;color:{ac}">'
                '{sg}{amt}</span>'
                '</div>'.format(
                    desc=row["description"],
                    dt=pd.to_datetime(row["date"]).strftime("%d %b"),
                    cat=row["category"],
                    ac=ac, sg=sg, amt=fmt(row["amount"]),
                ),
                unsafe_allow_html=True,
            )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif "Settings" in page:
    page_header("Settings", "Manage your data & preferences")
 
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("TOTAL ENTRIES",   len(df))
    s2.metric("EXPENSE ENTRIES", len(exp_df))
    s3.metric("INCOME ENTRIES",  len(inc_df))
    s4.metric("CATEGORIES",      df["category"].nunique() if not df.empty else 0)
 
    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("Data Management")
 
    sa, sb = st.columns(2)
 
    with sa:
        st.markdown(
            '<div style="class="dark-card" style=";margin-bottom:12px">'
            '<p style="font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#c8beb0;margin:0 0 8px">Export Data</p>'
            '<p style="font-size:16px;color:#c0b8a8;margin:0 0 12px">Download all transactions as CSV for backup or analysis.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        if not df.empty:
            st.download_button(
                "⬇  Download transactions.csv",
                df.to_csv(index=False).encode("utf-8"),
                "transactions.csv", "text/csv",
                use_container_width=True,
            )
 
    with sb:
        section_title("Category Distribution")
        if not df.empty:
            for cat in sorted(df["category"].unique()):
                count = len(df[df["category"] == cat])
                color = CAT_COLORS.get(cat, "#888")
                st.markdown(
                    '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:0.5px solid #1a1820">'
                    '<span style="font-size:16px;color:#c0b8a8;display:flex;align-items:center;gap:7px">'
                    '<span style="width:6px;height:6px;border-radius:50%;background:{color};display:inline-block"></span>'
                    '{cat}</span>'
                    '<span style="font-size:16px;color:#c8beb0">{count} entries</span>'
                    '</div>'.format(color=color, cat=cat, count=count),
                    unsafe_allow_html=True,
                )