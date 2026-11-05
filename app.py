"""
แดชบอร์ดแนวโน้มการใช้โทรศัพท์มือถือทั่วโลก (ไทยเป็นประเทศตัวอย่าง)
ข้อมูล: data/phone_*.parquet  ·  แหล่ง: World Bank WDI, ITU, StatCounter Global Stats
รันบนเครื่อง:  streamlit run app.py
"""
import hmac, io, os, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mobile Trends · ไทยและโลก", page_icon=":material/smartphone:", layout="wide")

# ---------------------------------------------------------------- ค่าคงที่
INK, MUTED, LINE, BG, ACC, TH_C = "#14162B", "#5E6478", "#E3E6EE", "#F6F7FB", "#0E7C86", "#E8590C"
GOOD, BAD, GRID = "#1E7D4F", "#B3412E", "#ECEEF4"
FONT = "IBM Plex Sans Thai, sans-serif"
BRAND_COLOR = {"Apple": "#1C1C1E", "Samsung": "#1428A0", "Xiaomi": "#FF6900", "Huawei": "#C7000B", "Oppo": "#1BA784", "Vivo": "#415FFF",
               "Realme": "#D4A100", "Honor": "#00A0E9", "Tecno": "#0063B1", "Infinix": "#6CB33E", "Google": "#34A853", "Motorola": "#5C7C8A",
               "Nokia": "#124191", "BlackBerry": "#6B6B6B", "LG": "#A50034", "Sony": "#8E8E93", "HTC": "#69A33A", "Unknown": "#C9CED6", "Other": "#9AA2AD"}
OS_COLOR = {"Android": "#3DDC84", "iOS": "#1C1C1E", "Other": "#9AA2AD"}
PF_COLOR = {"Mobile": ACC, "Desktop": "#9AA2AD", "Tablet": "#D4A100"}
INC = ["Low income", "Lower middle income", "Upper middle income", "High income"]
INC_TH = {"Low income": "รายได้ต่ำ", "Lower middle income": "ปานกลางล่าง", "Upper middle income": "ปานกลางบน", "High income": "รายได้สูง"}
INC_COLOR = {"รายได้ต่ำ": "#B3412E", "ปานกลางล่าง": "#E8590C", "ปานกลางบน": "#0E7C86", "รายได้สูง": "#14162B"}
METRIC = {  # คอลัมน์: (ชื่อไทย, หน่วย, ยิ่งมากยิ่งดี)
    "mobile_subs_per100": ("ผู้ใช้บริการมือถือต่อประชากร 100 คน", "ต่อ 100 คน", True),
    "internet_pct": ("ประชากรที่ใช้อินเทอร์เน็ต", "%", True),
    "cov_4g": ("ประชากรในพื้นที่สัญญาณ 4G", "%", True),
    "cov_5g": ("ประชากรในพื้นที่สัญญาณ 5G", "%", True),
    "mobile_web_share": ("การเข้าเว็บด้วยมือถือ", "%", None),
    "ios_share": ("ส่วนแบ่ง iOS ในมือถือ", "%", None),
    "price_data_gni": ("ค่าอินเทอร์เน็ตมือถือเทียบรายได้", "% ของรายได้ต่อเดือน", False),
    "price_voice_gni": ("ค่าโทรมือถือเทียบรายได้", "% ของรายได้ต่อเดือน", False),
    "fixed_tel_per100": ("โทรศัพท์บ้านต่อประชากร 100 คน", "ต่อ 100 คน", None),
}
SRC = "ที่มา: World Bank WDI · ITU · StatCounter Global Stats (วัดจากการเข้าเว็บ ไม่ใช่ยอดขาย) · ตัวชี้วัดโทรคมนาคมถึงปี 2024 · แบรนด์/ระบบถึงปี 2025"
CFG = {"displaylogo": False, "displayModeBar": "hover", "toImageButtonOptions": {"format": "png", "scale": 2}}

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label {{ font-family: {FONT} !important; }}
.block-container {{ padding-top: 4.2rem; max-width: 1340px; }}
h1 {{ font-weight: 700; color: {INK}; letter-spacing: -0.01em; }}
.top {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 1.5rem; align-items: end; border-bottom: 1px solid {LINE}; padding-bottom: 1rem; margin-bottom: 1.1rem; }}
.top .eyebrow {{ color: {ACC}; font-weight: 600; font-size: .85rem; letter-spacing: .06em; text-transform: uppercase; }}
.top h1 {{ font-size: 2.1rem; margin: .2rem 0 .3rem; line-height: 1.2; }}
.top p {{ color: {MUTED}; margin: 0; }}
.big {{ text-align: right; }} .big .n {{ font-size: 3rem; font-weight: 700; color: {ACC}; line-height: 1; }} .big .l {{ color: {MUTED}; font-size: .9rem; }}
.kpi {{ background: #fff; border-radius: 14px; padding: 1rem 1.1rem; border: 1px solid {LINE}; height: 100%; }}
.kpi .l {{ color: {MUTED}; font-size: .86rem; }} .kpi .v {{ font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.25; }}
.kpi .v small {{ font-size: .9rem; color: {MUTED}; font-weight: 500; margin-left: .2rem; }}
.kpi .d {{ font-size: .84rem; color: {MUTED}; margin-top: .25rem; }} .kpi .d b {{ color: {TH_C}; }}
[class*="st-key-card"] {{ background: #fff; border-radius: 14px; border-color: {LINE} !important; }}
.ct {{ font-weight: 600; font-size: 1.08rem; color: {INK}; line-height: 1.4; }} .cs {{ color: {MUTED}; font-size: .85rem; }}
.note {{ background: {BG}; border-radius: 10px; padding: .65rem .9rem; margin-top: .3rem; font-size: .9rem; line-height: 1.6; color: {INK}; }}
.note b {{ color: {ACC}; }}
.sec {{ font-size: 1.25rem; font-weight: 700; color: {INK}; margin: 1.5rem 0 .6rem; display: flex; gap: .6rem; align-items: center; }}
.sec::before {{ content: ""; width: 10px; height: 10px; border-radius: 3px; background: {ACC}; }}
.src {{ color: {MUTED}; font-size: .78rem; margin-top: 1.2rem; }}
.ans {{ background: #fff; border: 1px solid {LINE}; border-radius: 14px; padding: 1rem 1.1rem; height: 100%; }}
.ans .q {{ color: {MUTED}; font-size: .84rem; min-height: 2.4rem; }} .ans .a {{ color: {ACC}; font-weight: 700; font-size: 1.45rem; margin: .3rem 0; }}
.ans .t {{ font-size: .88rem; line-height: 1.5; }}
.tbl {{ width: 100%; border-collapse: collapse; font-size: .9rem; }} .tbl th {{ text-align: left; color: {MUTED}; border-bottom: 2px solid {INK}; padding: .5rem; }}
.tbl td {{ border-bottom: 1px solid {LINE}; padding: .55rem .5rem; vertical-align: top; }} .tbl td:first-child {{ font-weight: 600; }}
@media (max-width: 700px) {{ .top {{ grid-template-columns: 1fr; }} .big {{ text-align: left; }} }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- ข้อมูล
def path(name):
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ["data", ".", os.path.join("..", "final")]:
        for ext in [".parquet", ".csv"]:
            p = os.path.join(here, f, name + ext)
            if os.path.exists(p):
                return p
    return None


def read(name):
    p = path(name)
    if p is None:
        raise FileNotFoundError(name)
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load():
    t0 = time.perf_counter()
    C, B, O, P = read("phone_country_year"), read("phone_brand_year"), read("phone_os_year"), read("phone_platform_year")
    names = C.drop_duplicates("iso3").set_index("iso3")[["country", "region_th", "income_group", "is_asean"]]
    for d in (B, O, P):
        d[["country", "region_th", "income_group", "is_asean"]] = names.reindex(d["iso3"]).values
    C["income_th"] = C["income_group"].map(INC_TH)
    ms = (time.perf_counter() - t0) * 1000
    return C, B, O, P, {"rows": sum(map(len, (C, B, O, P))), "load_ms": ms}


@st.cache_data(show_spinner=False)
def side(name):
    p = path(name)
    return pd.read_csv(p) if p else None


# ---------------------------------------------------------------- ตัวช่วย
_n = [0]


def card():
    _n[0] += 1
    return st.container(border=True, key=f"card_{_n[0]}")


def title(t, s=""):
    st.markdown(f'<div class="ct">{t}</div>' + (f'<div class="cs">{s}</div>' if s else ""), unsafe_allow_html=True)


def sec(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def note(f, m, a=None):
    st.markdown(f'<div class="note"><b>ข้อค้นพบ</b> {f}<br><b>ความหมาย</b> {m}' + (f'<br><b>ข้อเสนอแนะ</b> {a}' if a else "") + "</div>",
                unsafe_allow_html=True)


def kpi(col, label, value, unit="", sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}<small>{unit}</small></div><div class="d">{sub}</div></div>',
                 unsafe_allow_html=True)


def src(x=""):
    st.markdown(f'<div class="src">{SRC}{" · " + x if x else ""}</div>', unsafe_allow_html=True)


def empty(msg, keys=None):
    st.info(msg)
    if keys:
        st.button("ล้างตัวกรอง", key=f"rst_{keys[0]}", on_click=lambda: [st.session_state.pop(k, None) for k in keys])


def show(fig):
    st.plotly_chart(fig, width="stretch", config=CFG)


def sty(fig, h=330, yt=None, xt=None, legend=True):
    fig.update_layout(height=h, font=dict(family=FONT, size=13, color=INK), margin=dict(l=4, r=8, t=34 if legend else 8, b=4),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, title=None, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor=LINE), modebar=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, title=xt, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, title=yt, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def f1(v, d=1, u=""):
    return "ไม่มีข้อมูล" if v is None or pd.isna(v) else f"{v:,.{d}f}{u}"


def table(rows, head):
    st.markdown('<table class="tbl"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table>", unsafe_allow_html=True)


def country_picker(key, label="ประเทศ (พิมพ์ชื่อภาษาอังกฤษเพื่อค้นหา)"):
    C = DATA[0]
    opts = sorted(C.loc[(C["iso3"] != "WLD") & ~C["no_telecom_data"], "country"].unique())
    return st.selectbox(label, opts, index=opts.index("Thailand"), key=key)


# ---------------------------------------------------------------- เข้าสู่ระบบ
def users():
    d = {"judge": "phone2025"}
    try:
        u = {str(k).strip().lower(): str(v).strip().strip('"\'“”') for k, v in dict(st.secrets["users"]).items()}
        return {**d, **u} if u else d
    except Exception:
        return d


def login():
    st.markdown('<div class="top"><div><div class="eyebrow">Mobile Trends Dashboard</div><h1>แนวโน้มการใช้โทรศัพท์มือถือทั่วโลก</h1>'
                '<p>การเข้าถึง แบรนด์ ระบบปฏิบัติการ และราคาค่าบริการ จาก 217 ประเทศ โดยมีประเทศไทยเป็นตัวอย่าง</p></div>'
                '<div class="big"><div class="n">217</div><div class="l">ประเทศ · ปี 2000–2025</div></div></div>', unsafe_allow_html=True)
    _, m, _ = st.columns([1, 1.1, 1])
    with m, card():
        title("เข้าสู่ระบบ", "กรอกชื่อผู้ใช้และรหัสผ่านที่ได้รับ")
        with st.form("f", border=False):
            u = st.text_input("ชื่อผู้ใช้"); p = st.text_input("รหัสผ่าน", type="password")
            ok = st.form_submit_button("เข้าสู่ระบบ", type="primary", width="stretch")
        if ok:
            us, u = users(), u.strip().lower()
            if u in us and hmac.compare_digest(p.strip(), us[u]):
                st.session_state["user"] = u; st.rerun()
            else:
                st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง กรุณาตรวจสอบภาษาของแป้นพิมพ์แล้วลองอีกครั้ง")
    src()


# ---------------------------------------------------------------- หน้า 1 ภาพรวม
def p_home():
    C, B, O, P, _ = DATA
    a, b = st.columns([1, 2])
    with a:
        name = country_picker("hm_c")
    yr = b.select_slider("ปี (ตัวชี้วัดโทรคมนาคมมีถึงปี 2024)", options=list(range(2010, 2025)), value=2024, key="hm_y")
    c = C[C["country"] == name].set_index("year"); w = C[C["iso3"] == "WLD"].set_index("year")
    iso = c["iso3"].iloc[0]; lbl = "ไทย" if name == "Thailand" else name
    st.markdown(f'<div class="top"><div><div class="eyebrow">ภาพรวม · {lbl} เทียบโลก · ปี {yr}</div><h1>{lbl}ใช้มือถือ {f1(c.loc[yr, "mobile_subs_per100"], 0)} เลขหมายต่อ 100 คน</h1>'
                f'<p>ค่าเฉลี่ยโลก {f1(w.loc[yr, "mobile_subs_per100"], 0)} · เกิน 100 ได้เพราะคนหนึ่งมีหลายซิม</p></div>'
                f'<div class="big"><div class="n">{f1(c.loc[yr, "internet_pct"], 0, "%")}</div><div class="l">ของประชากร{lbl}ใช้อินเทอร์เน็ต (โลก {f1(w.loc[yr, "internet_pct"], 0, "%")})</div></div></div>',
                unsafe_allow_html=True)
    k = st.columns(4)
    pf = P[(P["iso3"] == iso) & (P["year"] == min(yr + 1, 2025)) & (P["platform"] == "Mobile")]["share_pct"]
    kpi(k[0], "เข้าเว็บด้วยมือถือ", f1(pf.iloc[0] if len(pf) else np.nan, 0), "%", f"ปี {min(yr + 1, 2025)} · มากกว่าคอมพิวเตอร์" if len(pf) and pf.iloc[0] > 50 else "")
    br = B[(B["iso3"] == iso) & (B["year"] == min(yr + 1, 2025)) & (B["brand"] != "Unknown")].sort_values("share_pct")
    kpi(k[1], "แบรนด์อันดับ 1", br["brand"].iloc[-1] if len(br) else "-", "", f"ส่วนแบ่ง {f1(br['share_pct'].iloc[-1], 0, '%')}" if len(br) else "")
    kpi(k[2], "ส่วนแบ่ง iOS", f1(c.loc[yr, "ios_share"], 0), "%", f"โลก <b>{f1(w.loc[yr, 'ios_share'], 0, '%')}</b>")
    kpi(k[3], "ค่าเน็ตมือถือเทียบรายได้", f1(c.loc[yr, "price_data_gni"], 1), "%", "เป้าหมายสากล ≤ 2% ของรายได้ต่อเดือน")

    sec("การเข้าถึงเพิ่มขึ้นอย่างไร")
    l, r = st.columns(2)
    with l, card():
        title("ผู้ใช้อินเทอร์เน็ต (%)", f"{lbl} เทียบโลก 2000–2024")
        d = pd.concat([c["internet_pct"].rename(lbl), w["internet_pct"].rename("โลก")], axis=1).loc[2000:2024].reset_index().melt("year", var_name="กลุ่ม", value_name="%")
        fig = px.line(d, x="year", y="%", color="กลุ่ม", color_discrete_map={lbl: TH_C, "โลก": "#9AA2AD"}, labels={"year": "ปี"}); fig.update_traces(line_width=3)
        show(sty(fig, 300, "%")); i0, i1 = c.loc[2010, "internet_pct"], c.loc[2024, "internet_pct"]
        note(f"{lbl} {f1(i0, 0)}% (2010) → {f1(i1, 0)}% (2024) · โลก {f1(w.loc[2024, 'internet_pct'], 0)}%",
             "มือถือราคาถูกและเครือข่าย 4G ทำให้คนเข้าถึงอินเทอร์เน็ตได้เร็วกว่ายุคคอมพิวเตอร์")
    with r, card():
        title("มือถือแซงคอมพิวเตอร์", f"สัดส่วนการเข้าเว็บของ{lbl} (%)")
        d = P[P["iso3"] == iso].pivot(index="year", columns="platform", values="share_pct")
        fig = go.Figure()
        for pfm in ["Mobile", "Desktop", "Tablet"]:
            if pfm in d:
                fig.add_scatter(x=d.index, y=d[pfm], name={"Mobile": "มือถือ", "Desktop": "คอมพิวเตอร์", "Tablet": "แท็บเล็ต"}[pfm], line=dict(color=PF_COLOR[pfm], width=3))
        fig = sty(fig, 300, "%"); fig.update_yaxes(range=[0, 100]); show(fig)
        cross = d[d["Mobile"] > d["Desktop"]].index.min() if "Mobile" in d else None
        note(f"มือถือแซงคอมพิวเตอร์ครั้งแรกปี {cross}" if pd.notna(cross) else "มือถือยังไม่แซงคอมพิวเตอร์",
             "บริการออนไลน์ต้องออกแบบสำหรับจอมือถือเป็นหลัก", "หน่วยงานและธุรกิจควรทดสอบเว็บไซต์บนมือถือก่อนเสมอ")

    sec("แบรนด์ที่ใช้")
    with card():
        y2 = min(yr + 1, 2025)
        title(f"ส่วนแบ่งแบรนด์ ปี {y2}: {lbl} เทียบโลก", "วัดจากการเข้าเว็บ (StatCounter)")
        d = B[B["iso3"].isin([iso, "WLD"]) & (B["year"] == y2)].copy(); d["กลุ่ม"] = np.where(d["iso3"] == "WLD", "โลก", lbl)
        order = d[d["iso3"] == iso].sort_values("share_pct", ascending=False)["brand"].tolist()
        fig = px.bar(d, y="กลุ่ม", x="share_pct", color="brand", orientation="h", color_discrete_map=BRAND_COLOR, category_orders={"brand": order},
                     labels={"share_pct": "%", "brand": "แบรนด์"})
        fig.update_traces(hovertemplate="%{fullData.name}: %{x:.1f}%<extra></extra>")
        fig = sty(fig, 230, None, "%"); fig.update_xaxes(range=[0, 100]); show(fig)
        cn = d[(d["iso3"] == iso)].merge(B[["brand"]].drop_duplicates(), on="brand")
        china = B[(B["iso3"] == iso) & (B["year"] == y2) & (B["brand"].isin(["Xiaomi", "Huawei", "Oppo", "Vivo", "Realme", "Honor", "Tecno", "Infinix"]))]["share_pct"].sum()
        note(f"{order[0]} นำ{lbl} · แบรนด์จีนรวมกัน {china:.0f}%", "ตลาดแบ่งเป็นแบรนด์พรีเมียม (Apple, Samsung) กับแบรนด์จีนราคาคุ้มค่า",
             "ดูการเปลี่ยนแปลงตามเวลาในหน้า “แบรนด์และระบบปฏิบัติการ”")
    src()


# ---------------------------------------------------------------- หน้า 2 แบรนด์
def p_brand():
    C, B, O, P, _ = DATA
    st.title("แบรนด์และระบบปฏิบัติการ")
    a, b = st.columns([1, 2])
    with a:
        name = st.selectbox("ประเทศ", ["โลก (World)"] + sorted(B.loc[B["iso3"] != "WLD", "country"].dropna().unique()), index=0, key="br_c")
    yr = b.slider("ช่วงปี", 2010, 2025, (2012, 2025), key="br_y")
    iso = "WLD" if name.startswith("โลก") else B.loc[B["country"] == name, "iso3"].iloc[0]
    d = B[(B["iso3"] == iso) & B["year"].between(*yr)]
    if d.empty:
        empty("ไม่พบข้อมูลในช่วงที่เลือก กรุณาขยายช่วงปี", ["br_y"]); return
    l, r = st.columns([1.5, 1])
    with l, card():
        top = d[d["year"] == yr[1]].sort_values("share_pct", ascending=False)
        title(f"{top['brand'].iloc[0]} นำตลาดปี {yr[1]} ({top['share_pct'].iloc[0]:.0f}%)", "ส่วนแบ่งแบรนด์ (%) รวม 100% ทุกปี")
        order = d.groupby("brand")["share_pct"].mean().sort_values(ascending=False).index.tolist()
        fig = px.area(d, x="year", y="share_pct", color="brand", color_discrete_map=BRAND_COLOR, category_orders={"brand": order}, labels={"share_pct": "%", "year": "ปี", "brand": "แบรนด์"})
        fig.for_each_trace(lambda t: t.update(fillcolor=BRAND_COLOR.get(t.name), line=dict(width=.5, color="white")))
        fig = sty(fig, 400, "%"); fig.update_yaxes(range=[0, 100]); show(fig)
    with r, card():
        g = d.assign(o=np.where(d["brand_origin"] == "จีน", "แบรนด์จีน", np.where(d["brand"].isin(["Apple", "Samsung"]), "Apple + Samsung", "อื่น ๆ")))
        g = g.groupby(["year", "o"], as_index=False)["share_pct"].sum()
        cn = g[g["o"] == "แบรนด์จีน"].set_index("year")["share_pct"]
        title(f"แบรนด์จีนเพิ่มจาก {cn.iloc[0]:.0f}% เป็น {cn.iloc[-1]:.0f}%", f"ปี {yr[0]}–{yr[1]}")
        fig = px.line(g, x="year", y="share_pct", color="o", color_discrete_map={"แบรนด์จีน": "#FF6900", "Apple + Samsung": INK, "อื่น ๆ": "#9AA2AD"},
                      labels={"share_pct": "%", "year": "ปี", "o": "กลุ่ม"}); fig.update_traces(line_width=3)
        fig = sty(fig, 400, "%"); fig.update_yaxes(range=[0, 100]); show(fig)
    note(f"{'ทั่วโลก' if iso == 'WLD' else name}: แบรนด์จีน (Xiaomi, Oppo, Vivo, Huawei ฯลฯ) {cn.iloc[0]:.0f}% → {cn.iloc[-1]:.0f}%",
         "แบรนด์จีนแข่งด้วยสเปกสูงในราคาต่ำ แทนที่ Nokia และ BlackBerry ที่หายไปจากตลาด", "ติดตามการแข่งขันด้านราคา ซึ่งเป็นปัจจัยให้คนรายได้ปานกลางเข้าถึงสมาร์ตโฟน")
    l, r = st.columns(2)
    with l, card():
        o = O[(O["iso3"] == iso) & O["year"].between(*yr)]
        title("Android vs iOS", "ส่วนแบ่งระบบปฏิบัติการมือถือ (%)")
        fig = px.line(o, x="year", y="share_pct", color="os", color_discrete_map=OS_COLOR, labels={"share_pct": "%", "year": "ปี", "os": "ระบบ"})
        fig.update_traces(line_width=3); fig = sty(fig, 320, "%"); fig.update_yaxes(range=[0, 100]); show(fig)
    with r, card():
        y2 = 2025
        lead = B[(B["year"] == y2) & (B["iso3"] != "WLD") & (B["brand"] != "Unknown")].sort_values("share_pct").groupby("iso3").tail(1)
        title(f"แบรนด์อันดับ 1 ของแต่ละประเทศ ปี {y2}", "แผนที่โลก")
        fig = px.choropleth(lead, locations="iso3", color="brand", hover_name="country", color_discrete_map=BRAND_COLOR, labels={"brand": "แบรนด์"})
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)")
        show(sty(fig, 320))
        vc = lead["brand"].value_counts()
        st.caption(f"Samsung นำ {vc.get('Samsung', 0)} ประเทศ · Apple {vc.get('Apple', 0)} ประเทศ · แบรนด์จีน {lead['brand'].isin(['Xiaomi', 'Oppo', 'Vivo', 'Huawei', 'Tecno', 'Infinix', 'Realme', 'Honor']).sum()} ประเทศ")
    src()


# ---------------------------------------------------------------- หน้า 3 การเข้าถึงและราคา
def p_access():
    C, B, O, P, _ = DATA
    st.title("การเข้าถึงและราคา")
    CC = C[(C["iso3"] != "WLD") & ~C["no_telecom_data"] & C["income_group"].isin(INC)]
    yr = st.select_slider("ปี", options=list(range(2018, 2025)), value=2024, key="ac_y")
    y = CC[CC["year"] == yr]
    sec("ช่องว่างตามกลุ่มรายได้")
    g = CC[CC["year"].between(2010, 2024)].groupby(["year", "income_th"], as_index=False)[["internet_pct", "cov_4g"]].median()
    l, r = st.columns(2)
    for col, v, t in [(l, "internet_pct", "ใช้อินเทอร์เน็ต (%)"), (r, "cov_4g", "อยู่ในพื้นที่ 4G (%)")]:
        with col, card():
            title(f"มัธยฐาน{t} แยกกลุ่มรายได้", "2010–2024")
            fig = px.line(g, x="year", y=v, color="income_th", color_discrete_map=INC_COLOR, category_orders={"income_th": list(INC_COLOR)},
                          labels={"year": "ปี", v: "%", "income_th": "กลุ่มรายได้"}); fig.update_traces(line_width=3)
            fig = sty(fig, 300, "%"); fig.update_yaxes(range=[0, 105]); show(fig)
    med = y.groupby("income_th")[["internet_pct", "price_data_gni", "ios_share"]].median()
    note(f"ปี {yr} ประเทศรายได้ต่ำใช้อินเทอร์เน็ต {f1(med.loc['รายได้ต่ำ', 'internet_pct'], 0)}% เทียบรายได้สูง {f1(med.loc['รายได้สูง', 'internet_pct'], 0)}%",
         "แม้มีสัญญาณแล้ว แต่คนจำนวนมากในประเทศยากจนยังไม่ใช้อินเทอร์เน็ต", "ลดค่าบริการและราคาเครื่องควบคู่การขยายเครือข่าย")
    sec("ราคาและรายได้ กับการเลือกใช้")
    l, r = st.columns(2)
    for col, (xc, yc, xt, yt) in [(l, ("price_data_gni", "internet_pct", "ค่าเน็ตเทียบรายได้ (% สเกล log)", "ใช้อินเทอร์เน็ต (%)")),
                                  (r, ("gdp_pc_ppp", "ios_share", "GDP ต่อหัว (ดอลลาร์ PPP สเกล log)", "ส่วนแบ่ง iOS (%)"))]:
        with col, card():
            d = y.dropna(subset=[xc, yc])
            if len(d) < 10:
                empty("ปีนี้มีข้อมูลน้อยเกินไป กรุณาเลือกปีอื่น", ["ac_y"]); continue
            rr = d[[xc, yc]].corr(method="spearman").iloc[0, 1]
            title(("ค่าเน็ตยิ่งแพง คนยิ่งใช้อินเทอร์เน็ตน้อย" if xc.startswith("price") else "ประเทศยิ่งรวย ยิ่งใช้ iPhone มาก"), f"{len(d)} ประเทศ ปี {yr} · Spearman r = {rr:.2f}")
            fig = px.scatter(d, x=xc, y=yc, color="income_th", color_discrete_map=INC_COLOR, hover_name="country", log_x=True,
                             category_orders={"income_th": list(INC_COLOR)}, labels={xc: xt, yc: yt, "income_th": "กลุ่มรายได้"})
            th = d[d["iso3"] == "THA"]
            fig.add_scatter(x=th[xc], y=th[yc], mode="markers+text", text=["ไทย"] * len(th), textposition="top center", name="ไทย",
                            marker=dict(size=14, color=TH_C, line=dict(color="white", width=2)))
            show(sty(fig, 340, yt, xt))
    note("ราคาเทียบรายได้สัมพันธ์ทิศทางตรงข้ามกับการใช้อินเทอร์เน็ต ส่วนรายได้สัมพันธ์ทิศทางเดียวกับการใช้ iPhone",
         "ราคาและกำลังซื้อเป็นปัจจัยสำคัญของการเข้าถึงและการเลือกแบรนด์ (เป็นความสัมพันธ์ ไม่ใช่การพิสูจน์สาเหตุ)",
         "ใช้เป้าหมายค่าเน็ต ≤ 2% ของรายได้เป็นตัวชี้วัดนโยบาย")
    with card():
        name = country_picker("ac_c", "ดูความครอบคลุมเครือข่ายของประเทศ")
        c = C[C["country"] == name].set_index("year")[["cov_2g", "cov_3g", "cov_4g", "cov_5g"]].loc[2010:2024]
        title(f"ความครอบคลุมเครือข่ายของ{name}", "% ประชากร")
        fig = go.Figure()
        for g_, col in zip(["cov_2g", "cov_3g", "cov_4g", "cov_5g"], ["#C9CED6", "#9AA2AD", ACC, TH_C]):
            fig.add_scatter(x=c.index, y=c[g_], name=g_.replace("cov_", "").upper(), line=dict(color=col, width=3))
        fig = sty(fig, 280, "%"); fig.update_yaxes(range=[0, 105]); show(fig)
    src("ราคา: ITU ICT Price Baskets (นิยามแพ็กเกจเปลี่ยนปี 2017, 2021, 2025)")


# ---------------------------------------------------------------- หน้า 4 เปรียบเทียบ
def p_compare():
    C, B, O, P, _ = DATA
    st.title("เปรียบเทียบประเทศ")
    CC = C[(C["iso3"] != "WLD") & ~C["no_telecom_data"]]
    groups = {"อาเซียน": sorted(CC.loc[CC["is_asean"], "country"].unique())}
    for r in sorted(CC["region_th"].dropna().unique()):
        groups[f"ภูมิภาค: {r}"] = sorted(CC.loc[CC["region_th"] == r, "country"].unique())
    for g in INC:
        groups[f"กลุ่ม{INC_TH[g]}"] = sorted(CC.loc[CC["income_group"] == g, "country"].unique())

    def ap():
        if st.session_state.get("cp_g") in groups:
            st.session_state["cp_cs"] = groups[st.session_state["cp_g"]]
    st.session_state.setdefault("cp_g", "อาเซียน"); st.session_state.setdefault("cp_cs", groups["อาเซียน"])
    with card():
        a, b, c = st.columns([1.6, 1, .8])
        m = a.selectbox("ตัวชี้วัด", list(METRIC), format_func=lambda k: f"{METRIC[k][0]} ({METRIC[k][1]})", key="cp_m")
        yr = b.select_slider("ปี", options=list(range(2010, 2025)), value=2024, key="cp_y")
        order = c.radio("เรียง", ["มากไปน้อย", "น้อยไปมาก"], key="cp_o", horizontal=True)
        a, b = st.columns([1, 2.3])
        a.selectbox("ชุดประเทศ", list(groups) + ["กำหนดเอง"], key="cp_g", on_change=ap)
        cs = b.multiselect("ประเทศ", sorted(CC["country"].unique()), key="cp_cs", on_change=lambda: st.session_state.update(cp_g="กำหนดเอง"))
    name, unit, better = METRIC[m]
    if not cs:
        empty("ยังไม่ได้เลือกประเทศ กรุณาเลือกชุดประเทศ หรือเพิ่มอย่างน้อย 1 ประเทศ"); return
    d = CC[CC["country"].isin(cs) & (CC["year"] == yr)].dropna(subset=[m])
    if d.empty:
        empty(f"ไม่มีข้อมูล{name}ของประเทศที่เลือกในปี {yr} (ราคาค่าเน็ตเริ่มปี 2018, 5G เริ่มราวปี 2019)", ["cp_y"]); return
    with card():
        d = d.sort_values(m, ascending=(order == "น้อยไปมาก"))
        title(f"{name} ปี {yr}", f"หน่วย: {unit} · {len(d)} ประเทศ · ไทยเป็นสีส้ม")
        fig = go.Figure(go.Bar(x=d["country"], y=d[m], marker_color=[TH_C if x == "Thailand" else "#B7BDCB" for x in d["country"]],
                               text=None if len(d) > 15 else d[m].map(lambda v: f"{v:,.1f}"), textposition="outside", cliponaxis=False,
                               hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
        fig = sty(fig, 420 if len(d) <= 15 else 480, unit, legend=False)
        fig.update_xaxes(categoryorder="array", categoryarray=d["country"].tolist(), tickangle=-60 if len(d) > 15 else 0); show(fig)
        md_ = d[m].median(); th = d[d["iso3"] == "THA"]
        note(f"สูงสุด {d.loc[d[m].idxmax(), 'country']} ({d[m].max():,.1f}) · ต่ำสุด {d.loc[d[m].idxmin(), 'country']} ({d[m].min():,.1f}) · มัธยฐาน {md_:,.1f}"
             + (f" · ไทย {th[m].iloc[0]:,.1f}" if len(th) else ""),
             ("ค่าที่สูงกว่าดีกว่า" if better else "ค่าที่ต่ำกว่าดีกว่า" if better is False else "ค่านี้สะท้อนพฤติกรรม ไม่ได้บอกดีหรือแย่") + " · ความต่างสะท้อนรายได้ นโยบาย และโครงสร้างพื้นฐาน")
    with card():
        t = CC[CC["country"].isin(cs) & CC["year"].between(2010, 2024)].dropna(subset=[m])
        if t["country"].nunique() > 12:
            keep = d.nlargest(10, m)["country"].tolist() + (["Thailand"] if "Thailand" in cs else []); t = t[t["country"].isin(keep)]
        title(f"แนวโน้ม{name}", "แสดงไม่เกิน 10 ประเทศ + ไทย · คลิกชื่อเพื่อซ่อน/แสดง")
        fig = px.line(t, x="year", y=m, color="country", labels={"year": "ปี", m: unit, "country": "ประเทศ"})
        fig.update_traces(opacity=.6); fig.for_each_trace(lambda tr: tr.update(line=dict(width=4, color=TH_C), opacity=1) if tr.name == "Thailand" else None)
        show(sty(fig, 360, unit))
    src()


# ---------------------------------------------------------------- หน้า 5 ผลการวิเคราะห์
def p_analysis():
    C, B, O, P, _ = DATA
    st.title("ผลการวิเคราะห์")
    CC = C[(C["iso3"] != "WLD") & ~C["no_telecom_data"]]
    W = C[C["iso3"] == "WLD"].set_index("year"); TH = C[C["iso3"] == "THA"].set_index("year"); y = CC[CC["year"] == 2024]
    tabs = st.tabs(["สรุปผลการศึกษา", "สถิติเชิงพรรณนา", "ความสัมพันธ์", "คุณภาพข้อมูล"])
    with tabs[0]:
        inc = y.groupby("income_group")[["mobile_subs_per100", "internet_pct", "ios_share", "price_data_gni", "cov_4g"]].median()
        pw = P[P["iso3"] == "WLD"].pivot(index="year", columns="platform", values="share_pct")
        cn = B[(B["iso3"] == "WLD") & (B["brand_origin"] == "จีน")].groupby("year")["share_pct"].sum()
        r5 = y[["ios_share", "gdp_pc_ppp"]].dropna().corr(method="spearman").iloc[0, 1]; r6 = y[["price_data_gni", "internet_pct"]].dropna().corr(method="spearman").iloc[0, 1]
        Q = [("1. การเข้าถึงมือถือ", f"{W.loc[2024, 'mobile_subs_per100']:.0f} / 100 คน", f"โลก (2024) · รายได้ต่ำ {inc.loc['Low income', 'mobile_subs_per100']:.0f} · ไทย {TH.loc[2024, 'mobile_subs_per100']:.0f}"),
             ("2. มือถือสู่อินเทอร์เน็ต", f"{W.loc[2010, 'internet_pct']:.0f}% → {W.loc[2024, 'internet_pct']:.0f}%", f"โลก 2010–2024 · ไทย {TH.loc[2024, 'internet_pct']:.0f}%"),
             ("3. มือถือแซงคอมพิวเตอร์", f"ปี {pw[pw['Mobile'] > pw['Desktop']].index.min()}", f"โลก · ปี 2025 มือถือ {pw.loc[2025, 'Mobile']:.0f}%"),
             ("4. แบรนด์จีน", f"{cn.loc[2012]:.1f}% → {cn.loc[2025]:.0f}%", "ส่วนแบ่งรวมทั่วโลก 2012–2025"),
             ("5. iOS กับรายได้", f"r = {r5:.2f}", f"iOS รายได้สูง {inc.loc['High income', 'ios_share']:.0f}% · รายได้ต่ำ {inc.loc['Low income', 'ios_share']:.0f}%"),
             ("6. ราคากับการเข้าถึง", f"r = {r6:.2f}", f"ค่าเน็ต รายได้ต่ำ {inc.loc['Low income', 'price_data_gni']:.1f}% ของรายได้ · ไทย {TH.loc[2024, 'price_data_gni']:.1f}%")]
        for row in (Q[:3], Q[3:]):
            for col, (q, a_, t_) in zip(st.columns(3), row):
                col.markdown(f'<div class="ans"><div class="q">{q}</div><div class="a">{a_}</div><div class="t">{t_}</div></div>', unsafe_allow_html=True)
            st.write("")
        sec("ข้อเสนอแนะ")
        with card():
            table([("ค่าเน็ตแพงเทียบรายได้ = ใช้อินเทอร์เน็ตน้อย", "ตั้งเป้าค่าเน็ตมือถือ ≤ 2% ของรายได้ และอุดหนุนแพ็กเกจพื้นฐาน", "หน่วยงานกำกับกิจการโทรคมนาคม", "3–5 ปี"),
                   ("มือถือเป็นช่องทางหลักเข้าเว็บ", "ออกแบบบริการภาครัฐและธุรกิจสำหรับมือถือเป็นหลัก", "ภาครัฐ / ธุรกิจ", "ทันที"),
                   ("รายได้กำหนดการเลือก iOS/Android", "พัฒนาแอปสำหรับ Android ก่อนหรือพร้อมกัน ในประเทศรายได้ปานกลาง", "นักพัฒนาแอป", "1 ปี"),
                   ("แบรนด์จีนขยายตัว", "ใช้การแข่งขันด้านราคาขยายการเข้าถึงสมาร์ตโฟนในกลุ่มรายได้น้อย", "ผู้ให้บริการเครือข่าย", "1–2 ปี")],
                  ["ข้อค้นพบ", "ข้อเสนอแนะ", "ผู้เกี่ยวข้อง", "กรอบเวลา"])
        st.caption("ข้อจำกัด: StatCounter วัดจากการเข้าเว็บ ไม่ใช่ยอดขาย · ไม่มีมูลค่าตลาดและข้อมูลรายรุ่น · ตัวชี้วัดโทรคมนาคมล่าสุดปี 2024 · สหสัมพันธ์ไม่ใช่เหตุและผล")
    with tabs[1]:
        yr = st.select_slider("ปี", options=list(range(2018, 2025)), value=2024, key="an_y")
        yy = CC[CC["year"] == yr]
        rows = []
        for k in METRIC:
            s = yy[k].dropna()
            if len(s) < 5:
                continue
            q1, q3 = s.quantile([.25, .75]); tv = yy.loc[yy["iso3"] == "THA", k]
            rows.append({"ตัวชี้วัด": METRIC[k][0], "n": len(s), "ค่าเฉลี่ย": s.mean(), "มัธยฐาน": s.median(), "S.D.": s.std(), "IQR": q3 - q1,
                         "ต่ำสุด": s.min(), "สูงสุด": s.max(), "ความเบ้": s.skew(), "ไทย": tv.iloc[0] if len(tv) else np.nan})
        stt = pd.DataFrame(rows)
        with card():
            title(f"สถิติเชิงพรรณนา ปี {yr}", "หน่วยวิเคราะห์ = ประเทศ")
            st.dataframe(stt, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in stt.columns if c not in ("ตัวชี้วัด", "n")})
            sk = stt.loc[stt["ความเบ้"].idxmax()]
            note(f"{sk['ตัวชี้วัด']} เบ้มากที่สุด ({sk['ความเบ้']:.1f}) ค่าเฉลี่ยสูงกว่ามัธยฐาน {sk['ค่าเฉลี่ย'] / sk['มัธยฐาน']:.1f} เท่า", "จึงใช้มัธยฐานเป็นค่ากลาง")
        st.download_button("ดาวน์โหลดตารางสถิติ (CSV)", stt.round(2).to_csv(index=False).encode("utf-8-sig"), f"stats_{yr}.csv", "text/csv")
    with tabs[2]:
        V = ["mobile_subs_per100", "internet_pct", "cov_4g", "ios_share", "mobile_web_share", "price_data_gni", "gdp_pc_ppp", "fixed_tel_per100"]
        lab = {**{k: METRIC[k][0] for k in METRIC}, "gdp_pc_ppp": "GDP ต่อหัว"}
        cr = y[V].corr(method="spearman")
        with card():
            title("ค่าสหสัมพันธ์ของสเปียร์แมน ปี 2024", "สีเขียวอมฟ้า = ทิศทางเดียวกัน · สีส้ม = ทิศทางตรงข้าม")
            L = [lab[v] for v in V]
            fig = go.Figure(go.Heatmap(z=cr.values, x=L, y=L, zmin=-1, zmax=1, colorscale=[[0, TH_C], [.5, "#F6F7FB"], [1, ACC]],
                                       text=np.round(cr.values, 2), texttemplate="%{text}", hovertemplate="%{y} กับ %{x}: %{z:.2f}<extra></extra>"))
            fig = sty(fig, 520, legend=False); fig.update_xaxes(tickangle=-35); fig.update_yaxes(autorange="reversed"); show(fig)
            pr = cr.where(np.triu(np.ones(cr.shape, bool), 1)).stack().sort_values(key=abs, ascending=False)
            note(" · ".join(f"{lab[i]} กับ{lab[j]} ({v:.2f})" for (i, j), v in pr.head(3).items()), "ใช้สเปียร์แมนเพราะข้อมูลเบ้", "สหสัมพันธ์ไม่ใช่เหตุและผล")
    with tabs[3]:
        q = side("data_quality_summary")
        l, r = st.columns(2)
        with l, card():
            title("ข้อมูลก่อนและหลังทำความสะอาด")
            if q is not None:
                st.dataframe(q.rename(columns={q.columns[0]: "ตาราง"}), hide_index=True, width="stretch")
            note("แปลงรายเดือนเป็นรายปี ตัดค่า 0 ปลอม รวมชื่อแบรนด์ซ้ำ และตัดกลุ่มภูมิภาค", "ส่วนแบ่งทุกประเทศ-ปีรวมได้ 100% และรหัสประเทศจับคู่ข้ามตารางได้ 100%")
        with r, card():
            title("ธงคุณภาพข้อมูล", "ร้อยละของแถว (ประเทศ-ปี)")
            st.dataframe(pd.DataFrame({"ธง": ["ผู้ใช้มือถือผิดปกติ (ค่าจริง)", "ราคาเน็ตผิดปกติ (ค่าจริง)", "ผู้ใช้มือถือกระโดด > 40/100 คน", "ปีที่เปลี่ยนนิยามแพ็กเกจราคา"],
                                       "ร้อยละ": [C[c].mean() * 100 for c in ["out_mobile_subs_per100", "out_price_data_gni", "yoy_jump", "price_basket_changed"]]}).round(2),
                         hide_index=True, width="stretch")
            st.caption("ค่าผิดปกติอธิบายได้ เช่น ฮ่องกงมีหลายซิมต่อคน ประเทศรายได้ต่ำมากมีค่าเน็ตแพง จึงเก็บไว้และติดธง")
    src()


# ---------------------------------------------------------------- หน้า 6 ข้อมูลและคู่มือ
def p_data():
    C, B, O, P, info = DATA
    st.title("ข้อมูลและคู่มือ")
    t1, t2 = st.tabs(["ตารางข้อมูล", "คู่มือและแหล่งข้อมูล"])
    with t1:
        TB = {"ตัวชี้วัดรายประเทศ-ปี": C, "ส่วนแบ่งแบรนด์": B, "ระบบปฏิบัติการ": O, "อุปกรณ์ที่ใช้เข้าเว็บ": P}
        with card():
            a, b, c = st.columns(3)
            tn = a.selectbox("ตาราง", list(TB), key="dt_t"); d = TB[tn]
            q = b.text_input("ค้นหาประเทศ (ภาษาอังกฤษ)", key="dt_q", placeholder="เช่น Thai, Viet")
            yr = c.slider("ช่วงปี", int(d["year"].min()), int(d["year"].max()), (2020, int(d["year"].max())), key=f"dt_y_{tn}")
            asean = st.checkbox("เฉพาะอาเซียน", key="dt_a")
        f = d[d["year"].between(*yr)]
        if q.strip():
            f = f[f["country"].astype(str).str.contains(q.strip(), case=False, regex=False)]
        if asean:
            f = f[f["is_asean"] == True]
        if f.empty:
            empty("ไม่พบข้อมูลตามเงื่อนไข กรุณาตรวจคำค้นหาหรือกดล้างตัวกรอง", ["dt_q", "dt_a"]); return
        dic = side("data_dictionary"); lab = dict(zip(dic["column"], dic["description_th"])) if dic is not None else {}
        cols = [c for c in f.columns if c not in ("iso2", "region", "income_group", "n_months")]
        out = f[cols].rename(columns=lambda c: lab.get(c, c)[:40])
        st.caption(f"พบ {len(out):,} แถว · คลิกหัวคอลัมน์เพื่อเรียง")
        st.dataframe(out, hide_index=True, width="stretch", height=420)
        a, b, _ = st.columns([1, 1, 3])
        a.download_button("ดาวน์โหลด CSV", out.to_csv(index=False).encode("utf-8-sig"), "phone_data.csv", "text/csv", width="stretch")
        if len(out) <= 50000:
            buf = io.BytesIO()
            try:
                out.to_excel(buf, index=False); b.download_button("ดาวน์โหลด Excel", buf.getvalue(), "phone_data.xlsx", width="stretch")
            except ModuleNotFoundError:
                pass
        if dic is not None:
            with st.expander("คำอธิบายคอลัมน์ (Data Dictionary)"):
                st.dataframe(dic[["table", "column", "description_th", "unit_or_domain", "source"]], hide_index=True, width="stretch")
    with t2:
        l, r = st.columns(2)
        with l, card():
            title("วิธีใช้งาน")
            st.markdown("1. **ภาพรวม** เลือกประเทศ (ค่าเริ่มต้นคือไทย) และปี\n2. **แบรนด์และระบบปฏิบัติการ** แนวโน้มแบรนด์ แบรนด์จีน Android/iOS และแผนที่โลก\n"
                        "3. **การเข้าถึงและราคา** ช่องว่างตามรายได้ และผลของราคาและรายได้\n4. **เปรียบเทียบ** จัดอันดับตามชุดประเทศ\n"
                        "5. **ผลการวิเคราะห์** คำตอบ 6 หัวข้อ สถิติ ความสัมพันธ์ คุณภาพข้อมูล\n6. **ข้อมูลและคู่มือ** ค้นหาและดาวน์โหลด")
        with r, card():
            title("แหล่งข้อมูล")
            st.markdown("- **World Bank WDI** ผู้ใช้มือถือ อินเทอร์เน็ต โทรศัพท์บ้าน GDP ประชากร (CC BY 4.0)\n- **ITU** ความครอบคลุม 2G–5G การมีมือถือ ราคาค่าบริการ\n"
                        "- **StatCounter Global Stats** ส่วนแบ่งแบรนด์ ระบบปฏิบัติการ อุปกรณ์ (CC BY-SA 3.0) วัดจากการเข้าเว็บไซต์ ไม่ใช่ยอดขาย")
        with card():
            title("ประสิทธิภาพของระบบ")
            t0 = time.perf_counter(); _ = C[C["iso3"] == "THA"]; ms = (time.perf_counter() - t0) * 1000
            k = st.columns(3)
            kpi(k[0], "โหลดข้อมูลครั้งแรก", f"{info['load_ms']:.0f}", "ms"); kpi(k[1], "กรองข้อมูล 1 ประเทศ", f"{ms:.1f}", "ms"); kpi(k[2], "จำนวนแถวทั้งหมด", f"{info['rows']:,}")
            st.caption("ไฟล์ Parquet + แคชด้วย st.cache_data · ข้อมูลรายเดือนรวมเป็นรายปีไว้ล่วงหน้า")
    src()


# ---------------------------------------------------------------- เริ่มแอป
if "user" not in st.session_state:
    login(); st.stop()
try:
    with st.spinner("กำลังโหลดข้อมูล..."):
        DATA = load()
except FileNotFoundError as e:
    st.error(f"ไม่พบไฟล์ข้อมูล {e} กรุณาวางไฟล์ .parquet ไว้ในโฟลเดอร์ data หรือโฟลเดอร์เดียวกับ app.py"); st.stop()
except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ ({e}) กรุณารีเฟรชหน้า"); st.stop()

PAGES = [st.Page(p_home, title="ภาพรวม", icon=":material/home:", url_path="home", default=True),
         st.Page(p_brand, title="แบรนด์และระบบปฏิบัติการ", icon=":material/smartphone:", url_path="brands"),
         st.Page(p_access, title="การเข้าถึงและราคา", icon=":material/signal_cellular_alt:", url_path="access"),
         st.Page(p_compare, title="เปรียบเทียบ", icon=":material/leaderboard:", url_path="compare"),
         st.Page(p_analysis, title="ผลการวิเคราะห์", icon=":material/analytics:", url_path="analysis"),
         st.Page(p_data, title="ข้อมูลและคู่มือ", icon=":material/table_view:", url_path="data")]
with st.sidebar:
    st.caption(f"ผู้ใช้: {st.session_state['user']}")
    st.button("ออกจากระบบ", on_click=lambda: st.session_state.pop("user", None))
st.navigation(PAGES, position="top").run()
