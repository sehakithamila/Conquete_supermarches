"""
SoGood - Analyse nutritionnelle des produits de supermarché (Open Food Facts).

Lancer depuis la racine du projet :
    streamlit run app.py
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="SoGood - Que contiennent vraiment nos produits ?",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Chemins & constantes
# ----------------------------------------------------------------------------
def find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [Path.cwd(), *here.parents]:
        if (p / "data" / "gold" / "products").exists():
            return p
    return Path.cwd()


ROOT = find_root()
GOLD_PRODUCTS = ROOT / "data" / "gold" / "products"
RF_PATH = ROOT / "models" / "random_forest_nutriscore.joblib"
ENCODER_PATH = ROOT / "models" / "label_encoder.joblib"
BERT_DIR = ROOT / "models" / "distilbert_product"

GRADES = ["a", "b", "c", "d", "e"]
NUTRI_COLORS = {"a": "#1f8d4e", "b": "#85bb2f", "c": "#f7b500", "d": "#ef7d00", "e": "#d8232a"}
TEXT_ON = {"a": "#ffffff", "b": "#15222B", "c": "#15222B", "d": "#15222B", "e": "#ffffff"}
NOVA_COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#ffb74d", 4: "#c62828"}
NOVA_LABELS = {1: "Non transformé", 2: "Ingrédient culinaire", 3: "Transformé", 4: "Ultra-transformé"}
INK, MUTED, LINE = "#15222B", "#5B6B78", "#DCE2E8"

RF_FEATURES = [
    "energy_kcal_100g", "sugars_100g", "salt_100g", "fat_100g",
    "saturated-fat_100g", "proteins_100g", "fiber_100g", "additives_count", "nova_group",
]
PRODUCT_COLS = [
    "code", "product_name", "brands", "main_category", "nutriscore_grade", "nova_group",
    "additives_count", "energy_kcal_100g", "sugars_100g", "salt_100g", "fat_100g",
    "saturated-fat_100g", "proteins_100g", "fiber_100g", "is_food", "controversy_score",
]
# Axes du radar : plus la surface est grande, moins le profil est favorable
RADAR_AXES = [
    ("Énergie", "energy_kcal_100g"), ("Sucres", "sugars_100g"), ("Sel", "salt_100g"),
    ("Graisses", "fat_100g"), ("Gras saturés", "saturated-fat_100g"), ("Additifs", "additives_count"),
]
TILE_SPECS = [  # libellé, colonne, décimales, unité, "plus bas = mieux"
    ("Énergie", "energy_kcal_100g", 0, " kcal", True),
    ("Sucres", "sugars_100g", 1, " g", True),
    ("Sel", "salt_100g", 2, " g", True),
    ("Graisses", "fat_100g", 1, " g", True),
    ("Gras saturés", "saturated-fat_100g", 1, " g", True),
    ("Protéines", "proteins_100g", 1, " g", False),
    ("Fibres", "fiber_100g", 1, " g", False),
    ("Additifs", "additives_count", 0, "", True),
]
PRESETS = {  # valeurs indicatives pour 100 g
    "Chips": dict(energy=540, sugars=1.5, salt=1.4, fat=34, sat=3.5, prot=6, fiber=4, add=2, nova=4),
    "Yaourt nature": dict(energy=55, sugars=4.5, salt=0.1, fat=1.5, sat=1.0, prot=4.2, fiber=0, add=0, nova=1),
    "Soda": dict(energy=42, sugars=10.6, salt=0.0, fat=0, sat=0, prot=0, fiber=0, add=3, nova=4),
    "Pâte à tartiner": dict(energy=540, sugars=56, salt=0.1, fat=31, sat=11, prot=6.3, fiber=3.4, add=2, nova=4),
    "Salade composée": dict(energy=110, sugars=2.5, salt=0.6, fat=6, sat=1, prot=4, fiber=2, add=0, nova=3),
}

# ----------------------------------------------------------------------------
# Style : étiquette de rayon
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;800&family=Public+Sans:wght@400;500;600&display=swap');
:root {{ --ink:{INK}; --muted:{MUTED}; --line:{LINE}; }}
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea {{font-family:'Public Sans',sans-serif;}}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{font-family:'Archivo',sans-serif !important; font-weight:800; letter-spacing:-.02em; color:var(--ink);}}
.block-container {{padding-top:1.6rem; max-width:1380px;}}
#MainMenu, footer {{visibility:hidden;}}
header[data-testid="stHeader"] {{background:transparent;}}
section[data-testid="stSidebar"] {{border-right:1px solid var(--line);}}
section[data-testid="stSidebar"] h2 {{font-size:1.5rem; margin-bottom:0;}}

/* Hero : le logo Nutri-Score devient un graphique */
.hero {{display:flex; flex-wrap:wrap; gap:2.5rem; align-items:flex-end; justify-content:space-between;
        padding:.4rem 0 1.6rem; border-bottom:2px solid var(--ink);}}
.hero-text {{flex:1 1 380px; max-width:620px;}}
.hero-text h1 {{font-size:clamp(2rem,4.2vw,3.3rem); line-height:1.02; margin:0 0 .8rem; padding:0;}}
.hero-text p {{font-size:1.08rem; line-height:1.5; color:var(--muted); margin:0; max-width:46ch;}}
.scale {{display:flex; gap:10px; align-items:flex-end; flex:0 0 auto;}}
.sc {{display:flex; flex-direction:column; align-items:center; gap:6px; width:74px;}}
.sc-bar {{width:100%; border-radius:6px 6px 2px 2px; display:flex; justify-content:center; padding-top:10px;
          transform-origin:bottom; animation:grow .9s cubic-bezier(.2,.8,.2,1) both; animation-delay:calc(var(--i) * 110ms);}}
.sc-bar span {{font-family:'Archivo',sans-serif; font-weight:800; font-size:2.1rem; line-height:1;}}
.sc-pct {{font-family:'Archivo',sans-serif; font-weight:800; font-size:1.05rem; color:var(--ink);}}
@keyframes grow {{from {{transform:scaleY(.04); opacity:.2;}} to {{transform:scaleY(1); opacity:1;}}}}
@media (prefers-reduced-motion: reduce) {{ .sc-bar {{animation:none;}} }}

/* Chiffres clés : tableau à filets, pas de cartes */
.stats {{display:flex; flex-wrap:wrap; border-bottom:1px solid var(--line); margin:0 0 1.8rem;}}
.stat {{flex:1 1 170px; padding:1rem 1.2rem 1rem 0; margin-right:1.2rem; border-right:1px solid var(--line);}}
.stat:last-child {{border-right:0; margin-right:0;}}
.stat b {{display:block; font-family:'Archivo',sans-serif; font-weight:800; font-size:2.1rem; line-height:1.05; color:var(--ink);}}
.stat span {{font-size:.9rem; color:var(--muted);}}

.note {{border-left:3px solid var(--ink); padding:.15rem 0 .15rem .9rem; margin:.3rem 0 1rem; color:var(--ink); line-height:1.45;}}
.note small {{color:var(--muted);}}

/* Fiche produit */
.label-head {{background:#fff; border:1px solid var(--line); border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:.8rem;}}
.label-head h3 {{margin:0 0 .15rem; font-size:1.5rem;}}
.label-head .meta {{color:var(--muted); font-size:.92rem; margin-bottom:.9rem;}}
.tiles {{display:grid; grid-template-columns:repeat(4,1fr); gap:8px;}}
.tile {{background:#fff; border:1px solid var(--line); border-top:3px solid var(--ink); border-radius:3px; padding:.6rem .8rem;}}
.tile .l {{font-size:.8rem; color:var(--muted);}}
.tile .v {{font-family:'Archivo',sans-serif; font-weight:800; font-size:1.4rem; line-height:1.2;}}
.tile .d {{font-size:.78rem; font-weight:600; color:var(--muted);}}
.tile .d.good {{color:#1f8d4e;}} .tile .d.bad {{color:#d8232a;}}
@media (max-width:800px) {{ .tiles {{grid-template-columns:repeat(2,1fr);}} }}

.pill {{display:inline-block; font-weight:600; font-size:.85rem; padding:.2rem .7rem; border-radius:999px; margin-right:.4rem;}}
.chip {{display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.8rem; font-weight:600; margin:.12rem .25rem .12rem 0;}}
.chip.ok {{background:#E3F3E9; color:#14693a;}} .chip.ko {{background:#F6E1E2; color:#a11a20;}}
.ns-row {{display:inline-flex; gap:4px; align-items:flex-end;}}
.ns-box {{display:flex; align-items:center; justify-content:center; font-family:'Archivo',sans-serif; font-weight:800; border-radius:6px;}}

button[data-baseweb="tab"] p {{font-family:'Archivo',sans-serif; font-weight:600; font-size:1.02rem;}}
</style>
""",
    unsafe_allow_html=True,
)


def nutri_badge(grade: str, scale: float = 1.0) -> str:
    """Logo Nutri-Score A→E, la lettre active est agrandie."""
    boxes = []
    for g in GRADES:
        on = g == grade
        w = int((58 if on else 36) * scale)
        h = int((68 if on else 44) * scale)
        fs = int((34 if on else 18) * scale)
        op = 1 if on else 0.25
        boxes.append(
            f"<div class='ns-box' style='background:{NUTRI_COLORS[g]};color:{TEXT_ON[g]};"
            f"width:{w}px;height:{h}px;font-size:{fs}px;opacity:{op}'>{g.upper()}</div>"
        )
    return f"<div class='ns-row'>{''.join(boxes)}</div>"


def nova_pill(n) -> str:
    if n is None or pd.isna(n):
        return "<span class='pill' style='background:#E4E8EC;color:#15222B'>NOVA inconnu</span>"
    n = int(n)
    fg = "#fff" if n in (1, 4) else INK
    return (f"<span class='pill' style='background:{NOVA_COLORS[n]};color:{fg}'>"
            f"NOVA {n}, {NOVA_LABELS[n].lower()}</span>")


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color=INK),
        title=dict(font=dict(family="Archivo, sans-serif", size=17, color=INK)),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#E9EDF1", zeroline=False)
    fig.update_yaxes(gridcolor="#E9EDF1", zeroline=False)
    return fig


def round_bars(fig: go.Figure, radius: int = 6) -> go.Figure:
    try:
        fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
    except Exception:
        pass
    return fig


def pretty_cat(c: str) -> str:
    c = re.sub(r"^[a-z]{2}:", "", c or "").replace("-", " ").strip()
    return c.capitalize() if c else "Sans catégorie"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# ----------------------------------------------------------------------------
# Données et modèles
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des produits…")
def load_products() -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(GOLD_PRODUCTS), format="parquet")
    cols = [c for c in PRODUCT_COLS if c in dataset.schema.names]
    df = dataset.to_table(columns=cols).to_pandas()
    for c in ("product_name", "brands", "main_category", "nutriscore_grade"):
        df[c] = df[c].fillna("").astype(str)
    df["nutriscore_grade"] = df["nutriscore_grade"].str.lower().str.strip()
    df["brand"] = df["brands"].str.split(",").str[0].str.strip().str.lower()
    df["search"] = (df["product_name"] + " " + df["brands"]).str.lower()
    return df


@st.cache_resource
def load_rf():
    if not (RF_PATH.exists() and ENCODER_PATH.exists()):
        return None
    import joblib
    return {"model": joblib.load(RF_PATH), "encoder": joblib.load(ENCODER_PATH)}


@st.cache_resource(show_spinner="Chargement de DistilBERT…")
def load_bert():
    if not (BERT_DIR / "config.json").exists():
        return None, None, f"Dossier modèle introuvable : {BERT_DIR}"
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        return None, None, f"transformers/torch non installés : {e}"
    tok = AutoTokenizer.from_pretrained(str(BERT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
    model.eval()
    return tok, model, None


def predict_rf(feats: dict) -> dict:
    bundle = load_rf()
    model = bundle["model"]
    x = pd.DataFrame([[float(feats[f]) for f in RF_FEATURES]], columns=RF_FEATURES)
    probs = model.predict_proba(x)[0]
    classes = bundle["encoder"].inverse_transform(model.classes_)
    p = {g: 0.0 for g in GRADES}
    p.update({str(c): float(v) for c, v in zip(classes, probs)})
    return {"grade": max(p, key=p.get), "probs": p}


def predict_bert(text: str) -> float:
    """Probabilité d'être ultra-transformé (NOVA 4), label 1 dans 04_train_bert.py."""
    import torch

    tok, model, _ = load_bert()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().tolist()
    return float(probs[1])


try:
    df_all = load_products()
except Exception as e:
    st.error(f"Données gold introuvables dans `{GOLD_PRODUCTS}` : {e}\n\n"
             "Lance d'abord `python src/02_etl_spark.py`.")
    st.stop()


@st.cache_resource
def top_categories(n: int = 40) -> list[str]:
    vc = df_all["main_category"].replace("", np.nan).dropna().value_counts().head(n)
    return vc.index.tolist()


@st.cache_data(max_entries=2048, show_spinner=False)
def category_median(cat: str, col: str) -> float:
    return float(df_all.loc[df_all["main_category"] == cat, col].median())


@st.cache_resource
def radar_scale() -> dict:
    return {c: max(float(df_all[c].quantile(0.95)), 1e-9) for _, c in RADAR_AXES}


# ----------------------------------------------------------------------------
# Filtres (barre latérale)
# ----------------------------------------------------------------------------
st.sidebar.markdown("## SoGood")
st.sidebar.caption("Données Open Food Facts, projet IPSSI")
st.sidebar.markdown("#### Filtrer les produits")

sel_cats = st.sidebar.multiselect("Catégories (toutes si vide)", top_categories(), format_func=pretty_cat)
nova_sel = st.sidebar.multiselect(
    "Groupes NOVA", [1, 2, 3, 4], default=[1, 2, 3, 4],
    format_func=lambda n: f"{n}, {NOVA_LABELS[n].lower()}",
)
incl_nova_na = st.sidebar.checkbox("Inclure les produits sans NOVA", value=True)
min_add = st.sidebar.slider("Nombre minimum d'additifs", 0, 15, 0)
only_food = st.sidebar.checkbox("Produits alimentaires uniquement", value=True)

mask = np.ones(len(df_all), dtype=bool)
if only_food and "is_food" in df_all.columns:
    mask &= df_all["is_food"].fillna(True).to_numpy(dtype=bool)
if sel_cats:
    mask &= df_all["main_category"].isin(sel_cats).to_numpy()
nova_mask = df_all["nova_group"].isin(nova_sel)
if incl_nova_na:
    nova_mask |= df_all["nova_group"].isna()
mask &= nova_mask.to_numpy()
if min_add > 0:
    mask &= (df_all["additives_count"].fillna(0) >= min_add).to_numpy()
sub = df_all[mask]

st.sidebar.markdown(f"**{fmt_int(len(sub))}** produits sur {fmt_int(len(df_all))}")
st.sidebar.progress(len(sub) / max(len(df_all), 1))
st.sidebar.markdown("#### Modèles")
rf_ok = load_rf() is not None
bert_ok = (BERT_DIR / "config.json").exists()
st.sidebar.markdown(
    f"<span class='chip {'ok' if rf_ok else 'ko'}'>Random Forest {'prêt' if rf_ok else 'absent'}</span>"
    f"<span class='chip {'ok' if bert_ok else 'ko'}'>DistilBERT {'prêt' if bert_ok else 'absent'}</span>",
    unsafe_allow_html=True,
)

if sub.empty:
    st.warning("Aucun produit ne correspond à ces filtres. Élargis la sélection dans la barre latérale.")
    st.stop()

known = sub[sub["nutriscore_grade"].isin(GRADES)]
counts = known["nutriscore_grade"].value_counts().reindex(GRADES).fillna(0)

# ----------------------------------------------------------------------------
# Hero : la répartition du Nutri-Score, dessinée avec le logo lui-même
# ----------------------------------------------------------------------------
bars = ""
if counts.sum() > 0:
    share = counts / counts.sum()
    mx = max(float(share.max()), 1e-9)
    for i, g in enumerate(GRADES):
        h = 70 + 210 * float(share[g]) / mx
        bars += (f"<div class='sc' style='--i:{i}'><div class='sc-bar' style='height:{h:.0f}px;"
                 f"background:{NUTRI_COLORS[g]};color:{TEXT_ON[g]}'><span>{g.upper()}</span></div>"
                 f"<div class='sc-pct'>{share[g]:.0%}</div></div>")

st.markdown(
    "<div class='hero'><div class='hero-text'>"
    "<h1>Que contiennent vraiment les produits de nos supermarchés ?</h1>"
    f"<p>{fmt_int(len(sub))} produits passés au crible : Nutri-Score, niveau de transformation, "
    "additifs. Les barres montrent la part de chaque note, et se recalculent avec tes filtres.</p></div>"
    f"<div class='scale'>{bars}</div></div>",
    unsafe_allow_html=True,
)

nova_known = sub["nova_group"].dropna()
share_n4 = (nova_known == 4).mean() * 100 if len(nova_known) else 0
share_de = known["nutriscore_grade"].isin(["d", "e"]).mean() * 100 if len(known) else 0
n_brands = sub["brand"].replace("", np.nan).nunique()
st.markdown(
    "<div class='stats'>"
    f"<div class='stat'><b>{share_de:.0f} %</b><span>des produits notés sont D ou E</span></div>"
    f"<div class='stat'><b>{share_n4:.0f} %</b><span>sont ultra-transformés (NOVA 4)</span></div>"
    f"<div class='stat'><b>{sub['additives_count'].fillna(0).mean():.1f}</b><span>additifs par produit en moyenne</span></div>"
    f"<div class='stat'><b>{fmt_int(n_brands)}</b><span>marques différentes</span></div>"
    "</div>",
    unsafe_allow_html=True,
)

tab_overview, tab_explore, tab_compare, tab_predict, tab_brands = st.tabs(
    ["Vue d'ensemble", "Explorer", "Comparer", "Prédire", "Marques"]
)


# ----------------------------------------------------------------------------
# Composants de fiche produit
# ----------------------------------------------------------------------------
def radar_values(values: dict) -> list[float]:
    scale = radar_scale()
    out = []
    for _, col in RADAR_AXES:
        v = values.get(col)
        out.append(0.0 if v is None or pd.isna(v) else float(min(max(v, 0) / scale[col], 1.0)))
    return out


def radar_figure(series: list[tuple[str, dict, str, bool]], height: int = 360) -> go.Figure:
    """series = [(nom, valeurs, couleur, en_pointillé)]"""
    labels = [a for a, _ in RADAR_AXES]
    fig = go.Figure()
    for name, values, color, dashed in series:
        r = radar_values(values)
        fig.add_trace(go.Scatterpolar(
            r=r + r[:1], theta=labels + labels[:1], name=name,
            fill="none" if dashed else "toself",
            line=dict(color=color, width=2, dash="dash" if dashed else "solid"),
            fillcolor=None if dashed else (
                f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.16)"),
        ))
    fig.update_layout(
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#E3E8EC"),
                   angularaxis=dict(gridcolor="#E3E8EC")),
        showlegend=True, legend=dict(orientation="h", y=-0.1),
    )
    return style_fig(fig, height)


def tiles_html(row: pd.Series, cat: str) -> str:
    parts = []
    for label, key, nd, unit, lower_better in TILE_SPECS:
        v = row[key]
        if pd.isna(v):
            parts.append(f"<div class='tile'><div class='l'>{label}</div><div class='v'>-</div></div>")
            continue
        delta, cls = "", ""
        if cat:
            med = category_median(cat, key)
            if not np.isnan(med):
                diff = float(v) - med
                if round(diff, nd) != 0:
                    cls = "good" if (diff < 0) == lower_better else "bad"
                delta = f"{diff:+.{nd}f} vs médiane de la catégorie"
        parts.append(
            f"<div class='tile'><div class='l'>{label}</div><div class='v'>{v:.{nd}f}{unit}</div>"
            f"<div class='d {cls}'>{delta}</div></div>")
    return f"<div class='tiles'>{''.join(parts)}</div>"


def product_values(row: pd.Series) -> dict:
    return {col: row[col] for _, col in RADAR_AXES}


def product_card(row: pd.Series, key: str) -> None:
    name = row["product_name"] or "Produit sans nom"
    brand = row["brands"].split(",")[0].strip() if row["brands"] else "Marque inconnue"
    cat = row["main_category"]
    st.markdown(
        f"<div class='label-head'><h3>{html.escape(name)}</h3>"
        f"<div class='meta'>{html.escape(brand)}, {html.escape(pretty_cat(cat))}, "
        f"code {html.escape(str(row['code']))}</div>"
        f"{nutri_badge(row['nutriscore_grade'])}<div style='height:.7rem'></div>{nova_pill(row['nova_group'])}</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([2, 3])
    with left:
        series = [(name[:28], product_values(row), "#0f766e", False)]
        if cat:
            med = {col: category_median(cat, col) for _, col in RADAR_AXES}
            series.append(("Médiane de la catégorie", med, "#8896A3", True))
        st.plotly_chart(radar_figure(series, 330), width="stretch", key=f"radar_{key}")
    with right:
        st.markdown(tiles_html(row, cat), unsafe_allow_html=True)
        st.caption("Valeurs pour 100 g. Vert : meilleur que la médiane de la catégorie, rouge : moins bon.")


# ============================== VUE D'ENSEMBLE ==============================
with tab_overview:
    # Deux constats calculés sur les produits filtrés
    g = known.assign(de=known["nutriscore_grade"].isin(["d", "e"])).groupby("main_category").agg(
        n=("de", "size"), de=("de", "mean"))
    g = g[(g["n"] >= 100) & (g.index != "")]
    if len(g) >= 2:
        worst, best = g["de"].idxmax(), g["de"].idxmin()
        n1, n2 = st.columns(2)
        n1.markdown(f"<div class='note'><b>{html.escape(pretty_cat(worst))}</b> : {g.loc[worst, 'de']:.0%} de produits D ou E."
                    f"<br><small>La catégorie la moins bien notée (au moins 100 produits).</small></div>",
                    unsafe_allow_html=True)
        n2.markdown(f"<div class='note'><b>{html.escape(pretty_cat(best))}</b> : {g.loc[best, 'de']:.0%} de produits D ou E."
                    f"<br><small>La catégorie la mieux notée (au moins 100 produits).</small></div>",
                    unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        nova_counts = sub["nova_group"].dropna().astype(int).value_counts().reindex([1, 2, 3, 4]).fillna(0)
        fig = go.Figure(go.Bar(
            x=[f"{n}. {NOVA_LABELS[n]}" for n in nova_counts.index], y=nova_counts.values,
            marker_color=[NOVA_COLORS[n] for n in nova_counts.index],
            hovertemplate="%{x}<br>%{y:,.0f} produits<extra></extra>"))
        fig.update_layout(title="Combien de produits sont transformés ? (NOVA)")
        st.plotly_chart(round_bars(style_fig(fig)), width="stretch", key="ov_nova")
    with col_b:
        both = sub[sub["nutriscore_grade"].isin(GRADES) & sub["nova_group"].notna()]
        if len(both):
            ct = pd.crosstab(both["nova_group"].astype(int), both["nutriscore_grade"])
            ct = ct.reindex(index=[1, 2, 3, 4], columns=GRADES, fill_value=0)
            pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
            pct.index = [f"NOVA {i}" for i in pct.index]
            pct.columns = [c.upper() for c in pct.columns]
            fig = px.imshow(pct, text_auto=".0%", aspect="auto",
                            color_continuous_scale=[[0, "#F1F6F2"], [1, "#1f8d4e"]],
                            labels=dict(x="Nutri-Score", y="", color="Part"))
            fig.update_layout(title="Transformation et Nutri-Score vont-ils de pair ?",
                              coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig), width="stretch", key="ov_heat")

    col_c, col_d = st.columns(2)
    with col_c:
        top = known["main_category"].replace("", np.nan).dropna().value_counts().head(8).index
        sb = (known[known["main_category"].isin(top)]
              .groupby(["main_category", "nutriscore_grade"]).size().reset_index(name="n"))
        if len(sb):
            sb["Catégorie"] = sb["main_category"].map(pretty_cat)
            sb["Note"] = sb["nutriscore_grade"].str.upper()
            cmap = {k.upper(): v for k, v in NUTRI_COLORS.items()}
            cmap["(?)"] = "#E7ECF0"
            fig = px.sunburst(sb, path=["Catégorie", "Note"], values="n", color="Note",
                              color_discrete_map=cmap)
            fig.update_layout(title="Les 8 plus grandes catégories, note par note")
            fig.update_traces(hovertemplate="%{label}<br>%{value:,.0f} produits<extra></extra>")
            st.plotly_chart(style_fig(fig, 420), width="stretch", key="ov_sun")
    with col_d:
        scat = known.dropna(subset=["sugars_100g", "salt_100g"])
        if len(scat) > 3000:
            scat = scat.sample(3000, random_state=0)
        if len(scat):
            scat = scat.assign(Note=scat["nutriscore_grade"].str.upper())
            fig = px.scatter(
                scat, x="sugars_100g", y="salt_100g", color="Note",
                category_orders={"Note": [c.upper() for c in GRADES]},
                color_discrete_map={k.upper(): v for k, v in NUTRI_COLORS.items()},
                hover_data=["product_name", "brands"], opacity=0.6,
                labels={"sugars_100g": "Sucres (g/100 g)", "salt_100g": "Sel (g/100 g)"},
                title="Sucres et sel, sur 3 000 produits au hasard")
            fig.update_layout(xaxis_range=[0, 80], yaxis_range=[0, 6])
            st.plotly_chart(style_fig(fig, 420), width="stretch", key="ov_scatter")

# ================================ EXPLORER ==================================
with tab_explore:
    f1, f2 = st.columns([3, 2])
    q = f1.text_input("Chercher un produit ou une marque", placeholder="Nutella, chips, yaourt bio…")
    grade_filter = f2.multiselect("Nutri-Score", [c.upper() for c in GRADES], placeholder="Toutes les notes")

    s = sub
    if q:
        s = s[s["search"].str.contains(q.lower().strip(), regex=False, na=False)]
    if grade_filter:
        s = s[s["nutriscore_grade"].str.upper().isin(grade_filter)]

    st.caption(f"{fmt_int(len(s))} résultats, les 300 premiers sont affichés. "
               "Clique sur une ligne pour ouvrir la fiche du produit.")
    view = s.head(300).reset_index(drop=True)
    display = view.assign(nutriscore_grade=view["nutriscore_grade"].str.upper())
    event = st.dataframe(
        display, width="stretch", height=360, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="explore_table",
        column_order=["product_name", "brands", "nutriscore_grade", "nova_group",
                      "additives_count", "sugars_100g", "salt_100g", "controversy_score"],
        column_config={
            "product_name": st.column_config.TextColumn("Produit", width="large"),
            "brands": st.column_config.TextColumn("Marque"),
            "nutriscore_grade": st.column_config.TextColumn("Note", width="small"),
            "nova_group": st.column_config.NumberColumn("NOVA", format="%d", width="small"),
            "additives_count": st.column_config.NumberColumn("Additifs", format="%d", width="small"),
            "sugars_100g": st.column_config.ProgressColumn("Sucres (g)", min_value=0, max_value=60, format="%.1f"),
            "salt_100g": st.column_config.ProgressColumn("Sel (g)", min_value=0, max_value=3, format="%.2f"),
            "controversy_score": st.column_config.NumberColumn("Controverse", format="%.1f"),
        },
    )
    picked = event.selection.rows if event and event.selection else []
    if picked and picked[0] < len(view):
        st.markdown("")
        product_card(view.iloc[picked[0]], key="explore")

# ================================ COMPARER ==================================
def pick_product(col, label: str, key: str):
    q = col.text_input(label, key=f"{key}_q", placeholder="Nom ou marque du produit")
    if not q.strip():
        col.caption("Tape un nom pour lancer la recherche.")
        return None
    res = df_all[df_all["search"].str.contains(q.lower().strip(), regex=False, na=False)].head(40)
    if res.empty:
        col.warning("Aucun produit trouvé. Essaie un autre mot.")
        return None
    idx = col.selectbox(
        "Résultat", res.index.tolist(), key=f"{key}_sel",
        format_func=lambda i: f"{df_all.at[i, 'product_name'][:55]} ({df_all.at[i, 'brand'] or 'marque inconnue'})")
    return df_all.loc[idx]


with tab_compare:
    st.markdown("### Deux produits, côte à côte")
    ca, cb = st.columns(2)
    pa = pick_product(ca, "Premier produit", "cmp1")
    pb = pick_product(cb, "Second produit", "cmp2")
    if pa is not None and pb is not None:
        h1, h2 = st.columns(2)
        for col, p in ((h1, pa), (h2, pb)):
            with col:
                st.markdown(
                    f"<div class='label-head'><h3>{html.escape(p['product_name'] or 'Sans nom')}</h3>"
                    f"<div class='meta'>{html.escape(p['brand'] or 'marque inconnue')}</div>"
                    f"{nutri_badge(p['nutriscore_grade'], 0.8)}<div style='height:.6rem'></div>"
                    f"{nova_pill(p['nova_group'])}</div>", unsafe_allow_html=True)
        r1, r2 = st.columns([3, 2])
        with r1:
            st.plotly_chart(
                radar_figure([(pa["product_name"][:28] or "Produit 1", product_values(pa), "#0f766e", False),
                              (pb["product_name"][:28] or "Produit 2", product_values(pb), "#7c3aed", False)], 400),
                width="stretch", key="cmp_radar")
        with r2:
            rows = []
            for label, key, nd, unit, _ in TILE_SPECS:
                rows.append({"Pour 100 g": label,
                             "Produit 1": None if pd.isna(pa[key]) else round(float(pa[key]), nd),
                             "Produit 2": None if pd.isna(pb[key]) else round(float(pb[key]), nd)})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=330)
            st.caption("Plus la surface du radar est grande, moins le profil est favorable.")

# ================================= PRÉDIRE ==================================
def apply_preset(name: str) -> None:
    for k, v in PRESETS[name].items():
        st.session_state[f"rf_{k}"] = int(v) if k in ("add", "nova") else float(v)


with tab_predict:
    sub_rf, sub_bert = st.tabs(["Estimer le Nutri-Score", "Repérer l'ultra-transformé"])

    with sub_rf:
        if not rf_ok:
            st.error("Modèle introuvable. Lance `python src/03_train_rf.py`.")
        else:
            st.markdown("### Règle les curseurs, la note se met à jour")
            for k, v in PRESETS["Chips"].items():
                st.session_state.setdefault(f"rf_{k}", int(v) if k in ("add", "nova") else float(v))

            st.caption("Pars d'un produit courant (valeurs indicatives pour 100 g) :")
            pcols = st.columns(len(PRESETS))
            for col, name in zip(pcols, PRESETS):
                col.button(name, on_click=apply_preset, args=(name,), key=f"preset_{name}", width="stretch")

            left, right = st.columns([3, 2], gap="large")
            with left:
                a, b = st.columns(2)
                a.slider("Énergie (kcal)", 0.0, 900.0, step=5.0, key="rf_energy")
                b.slider("Sucres (g)", 0.0, 100.0, step=0.5, key="rf_sugars")
                a.slider("Sel (g)", 0.0, 5.0, step=0.05, key="rf_salt")
                b.slider("Graisses (g)", 0.0, 100.0, step=0.5, key="rf_fat")
                a.slider("Graisses saturées (g)", 0.0, 50.0, step=0.5, key="rf_sat")
                b.slider("Protéines (g)", 0.0, 60.0, step=0.5, key="rf_prot")
                a.slider("Fibres (g)", 0.0, 40.0, step=0.5, key="rf_fiber")
                b.slider("Additifs", 0, 20, step=1, key="rf_add")
                st.select_slider("Groupe NOVA", [1, 2, 3, 4], key="rf_nova",
                                 format_func=lambda n: f"{n}, {NOVA_LABELS[n].lower()}")
            res = predict_rf({
                "energy_kcal_100g": st.session_state["rf_energy"], "sugars_100g": st.session_state["rf_sugars"],
                "salt_100g": st.session_state["rf_salt"], "fat_100g": st.session_state["rf_fat"],
                "saturated-fat_100g": st.session_state["rf_sat"], "proteins_100g": st.session_state["rf_prot"],
                "fiber_100g": st.session_state["rf_fiber"], "additives_count": st.session_state["rf_add"],
                "nova_group": st.session_state["rf_nova"],
            })
            with right:
                st.markdown(nutri_badge(res["grade"], 1.7), unsafe_allow_html=True)
                st.markdown(f"**Note estimée : {res['grade'].upper()}**, avec {res['probs'][res['grade']]:.0%} de confiance.")
                fig = go.Figure(go.Bar(
                    x=[res["probs"][c] for c in GRADES], y=[c.upper() for c in GRADES], orientation="h",
                    marker_color=[NUTRI_COLORS[c] for c in GRADES],
                    hovertemplate="%{y} : %{x:.1%}<extra></extra>"))
                fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(autorange="reversed"),
                                  title="Probabilité de chaque note")
                st.plotly_chart(round_bars(style_fig(fig, 280), 4), width="stretch", key="rf_probs")

            with st.expander("Qu'est-ce qui pèse le plus dans la décision du modèle ?"):
                imp = pd.Series(load_rf()["model"].feature_importances_, index=RF_FEATURES).sort_values()
                fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=INK))
                fig.update_layout(xaxis=dict(tickformat=".0%"), title="Importance des variables")
                st.plotly_chart(round_bars(style_fig(fig, 320), 4), width="stretch", key="rf_imp")

    with sub_bert:
        st.markdown("### Les ingrédients trahissent-ils un produit ultra-transformé ?")
        st.caption("DistilBERT lit la liste d'ingrédients et estime la probabilité d'un classement NOVA 4. "
                   "Il ne prédit pas le Nutri-Score.")
        tok, bert, err = load_bert()
        if bert is None:
            st.error(f"{err}\n\nLance `python src/04_train_bert.py`.")
        else:
            ingredients = st.text_area(
                "Liste d'ingrédients", height=130,
                value="Sucre, huile de palme, noisettes 13%, cacao maigre 7,4%, lait écrémé en poudre 6,6%, "
                      "lactosérum en poudre, émulsifiant : lécithines, vanilline.")
            if st.button("Analyser les ingrédients", type="primary"):
                if not ingredients.strip():
                    st.warning("Colle une liste d'ingrédients pour lancer l'analyse.")
                else:
                    with st.spinner("Analyse en cours…"):
                        p = predict_bert(ingredients)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=p * 100, number=dict(suffix=" %"),
                        title=dict(text="Probabilité d'être ultra-transformé"),
                        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=INK),
                                   steps=[dict(range=[0, 33], color="#cfe8d6"),
                                          dict(range=[33, 66], color="#fde7b0"),
                                          dict(range=[66, 100], color="#f3b8ba")])))
                    st.plotly_chart(style_fig(fig, 300), width="stretch", key="bert_gauge")
                    if p >= 0.5:
                        st.markdown(nova_pill(4), unsafe_allow_html=True)
                    else:
                        st.success("Plutôt non ultra-transformé (NOVA 1 à 3).")

# ================================= MARQUES ==================================
with tab_brands:
    st.markdown("### Quelles marques cumulent additifs, sucre et mauvaises notes ?")
    st.caption("Le score de controverse additionne le nombre d'additifs, 5 points pour un Nutri-Score D ou E "
               "et 3 points pour un NOVA 4. Calcul sur les produits filtrés.")
    o1, o2, o3 = st.columns([2, 2, 1])
    order = o1.radio("Classement", ["Les plus controversées", "Les moins controversées"], horizontal=True)
    min_n = o2.slider("Produits minimum par marque", 5, 100, 10)
    top_n = o3.selectbox("Afficher", [10, 20, 30], index=1)

    grouped = (sub[sub["brand"] != ""].groupby("brand")
               .agg(n=("code", "count"), controverse=("controversy_score", "mean"),
                    additifs=("additives_count", "mean"), sucres=("sugars_100g", "mean"),
                    sel=("salt_100g", "mean"))
               .reset_index().rename(columns={"brand": "marque"}))
    grouped = grouped[grouped["n"] >= min_n]

    if grouped.empty:
        st.info("Aucune marque n'atteint ce seuil. Baisse le nombre minimum de produits.")
    else:
        bubbles = grouped.copy()
        if len(bubbles) > 400:
            bubbles = bubbles.nlargest(400, "n")
        fig = px.scatter(
            bubbles, x="additifs", y="sucres", size="n", color="controverse", size_max=42,
            hover_name="marque", color_continuous_scale=["#1f8d4e", "#f7b500", "#d8232a"],
            labels={"additifs": "Additifs par produit (moyenne)", "sucres": "Sucres (g/100 g, moyenne)",
                    "controverse": "Controverse", "n": "Produits"},
            title="Chaque bulle est une marque, sa taille est son nombre de produits")
        st.plotly_chart(style_fig(fig, 460), width="stretch", key="brands_bubble")

        ranked = grouped.sort_values("controverse", ascending=(order == "Les moins controversées")).head(top_n)
        plot_df = ranked.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=plot_df["controverse"], y=plot_df["marque"], orientation="h",
            marker=dict(color=plot_df["controverse"], colorscale=["#1f8d4e", "#f7b500", "#d8232a"]),
            customdata=plot_df[["n"]],
            hovertemplate="<b>%{y}</b><br>Controverse %{x:.2f}<br>%{customdata[0]} produits<extra></extra>"))
        fig.update_layout(title=f"{order} : top {len(ranked)}", margin=dict(l=160))
        st.plotly_chart(round_bars(style_fig(fig, max(380, 26 * len(ranked) + 90)), 4),
                        width="stretch", key="brands_bar")
        st.dataframe(
            ranked, width="stretch", hide_index=True,
            column_config={
                "marque": st.column_config.TextColumn("Marque"),
                "n": st.column_config.NumberColumn("Produits", format="%d"),
                "controverse": st.column_config.NumberColumn("Controverse", format="%.2f"),
                "additifs": st.column_config.NumberColumn("Additifs moy.", format="%.2f"),
                "sucres": st.column_config.NumberColumn("Sucres moy. (g)", format="%.1f"),
                "sel": st.column_config.NumberColumn("Sel moy. (g)", format="%.2f"),
            },
        )

st.markdown("---")
st.caption("SoGood utilise les données d'Open Food Facts (licence ODbL). "
           "Les prédictions sont indicatives et ne remplacent pas le calcul officiel du Nutri-Score.")