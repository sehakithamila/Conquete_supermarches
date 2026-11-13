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
    page_title="SoGood · Analyse nutritionnelle",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Chemins & constantes
# ----------------------------------------------------------------------------
def find_root() -> Path:
    """Remonte jusqu'au dossier qui contient data/gold/products."""
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
NOVA_COLORS = {1: "#2e7d32", 2: "#9ccc65", 3: "#ffb74d", 4: "#c62828"}
NOVA_LABELS = {1: "Non transformé", 2: "Ingrédient culinaire", 3: "Transformé", 4: "Ultra-transformé"}

# Les features du RF doivent être identiques à celles de 03_train_rf.py
RF_FEATURES = [
    "energy_kcal_100g", "sugars_100g", "salt_100g", "fat_100g",
    "saturated-fat_100g", "proteins_100g", "fiber_100g", "additives_count", "nova_group",
]

PRODUCT_COLS = [
    "code", "product_name", "brands", "main_category", "nutriscore_grade", "nova_group",
    "additives_count", "energy_kcal_100g", "sugars_100g", "salt_100g", "fat_100g",
    "saturated-fat_100g", "proteins_100g", "fiber_100g", "is_food", "controversy_score",
]

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg,#1f8d4e 0%,#85bb2f 55%,#f7b500 100%);
       border-radius: 20px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; color: #fff;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: .35rem 0 0; font-size: 1.05rem; opacity: .95;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.18);
       border-radius: 14px; padding: .8rem 1rem;}
.ns-row {display: inline-flex; gap: 4px; align-items: center;}
.ns-box {display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 800;
         border-radius: 8px;}
.pill {display: inline-block; color: #fff; font-weight: 600; font-size: .85rem;
       padding: .25rem .7rem; border-radius: 999px; margin-right: .4rem;}
.muted {opacity: .65; font-size: .9rem;}
</style>
""",
    unsafe_allow_html=True,
)


def nutri_badge(grade: str, scale: float = 1.0) -> str:
    """Logo Nutri-Score A→E avec la lettre active mise en avant."""
    boxes = []
    for g in GRADES:
        on = g == grade
        w, fs = (int(56 * scale), int(30 * scale)) if on else (int(36 * scale), int(17 * scale))
        h = int(64 * scale) if on else int(44 * scale)
        op = 1 if on else 0.28
        boxes.append(
            f"<div class='ns-box' style='background:{NUTRI_COLORS[g]};width:{w}px;height:{h}px;"
            f"font-size:{fs}px;opacity:{op}'>{g.upper()}</div>"
        )
    return f"<div class='ns-row'>{''.join(boxes)}</div>"


def nova_pill(n) -> str:
    if n is None or pd.isna(n):
        return "<span class='pill' style='background:#9e9e9e'>NOVA inconnu</span>"
    n = int(n)
    return f"<span class='pill' style='background:{NOVA_COLORS[n]}'>NOVA {n} · {NOVA_LABELS[n]}</span>"


def style_fig(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title_font_size=16, font=dict(family="sans-serif"),
    )
    return fig


def pretty_cat(c: str) -> str:
    c = re.sub(r"^[a-z]{2}:", "", c or "").replace("-", " ").strip()
    return c.capitalize() if c else "—"


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", " ")


# ----------------------------------------------------------------------------
# Chargement des données et des modèles
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
    # colonne de recherche précalculée (évite de recalculer à chaque frappe)
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
    """Retourne (tokenizer, model, erreur)."""
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
    """Probabilité que le produit soit ultra-transformé (NOVA 4) d'après ses ingrédients."""
    import torch

    tok, model, _ = load_bert()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().tolist()
    return float(probs[1])  # label 1 = NOVA 4 (cf. 04_train_bert.py)


try:
    df_all = load_products()
except Exception as e:
    st.error(
        f"Données gold introuvables dans `{GOLD_PRODUCTS}` : {e}\n\n"
        "Lance d'abord `python src/02_etl_spark.py`."
    )
    st.stop()


@st.cache_resource
def top_categories(n: int = 40) -> list[str]:
    vc = df_all["main_category"].replace("", np.nan).dropna().value_counts().head(n)
    return vc.index.tolist()


@st.cache_data(max_entries=1024, show_spinner=False)
def category_median(cat: str, col: str) -> float:
    return float(df_all.loc[df_all["main_category"] == cat, col].median())


# ----------------------------------------------------------------------------
# Sidebar : filtres
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 🥗 SoGood")
st.sidebar.caption("Open Food Facts · Projet IPSSI")
st.sidebar.divider()
st.sidebar.markdown("#### Filtres")

sel_cats = st.sidebar.multiselect(
    "Catégories (vide = toutes)", top_categories(), format_func=pretty_cat
)
nova_sel = st.sidebar.multiselect(
    "Groupes NOVA", [1, 2, 3, 4], default=[1, 2, 3, 4],
    format_func=lambda n: f"{n} · {NOVA_LABELS[n]}",
)
incl_nova_na = st.sidebar.checkbox("Inclure les produits sans NOVA", value=True)
min_add = st.sidebar.slider("Nombre minimum d'additifs", 0, 15, 0)
only_food = st.sidebar.checkbox("Alimentaire uniquement", value=True)

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

st.sidebar.divider()
st.sidebar.metric("Produits filtrés", fmt_int(len(sub)))
st.sidebar.progress(len(sub) / max(len(df_all), 1), text=f"{len(sub) / len(df_all):.1%} du dataset")

st.sidebar.divider()
st.sidebar.markdown("#### Modèles")
st.sidebar.markdown(("✅" if load_rf() else "❌") + " Random Forest (Nutri-Score)")
st.sidebar.markdown(("✅" if (BERT_DIR / "config.json").exists() else "❌") + " DistilBERT (NOVA 4)")

# ----------------------------------------------------------------------------
# En-tête
# ----------------------------------------------------------------------------
st.markdown(
    "<div class='hero'><h1>🥗 SoGood</h1>"
    "<p>Cartographier la qualité nutritionnelle des produits de supermarché "
    "à partir d'Open Food Facts.</p></div>",
    unsafe_allow_html=True,
)

if sub.empty:
    st.warning("Aucun produit ne correspond aux filtres. Élargis ta sélection dans la barre latérale.")
    st.stop()

known = sub[sub["nutriscore_grade"].isin(GRADES)]

tab_overview, tab_explore, tab_predict, tab_brands = st.tabs(
    ["📊 Vue d'ensemble", "🔎 Explorer", "🤖 Prédire", "🏷️ Marques"]
)

# ============================== VUE D'ENSEMBLE ==============================
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Produits", fmt_int(len(sub)))
    de_share = known["nutriscore_grade"].isin(["d", "e"]).mean() * 100 if len(known) else 0
    c2.metric("Part Nutri-Score D/E", f"{de_share:.1f} %",
              help=f"Calculée sur les {fmt_int(len(known))} produits qui ont un Nutri-Score.")
    c3.metric("Additifs (moyenne)", f"{sub['additives_count'].fillna(0).mean():.1f}")
    c4.metric("Score de controverse (médiane)", f"{sub['controversy_score'].median():.1f}")
    st.caption(f"Nutri-Score renseigné pour {len(known) / len(sub):.0%} des produits filtrés.")

    col_a, col_b = st.columns(2)
    with col_a:
        counts = known["nutriscore_grade"].value_counts().reindex(GRADES).fillna(0)
        fig = go.Figure(go.Bar(
            x=[g.upper() for g in GRADES], y=counts.values,
            marker_color=[NUTRI_COLORS[g] for g in GRADES],
            hovertemplate="Nutri-Score %{x}<br>%{y:,.0f} produits<extra></extra>"))
        fig.update_layout(title="Distribution du Nutri-Score")
        st.plotly_chart(style_fig(fig), width="stretch", key="ov_nutri")
    with col_b:
        nova_counts = sub["nova_group"].dropna().astype(int).value_counts().reindex([1, 2, 3, 4]).fillna(0)
        fig = go.Figure(go.Bar(
            x=[f"{n} · {NOVA_LABELS[n]}" for n in nova_counts.index], y=nova_counts.values,
            marker_color=[NOVA_COLORS[n] for n in nova_counts.index],
            hovertemplate="%{x}<br>%{y:,.0f} produits<extra></extra>"))
        fig.update_layout(title="Distribution NOVA (niveau de transformation)")
        st.plotly_chart(style_fig(fig), width="stretch", key="ov_nova")

    col_c, col_d = st.columns(2)
    with col_c:
        both = sub[sub["nutriscore_grade"].isin(GRADES) & sub["nova_group"].notna()]
        if len(both):
            ct = pd.crosstab(both["nova_group"].astype(int), both["nutriscore_grade"])
            ct = ct.reindex(index=[1, 2, 3, 4], columns=GRADES, fill_value=0)
            pct = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
            pct.index = [f"NOVA {i}" for i in pct.index]
            pct.columns = [g.upper() for g in pct.columns]
            fig = px.imshow(pct, text_auto=".0%", aspect="auto",
                            color_continuous_scale="YlGn",
                            labels=dict(x="Nutri-Score", y="", color="Part"))
            fig.update_layout(title="Répartition du Nutri-Score par groupe NOVA",
                              coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig, 400), width="stretch", key="ov_heat")
    with col_d:
        scat = known.dropna(subset=["sugars_100g", "salt_100g"])
        if len(scat) > 3000:
            scat = scat.sample(3000, random_state=0)
        if len(scat):
            scat = scat.assign(grade=scat["nutriscore_grade"].str.upper())
            fig = px.scatter(
                scat, x="sugars_100g", y="salt_100g", color="grade",
                category_orders={"grade": [g.upper() for g in GRADES]},
                color_discrete_map={g.upper(): c for g, c in NUTRI_COLORS.items()},
                hover_data=["product_name", "brands"], opacity=0.55,
                labels={"sugars_100g": "Sucres (g/100g)", "salt_100g": "Sel (g/100g)", "grade": "Nutri-Score"},
                title="Sucres vs sel (échantillon de 3 000)")
            fig.update_layout(xaxis_range=[0, 80], yaxis_range=[0, 6])
            st.plotly_chart(style_fig(fig, 400), width="stretch", key="ov_scatter")

# ================================ EXPLORER ==================================
def product_card(row: pd.Series) -> None:
    name = row["product_name"] or "Produit sans nom"
    brand = row["brands"].split(",")[0].strip() if row["brands"] else "Marque inconnue"
    cat = row["main_category"]

    st.markdown(f"<h3 style='margin-bottom:0'>{html.escape(name)}</h3>"
                f"<div class='muted'>{html.escape(brand)} · {html.escape(pretty_cat(cat))} · "
                f"code {html.escape(str(row['code']))}</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 2])
    with left:
        st.markdown(nutri_badge(row["nutriscore_grade"]), unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>" + nova_pill(row["nova_group"]),
                    unsafe_allow_html=True)
    with right:
        metrics = [
            ("Énergie", "energy_kcal_100g", 0, " kcal", True),
            ("Sucres", "sugars_100g", 1, " g", True),
            ("Sel", "salt_100g", 2, " g", True),
            ("Graisses", "fat_100g", 1, " g", True),
            ("Graisses sat.", "saturated-fat_100g", 1, " g", True),
            ("Protéines", "proteins_100g", 1, " g", False),
            ("Fibres", "fiber_100g", 1, " g", False),
            ("Additifs", "additives_count", 0, "", True),
        ]
        for chunk in (metrics[:4], metrics[4:]):
            cols = st.columns(4)
            for col, (label, key, nd, unit, lower_is_better) in zip(cols, chunk):
                v = row[key]
                if pd.isna(v):
                    col.metric(label, "–")
                    continue
                delta = None
                if key != "additives_count" and cat:
                    med = category_median(cat, key)
                    if not np.isnan(med):
                        delta = f"{v - med:+.{nd}f} vs médiane"
                col.metric(label, f"{v:.{nd}f}{unit}", delta=delta,
                           delta_color="inverse" if lower_is_better else "normal",
                           help="Pour 100 g · écart à la médiane de la même catégorie" if delta else None)


with tab_explore:
    f1, f2 = st.columns([3, 2])
    q = f1.text_input("Rechercher un produit ou une marque", placeholder="Nutella, chips, yaourt bio…")
    grade_filter = f2.multiselect("Nutri-Score", [g.upper() for g in GRADES], placeholder="Tous")

    s = sub
    if q:
        s = s[s["search"].str.contains(q.lower().strip(), regex=False, na=False)]
    if grade_filter:
        s = s[s["nutriscore_grade"].str.upper().isin(grade_filter)]

    st.caption(f"{fmt_int(len(s))} résultats · 300 premiers affichés · clique sur une ligne pour voir la fiche produit")
    view = s.head(300).reset_index(drop=True)
    display = view.assign(nutriscore_grade=view["nutriscore_grade"].str.upper())
    event = st.dataframe(
        display,
        width="stretch", height=380, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="explore_table",
        column_order=["product_name", "brands", "nutriscore_grade", "nova_group",
                      "additives_count", "sugars_100g", "salt_100g", "controversy_score"],
        column_config={
            "product_name": st.column_config.TextColumn("Produit", width="large"),
            "brands": st.column_config.TextColumn("Marque"),
            "nutriscore_grade": st.column_config.TextColumn("Nutri", width="small"),
            "nova_group": st.column_config.NumberColumn("NOVA", format="%d", width="small"),
            "additives_count": st.column_config.NumberColumn("Additifs", format="%d", width="small"),
            "sugars_100g": st.column_config.ProgressColumn("Sucres (g)", min_value=0, max_value=60, format="%.1f"),
            "salt_100g": st.column_config.ProgressColumn("Sel (g)", min_value=0, max_value=3, format="%.2f"),
            "controversy_score": st.column_config.NumberColumn("Controverse", format="%.1f"),
        },
    )
    picked = event.selection.rows if event and event.selection else []
    if picked and picked[0] < len(view):
        st.divider()
        product_card(view.iloc[picked[0]])

# ================================= PRÉDIRE ==================================
with tab_predict:
    sub_rf, sub_bert = st.tabs(["Nutri-Score estimé (Random Forest)", "Ultra-transformé ? (DistilBERT)"])

    # ---- Random Forest ----
    with sub_rf:
        st.markdown("Le **Random Forest** estime le Nutri-Score (A→E) à partir des valeurs nutritionnelles "
                    "pour 100 g, du nombre d'additifs et du groupe NOVA.")
        if load_rf() is None:
            st.error("Modèle introuvable. Lance `python src/03_train_rf.py`.")
        else:
            with st.form("rf_form"):
                a, b, c, d = st.columns(4)
                f_energy = a.number_input("Énergie (kcal)", 0.0, 1000.0, 520.0)
                f_sug = b.number_input("Sucres (g)", 0.0, 100.0, 48.0)
                f_salt = c.number_input("Sel (g)", 0.0, 100.0, 1.2)
                f_add = d.number_input("Additifs", 0, 40, 4, step=1)
                a, b, c, d = st.columns(4)
                f_fat = a.number_input("Graisses (g)", 0.0, 100.0, 30.0)
                f_sat = b.number_input("Graisses sat. (g)", 0.0, 100.0, 18.0)
                f_prot = c.number_input("Protéines (g)", 0.0, 100.0, 6.0)
                f_fib = d.number_input("Fibres (g)", 0.0, 100.0, 3.0)
                f_nova = st.select_slider("Groupe NOVA", [1, 2, 3, 4], value=4,
                                          format_func=lambda n: f"{n} · {NOVA_LABELS[n]}")
                go_rf = st.form_submit_button("Prédire le Nutri-Score", type="primary", width="stretch")

            if go_rf:
                res = predict_rf({
                    "energy_kcal_100g": f_energy, "sugars_100g": f_sug, "salt_100g": f_salt,
                    "fat_100g": f_fat, "saturated-fat_100g": f_sat, "proteins_100g": f_prot,
                    "fiber_100g": f_fib, "additives_count": f_add, "nova_group": f_nova,
                })
                left, right = st.columns([1, 2])
                with left:
                    st.markdown("<div style='height:1rem'></div>" + nutri_badge(res["grade"], 1.6),
                                unsafe_allow_html=True)
                    st.caption(f"Confiance : {res['probs'][res['grade']]:.0%}")
                with right:
                    fig = go.Figure(go.Bar(
                        x=[g.upper() for g in GRADES], y=[res["probs"][g] for g in GRADES],
                        marker_color=[NUTRI_COLORS[g] for g in GRADES],
                        hovertemplate="%{x} : %{y:.1%}<extra></extra>"))
                    fig.update_layout(title="Probabilité par classe", yaxis=dict(tickformat=".0%"))
                    st.plotly_chart(style_fig(fig, 300), width="stretch", key="rf_probs")

    # ---- BERT ----
    with sub_bert:
        st.markdown("**DistilBERT** lit la liste d'ingrédients et estime la probabilité que le produit soit "
                    "**ultra-transformé (NOVA 4)**. Ce modèle ne prédit pas le Nutri-Score.")
        tok, bert, err = load_bert()
        if bert is None:
            st.error(f"{err}\n\nLance `python src/04_train_bert.py`.")
        else:
            ingredients = st.text_area(
                "Liste d'ingrédients", height=140,
                value="Sucre, huile de palme, noisettes 13%, cacao maigre 7,4%, lait écrémé en poudre 6,6%, "
                      "lactosérum en poudre, émulsifiant : lécithines, vanilline.")
            if st.button("Analyser les ingrédients", type="primary", width="stretch"):
                if not ingredients.strip():
                    st.warning("Saisis une liste d'ingrédients.")
                else:
                    with st.spinner("Analyse en cours…"):
                        p = predict_bert(ingredients)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=p * 100, number=dict(suffix=" %"),
                        title=dict(text="Probabilité d'être ultra-transformé"),
                        gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#333"),
                                   steps=[dict(range=[0, 33], color="#9ccc65"),
                                          dict(range=[33, 66], color="#ffb74d"),
                                          dict(range=[66, 100], color="#e57373")])))
                    st.plotly_chart(style_fig(fig, 300), width="stretch", key="bert_gauge")
                    if p >= 0.5:
                        st.markdown(nova_pill(4), unsafe_allow_html=True)
                    else:
                        st.success("Plutôt non ultra-transformé (NOVA 1 à 3)")

# ================================= MARQUES ==================================
with tab_brands:
    st.caption("Calculé sur les produits filtrés. Le score de controverse augmente avec le nombre d'additifs, "
               "un Nutri-Score D/E et un NOVA 4.")
    o1, o2, o3 = st.columns([2, 2, 1])
    order = o1.radio("Classement", ["Plus controversées", "Moins controversées"], horizontal=True)
    min_n = o2.slider("Produits minimum par marque", 5, 100, 10)
    top_n = o3.selectbox("Top", [10, 20, 30], index=1)

    brand = sub["brands"].str.split(",").str[0].str.strip().str.lower()
    grouped = (sub.groupby(brand)
               .agg(n=("code", "count"), controverse=("controversy_score", "mean"),
                    additifs=("additives_count", "mean"), sucres=("sugars_100g", "mean"),
                    sel=("salt_100g", "mean"))
               .reset_index().rename(columns={"brands": "marque"}))
    grouped = grouped[(grouped["marque"] != "") & (grouped["n"] >= min_n)]
    grouped = grouped.sort_values("controverse", ascending=(order == "Moins controversées")).head(top_n)

    if grouped.empty:
        st.info("Aucune marque ne remplit ce seuil. Baisse le minimum de produits.")
    else:
        plot_df = grouped.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=plot_df["controverse"], y=plot_df["marque"], orientation="h",
            marker=dict(color=plot_df["controverse"], colorscale="RdYlGn_r"),
            customdata=plot_df[["n"]],
            hovertemplate="<b>%{y}</b><br>Controverse %{x:.2f}<br>%{customdata[0]} produits<extra></extra>"))
        fig.update_layout(title=f"Top {len(grouped)} marques · {order.lower()}", margin=dict(l=160))
        st.plotly_chart(style_fig(fig, max(380, 26 * len(grouped) + 90)), width="stretch", key="brands_bar")
        st.dataframe(
            grouped, width="stretch", hide_index=True,
            column_config={
                "marque": st.column_config.TextColumn("Marque"),
                "n": st.column_config.NumberColumn("Produits", format="%d"),
                "controverse": st.column_config.NumberColumn("Controverse", format="%.2f"),
                "additifs": st.column_config.NumberColumn("Additifs moy.", format="%.2f"),
                "sucres": st.column_config.NumberColumn("Sucres moy. (g)", format="%.1f"),
                "sel": st.column_config.NumberColumn("Sel moy. (g)", format="%.2f"),
            },
        )

st.divider()
st.caption("SoGood · Données Open Food Facts (licence ODbL) · Pipeline Spark + Random Forest + DistilBERT")