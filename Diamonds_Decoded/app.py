# =============================================================================
# 💎 DIAMONDS DECODED — STREAMLIT DASHBOARD
# Price Forecasting & Customer Segmentation
# Across Natural & Lab-Grown Diamonds
# =============================================================================
# Author  : Lindiwe Songelwa
# Version : 1.0 | 2026
# Deploy  : streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title  = "Diamonds Decoded",
    page_icon   = "💎",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# =============================================================================
# GLOBAL STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0F1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1E2130;
        border: 1px solid #2C3E7A;
        border-radius: 8px;
        padding: 12px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1D2E;
    }

    /* Headers */
    h1 { color: #E8D5A3; }
    h2 { color: #C9A84C; }
    h3 { color: #FFFFFF; }

    /* Info boxes */
    .stAlert { border-radius: 8px; }

    /* Custom badge */
    .badge-natural {
        background-color: #2C3E7A;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: bold;
    }
    .badge-lab {
        background-color: #1E8C5A;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: bold;
    }
    .archetype-card {
        background-color: #1E2130;
        border-left: 4px solid #C9A84C;
        border-radius: 6px;
        padding: 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS & DATA
# =============================================================================

@st.cache_resource
def load_models():
    """Load all trained models and scaler. Cached for performance."""
    base = os.path.dirname(__file__)
    rf     = joblib.load(os.path.join(base, 'models', 'rf_model.joblib'))
    xgb    = joblib.load(os.path.join(base, 'models', 'xgb_combined.joblib'))
    scaler = joblib.load(os.path.join(base, 'models', 'scaler.joblib'))
    return rf, xgb, scaler


@st.cache_data
def load_data():
    """Load df_sample for dashboard visualisations. Cached for performance."""
    base = os.path.dirname(__file__)
    df   = pd.read_csv(os.path.join(base, 'data', 'df_sample.csv'))
    return df


# Load with error handling
try:
    rf_model, xgb_model, scaler = load_models()
    df = load_data()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error   = str(e)

# =============================================================================
# CONSTANTS
# =============================================================================

CUT_MAP     = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
COLOR_MAP   = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
CLARITY_MAP = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3,
               'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

MODEL_FEATURES = [
    'carat', 'cut_encoded', 'color_encoded', 'clarity_encoded',
    'depth', 'table', 'cut_clarity_score',
    'log_price_per_carat', 'days_on_market', 'diamond_type_encoded'
]

ARCHETYPE_PROFILES = {
    '💍 The Traditionalist': {
        'description' : 'Heritage-driven buyer. Prioritises natural provenance, GIA certification, and luxury retail experience. Longer consideration period reflects deliberate, once-in-a-lifetime purchase intent.',
        'diamond_type': 'Natural',
        'certification': 'GIA',
        'retailer'    : 'Luxury / Brick-and-Mortar',
        'budget'      : 'Premium–Luxury ($5,000–$18,000+)',
        'colour'      : '#2C3E7A',
    },
    '🌱 The Ethical Buyer': {
        'description' : 'Values-driven, digitally native buyer. Lab-grown preference reflects environmental and ethical consciousness. Fast decision cycle via online retail.',
        'diamond_type': 'Lab-Grown',
        'certification': 'IGI',
        'retailer'    : 'Online',
        'budget'      : 'Mid ($1,000–$5,000)',
        'colour'      : '#1E8C5A',
    },
    '💸 The Value Hunter': {
        'description' : 'Budget-optimised buyer maximising carat weight per dollar. Lab-grown enables larger stones within fixed budgets. Exclusively online channel.',
        'diamond_type': 'Lab-Grown',
        'certification': 'IGI',
        'retailer'    : 'Online',
        'budget'      : 'Budget–Mid (under $3,000)',
        'colour'      : '#E74C3C',
    },
    '👑 The Luxury Collector': {
        'description' : 'Unconstrained budget with exceptional quality standards. Natural diamond preference is non-negotiable. Least susceptible to lab-grown substitution.',
        'diamond_type': 'Natural',
        'certification': 'GIA',
        'retailer'    : 'Luxury',
        'budget'      : 'Luxury ($12,000+)',
        'colour'      : '#F39C12',
    },
    '💡 The Pragmatist': {
        'description' : 'Open to both diamond types. Purchasing decision is driven purely by price-per-carat value ratio. Most sensitive to the evolving natural vs lab-grown price gap.',
        'diamond_type': 'Natural or Lab-Grown',
        'certification': 'GIA or IGI',
        'retailer'    : 'Mid-tier / Online',
        'budget'      : 'Mid ($2,000–$7,000)',
        'colour'      : '#8E44AD',
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_feature_vector(carat, cut, color, clarity, depth, table,
                          diamond_type, days_on_market=30):
    """
    Builds a single-row feature DataFrame matching the model's training schema.
    All encoding and derived feature logic mirrors Phase 2 exactly.
    """
    cut_enc     = CUT_MAP[cut]
    color_enc   = COLOR_MAP[color]
    clarity_enc = CLARITY_MAP[clarity]

    # Derived features — match Phase 2 engineering exactly
    cut_clarity_score    = (cut_enc / 4) * 5 + (clarity_enc / 7) * 5
    price_per_carat_est  = 4000 if diamond_type == 'Natural' else 900
    log_price_per_carat  = np.log1p(price_per_carat_est)
    diamond_type_encoded = 1 if diamond_type == 'Natural' else 0

    row = pd.DataFrame([{
        'carat'               : carat,
        'cut_encoded'         : cut_enc,
        'color_encoded'       : color_enc,
        'clarity_encoded'     : clarity_enc,
        'depth'               : depth,
        'table'               : table,
        'cut_clarity_score'   : cut_clarity_score,
        'log_price_per_carat' : log_price_per_carat,
        'days_on_market'      : days_on_market,
        'diamond_type_encoded': diamond_type_encoded,
    }])

    return row


def predict_price(feature_row, model):
    """Scale features and return predicted price in USD."""
    X_scaled  = scaler.transform(feature_row)
    log_price = model.predict(X_scaled)[0]
    price_usd = np.expm1(log_price)
    return round(price_usd, 2)


def get_value_tier(price):
    if price < 1000:
        return 'Budget', '#95A5A6'
    elif price < 5000:
        return 'Mid', '#3498DB'
    elif price < 12000:
        return 'Premium', '#9B59B6'
    else:
        return 'Luxury', '#F39C12'


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

with st.sidebar:
    st.markdown("## 💎 Diamonds Decoded")
    st.markdown("*Price Forecasting & Customer Segmentation*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        options=[
            "💎 Price Predictor",
            "👥 Customer Profiler",
            "📊 Market Intelligence",
            "🌍 SA & Global Context",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model Status**")
    if models_loaded:
        st.success("✅ Models loaded")
        st.caption("Random Forest | XGBoost | StandardScaler")
    else:
        st.error("❌ Models not found")
        st.caption(f"Error: {model_error}")

    st.markdown("---")
    st.caption("Author: Lindiwe Songelwa")
    st.caption("Dataset: Kaggle — shivam2503")
    st.caption("Version 1.0 | 2026")


# =============================================================================
# PAGE 1 — PRICE PREDICTOR
# =============================================================================

if page == "💎 Price Predictor":

    st.title("💎 Diamond Price Predictor")
    st.markdown(
        "Enter diamond attributes below to receive an instant retail price "
        "estimate from the trained ensemble models."
    )
    st.markdown("---")

    if not models_loaded:
        st.error(f"Models could not be loaded. Please check the models/ directory.\n\n{model_error}")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚖️ Weight & Type")
        carat        = st.slider("Carat Weight", 0.20, 5.00, 1.00, 0.01)
        diamond_type = st.selectbox("Diamond Type", ["Natural", "Lab-Grown"])
        days_market  = st.slider("Days on Market", 1, 180, 30)

    with col2:
        st.subheader("🔬 The 4Cs")
        cut     = st.selectbox("Cut Grade",
                               ["Ideal", "Premium", "Very Good", "Good", "Fair"])
        color   = st.selectbox("Colour Grade",
                               ["D", "E", "F", "G", "H", "I", "J"])
        clarity = st.selectbox("Clarity Grade",
                               ["IF", "VVS1", "VVS2", "VS1", "VS2",
                                "SI1", "SI2", "I1"])

    with col3:
        st.subheader("📐 Proportions")
        depth = st.slider("Depth %", 50.0, 75.0, 61.5, 0.1)
        table = st.slider("Table %", 50.0, 75.0, 57.0, 0.1)
        model_choice = st.selectbox(
            "Forecasting Model",
            ["Random Forest (R²=0.9997)", "XGBoost (R²=0.9997)"]
        )

    st.markdown("---")

    if st.button("🔮 Predict Price", use_container_width=True, type="primary"):

        features = build_feature_vector(
            carat, cut, color, clarity, depth, table,
            diamond_type, days_market
        )

        model = rf_model if "Random Forest" in model_choice else xgb_model

        with st.spinner("Computing prediction..."):
            predicted_price = predict_price(features, model)
            tier, tier_colour = get_value_tier(predicted_price)

            # Revenue estimate
            margin = 0.40 if diamond_type == "Natural" else 0.73
            revenue_est = predicted_price * (1 - margin)

        st.markdown("### 📊 Prediction Results")
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("💰 Predicted Price (USD)", f"${predicted_price:,.0f}")
        m2.metric("📈 Revenue Est. (USD)",
                  f"${revenue_est:,.0f}",
                  f"After {margin*100:.0f}% retail margin")
        m3.metric("💎 Price Per Carat", f"${predicted_price/carat:,.0f}")
        m4.metric("🏷️ Value Tier", tier)

        # Confidence context
        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.info(
                f"**Model:** {model_choice.split(' (')[0]}\n\n"
                f"**Diamond Type:** {diamond_type}\n\n"
                f"**Certification Recommendation:** "
                f"{'GIA — gold standard for natural stones' if diamond_type == 'Natural' else 'IGI — dominant for lab-grown; GIA commands 30–50% premium'}\n\n"
                f"**Retailer Channel:** "
                f"{'Luxury or Brick-and-Mortar' if tier in ['Premium','Luxury'] else 'Online or Brick-and-Mortar'}"
            )

        with col_b:
            # Carat vs price gauge
            price_range = [500, 2500, 5000, 12000, 20000]
            labels       = ['Budget', 'Mid', 'Premium', 'Luxury', 'Ultra']
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = predicted_price,
                title = {'text': "Price Tier Position"},
                gauge = {
                    'axis'  : {'range': [0, 20000]},
                    'bar'   : {'color': tier_colour},
                    'steps' : [
                        {'range': [0,     1000],  'color': '#2C2C2C'},
                        {'range': [1000,  5000],  'color': '#1A3A5C'},
                        {'range': [5000,  12000], 'color': '#3D1A5C'},
                        {'range': [12000, 20000], 'color': '#5C3D00'},
                    ],
                    'threshold': {
                        'line' : {'color': "white", 'width': 3},
                        'value': predicted_price
                    }
                },
                number={'prefix': "$", 'valueformat': ",.0f"}
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20),
                              paper_bgcolor='rgba(0,0,0,0)',
                              font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "⚠️ Disclaimer: Price predictions are estimates based on a "
            "15,000-row stratified sample of the Kaggle diamonds dataset "
            "(Shivam2503, Kaggle). Predictions do not constitute financial "
            "or commercial advice."
        )


# =============================================================================
# PAGE 2 — CUSTOMER PROFILER
# =============================================================================

elif page == "👥 Customer Profiler":

    st.title("👥 Customer Profiler")
    st.markdown(
        "Answer the questions below to identify which buyer archetype best "
        "matches a customer's purchase profile — and receive a tailored "
        "product and channel recommendation."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🛒 Purchase Preferences")
        budget = st.selectbox(
            "Budget Range",
            ["Under $1,000", "$1,000–$3,000", "$3,000–$5,000",
             "$5,000–$12,000", "$12,000+"]
        )
        type_pref = st.selectbox(
            "Diamond Type Preference",
            ["Natural only", "Lab-grown only",
             "Open to either — price drives decision",
             "Open to either — ethics drive decision"]
        )
        channel = st.selectbox(
            "Preferred Shopping Channel",
            ["Online", "Brick-and-Mortar", "Luxury boutique / jeweller"]
        )

    with col2:
        st.subheader("💎 Quality Priorities")
        quality_focus = st.selectbox(
            "Top Priority When Choosing a Diamond",
            ["Maximise carat size for budget",
             "Exceptional cut and clarity",
             "Ethical sourcing / sustainability",
             "Heritage, provenance, and rarity",
             "Best price-per-carat ratio"]
        )
        certification_pref = st.selectbox(
            "Certification Preference",
            ["GIA (I trust the gold standard)",
             "IGI (reliable and cost-effective)",
             "No preference"]
        )
        occasion = st.selectbox(
            "Purchase Occasion",
            ["Engagement ring", "Anniversary gift",
             "Investment / collection", "Self-purchase", "Other"]
        )

    st.markdown("---")

    if st.button("🔍 Identify Archetype", use_container_width=True, type="primary"):

        # Scoring logic — maps responses to archetype scores
        scores = {a: 0 for a in ARCHETYPE_PROFILES}

        # Budget
        budget_map = {
            "Under $1,000"    : "💸 The Value Hunter",
            "$1,000–$3,000"   : "💸 The Value Hunter",
            "$3,000–$5,000"   : "🌱 The Ethical Buyer",
            "$5,000–$12,000"  : "💍 The Traditionalist",
            "$12,000+"        : "👑 The Luxury Collector",
        }
        scores[budget_map[budget]] += 3

        # Type preference
        type_map = {
            "Natural only"                           : ["💍 The Traditionalist", "👑 The Luxury Collector"],
            "Lab-grown only"                         : ["🌱 The Ethical Buyer", "💸 The Value Hunter"],
            "Open to either — price drives decision" : ["💡 The Pragmatist"],
            "Open to either — ethics drive decision" : ["🌱 The Ethical Buyer", "💡 The Pragmatist"],
        }
        for a in type_map[type_pref]:
            scores[a] += 2

        # Quality focus
        quality_map = {
            "Maximise carat size for budget"   : "💸 The Value Hunter",
            "Exceptional cut and clarity"      : "👑 The Luxury Collector",
            "Ethical sourcing / sustainability": "🌱 The Ethical Buyer",
            "Heritage, provenance, and rarity" : "💍 The Traditionalist",
            "Best price-per-carat ratio"       : "💡 The Pragmatist",
        }
        scores[quality_map[quality_focus]] += 3

        # Channel
        channel_map = {
            "Online"                          : ["🌱 The Ethical Buyer", "💸 The Value Hunter"],
            "Brick-and-Mortar"                : ["💍 The Traditionalist", "💡 The Pragmatist"],
            "Luxury boutique / jeweller"      : ["👑 The Luxury Collector", "💍 The Traditionalist"],
        }
        for a in channel_map[channel]:
            scores[a] += 1

        # Certification
        if "GIA" in certification_pref:
            scores["💍 The Traditionalist"]  += 1
            scores["👑 The Luxury Collector"] += 1
        elif "IGI" in certification_pref:
            scores["🌱 The Ethical Buyer"]   += 1
            scores["💸 The Value Hunter"]    += 1

        # Identify top archetype
        top_archetype = max(scores, key=scores.get)
        profile       = ARCHETYPE_PROFILES[top_archetype]

        st.markdown(f"### {top_archetype}")
        st.markdown(
            f"<div class='archetype-card'>{profile['description']}</div>",
            unsafe_allow_html=True
        )

        st.markdown("---")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("💎 Diamond Type",  profile['diamond_type'])
        r2.metric("🏅 Certification", profile['certification'])
        r3.metric("🏪 Retailer",      profile['retailer'])
        r4.metric("💰 Budget Range",  profile['budget'])

        # Score breakdown chart
        st.markdown("#### Archetype Match Scores")
        score_df = pd.DataFrame({
            'Archetype': list(scores.keys()),
            'Score'    : list(scores.values())
        }).sort_values('Score', ascending=True)

        fig = px.bar(
            score_df, x='Score', y='Archetype',
            orientation='h',
            color='Score',
            color_continuous_scale=['#1A2A4A', '#2C3E7A', '#C9A84C'],
            title="Archetype Match Score Breakdown"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False,
            coloraxis_showscale=False,
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 3 — MARKET INTELLIGENCE
# =============================================================================

elif page == "📊 Market Intelligence":

    st.title("📊 Market Intelligence")
    st.markdown(
        "Comparative analysis of price dynamics, revenue architecture, "
        "and market velocity across natural and lab-grown diamonds."
    )
    st.markdown("---")

    # --- Key metrics row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Natural Mean Price",
              f"${df[df['diamond_type']=='natural']['price'].mean():,.0f}")
    m2.metric("Lab-Grown Mean Price",
              f"${df[df['diamond_type']=='lab_grown']['price'].mean():,.0f}")
    m3.metric("Natural Mean Rev/ct",
              f"${df[df['diamond_type']=='natural']['revenue_per_carat'].mean():,.0f}")
    m4.metric("Lab-Grown Mean Rev/ct",
              f"${df[df['diamond_type']=='lab_grown']['revenue_per_carat'].mean():,.0f}")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Price Distributions",
        "📉 Revenue Architecture",
        "⚡ Market Velocity",
        "🏆 Feature Importance"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df, x='price', color='diamond_type',
                barmode='overlay', nbins=60,
                color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
                title="Retail Price Distribution — Natural vs Lab-Grown",
                labels={'price': 'Price (USD)', 'diamond_type': 'Type'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig = px.box(
                df, x='diamond_type', y='price',
                color='diamond_type',
                color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
                title="Price Spread — Natural vs Lab-Grown",
                labels={'price': 'Price (USD)', 'diamond_type': 'Type'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white',
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # Carat vs Price scatter
        sample_scatter = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(
            sample_scatter, x='carat', y='price',
            color='diamond_type',
            color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
            opacity=0.4, size_max=4,
            title="Carat vs Price — Natural vs Lab-Grown",
            labels={'carat': 'Carat Weight', 'price': 'Price (USD)',
                    'diamond_type': 'Type'}
        )
        for magic in [0.5, 1.0, 1.5, 2.0]:
            fig.add_vline(x=magic, line_dash="dot",
                          line_color="#E74C3C", opacity=0.5,
                          annotation_text=f"{magic}ct",
                          annotation_font_color="#E74C3C")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dotted lines indicate commercially preferred 'magic size' carat thresholds.")

    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            # Revenue per carat — natural (independent scale)
            nat_data = df[df['diamond_type'] == 'natural']['revenue_per_carat']
            fig = px.histogram(
                nat_data, nbins=60,
                title="Revenue Per Carat — Natural Diamonds",
                labels={'value': 'Revenue Per Carat (USD)'},
                color_discrete_sequence=['#2C3E7A']
            )
            fig.add_vline(x=nat_data.mean(), line_dash="dash",
                          line_color="#E74C3C",
                          annotation_text=f"Mean: ${nat_data.mean():,.0f}",
                          annotation_font_color="#E74C3C")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Revenue per carat — lab-grown (independent scale)
            lab_data = df[df['diamond_type'] == 'lab_grown']['revenue_per_carat']
            fig = px.histogram(
                lab_data, nbins=60,
                title="Revenue Per Carat — Lab-Grown Diamonds",
                labels={'value': 'Revenue Per Carat (USD)'},
                color_discrete_sequence=['#1E8C5A']
            )
            fig.add_vline(x=lab_data.mean(), line_dash="dash",
                          line_color="#E74C3C",
                          annotation_text=f"Mean: ${lab_data.mean():,.0f}",
                          annotation_font_color="#E74C3C")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "📊 **Revenue Architecture Gap:** Natural diamonds retain "
            "significantly more upstream revenue per carat (40% retailer "
            "margin) compared to lab-grown equivalents (73% retailer margin). "
            "Volume growth in lab-grown retail does not translate into "
            "proportional upstream revenue recovery for producers. "
            "*(Edahn Golan, Q3 2025)*"
        )

        # Revenue by value tier
        gap = df.groupby(['value_tier', 'diamond_type'])[
            ['price_per_carat', 'revenue_per_carat']
        ].mean().reset_index()
        tier_order = ['Budget', 'Mid', 'Premium', 'Luxury']
        gap['value_tier'] = pd.Categorical(
            gap['value_tier'], categories=tier_order, ordered=True
        )
        gap = gap.sort_values('value_tier')

        fig = px.line(
            gap, x='value_tier', y='revenue_per_carat',
            color='diamond_type',
            color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
            markers=True,
            title="Mean Revenue Per Carat by Value Tier",
            labels={'revenue_per_carat': 'Mean Revenue Per Carat (USD)',
                    'value_tier': 'Value Tier', 'diamond_type': 'Type'}
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)',
                          font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.violin(
                df, y='days_on_market', x='diamond_type',
                color='diamond_type',
                color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
                box=True,
                title="Days on Market — Natural vs Lab-Grown",
                labels={'days_on_market': 'Days on Market',
                        'diamond_type': 'Type'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            dom_tier = df.groupby(
                ['value_tier', 'diamond_type']
            )['days_on_market'].mean().reset_index()
            dom_tier['value_tier'] = pd.Categorical(
                dom_tier['value_tier'], categories=tier_order, ordered=True
            )
            dom_tier = dom_tier.sort_values('value_tier')

            fig = px.bar(
                dom_tier, x='value_tier', y='days_on_market',
                color='diamond_type', barmode='group',
                color_discrete_map={'natural': '#2C3E7A', 'lab_grown': '#1E8C5A'},
                title="Mean Days on Market by Value Tier",
                labels={'days_on_market': 'Mean Days on Market',
                        'value_tier': 'Value Tier', 'diamond_type': 'Type'},
                text_auto='.0f'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        # Feature importance from model
        features_display = [f.replace('_', ' ').title()
                            for f in MODEL_FEATURES]

        col_a, col_b = st.columns(2)
        with col_a:
            fi_rf_df = pd.DataFrame({
                'Feature'    : features_display,
                'Importance' : rf_model.feature_importances_
            }).sort_values('Importance')
            fig = px.bar(
                fi_rf_df,
                x='Importance', y='Feature',
                orientation='h',
                title="Random Forest Feature Importance",
                color='Importance',
                color_continuous_scale=['#1A2A4A', '#2C3E7A', '#C9A84C'],
                labels={'Feature': 'Feature', 'Importance': 'Importance'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white',
                              coloraxis_showscale=False,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fi_xgb_df = pd.DataFrame({
                'Feature'    : features_display,
                'Importance' : xgb_model.feature_importances_
            }).sort_values('Importance')
            fig = px.bar(
                fi_xgb_df,
                x='Importance', y='Feature',
                orientation='h',
                title="XGBoost Feature Importance",
                color='Importance',
                color_continuous_scale=['#0A2A1A', '#1E8C5A', '#A8E6CF'],
                labels={'Feature': 'Feature', 'Importance': 'Importance'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              plot_bgcolor='rgba(0,0,0,0)',
                              font_color='white',
                              coloraxis_showscale=False,
                              showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# PAGE 4 — SA & GLOBAL CONTEXT
# =============================================================================

elif page == "🌍 SA & Global Context":

    st.title("🌍 SA & Global Diamond Industry Context")
    st.markdown(
        "Situating the analytical findings of this study within the broader "
        "South African mining economy and global diamond industry landscape."
    )
    st.markdown("---")

    # --- Price collapse narrative ---
    st.subheader("📉 The Price Collapse Narrative (2020–2024)")

    years_nat = [2020, 2021, 2022, 2022.4, 2023, 2024]
    price_nat = [5200, 5800, 6819, 6400,   5500, 4997]
    years_lab = [2020, 2021, 2022, 2023, 2024]
    price_lab = [3410, 2800, 2100, 1500,  892]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years_nat, y=price_nat,
        mode='lines+markers',
        name='Natural Diamond (1ct avg.)',
        line=dict(color='#2C3E7A', width=3),
        marker=dict(size=9)
    ))
    fig.add_trace(go.Scatter(
        x=years_lab, y=price_lab,
        mode='lines+markers',
        name='Lab-Grown Diamond (1ct avg.)',
        line=dict(color='#1E8C5A', width=3),
        marker=dict(size=9)
    ))
    fig.add_annotation(x=2022, y=6819,
                       text="Natural peak: $6,819", showarrow=True,
                       arrowhead=2, font=dict(color='#2C3E7A', size=11))
    fig.add_annotation(x=2024, y=892,
                       text="Lab-grown Dec 2024: $892 (−74%)",
                       showarrow=True, arrowhead=2,
                       font=dict(color='#1E8C5A', size=11))
    fig.add_vrect(x0=2020, x1=2022, fillcolor="#F39C12",
                  opacity=0.06, layer="below",
                  annotation_text="COVID recovery",
                  annotation_font_color="#F39C12")
    fig.add_vrect(x0=2022, x1=2024, fillcolor="#E74C3C",
                  opacity=0.06, layer="below",
                  annotation_text="Market correction",
                  annotation_font_color="#E74C3C")
    fig.update_layout(
        title="Average Retail Price of a 1-Carat Polished Diamond (USD), 2020–2024",
        xaxis_title="Year",
        yaxis_title="Price (USD)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        yaxis=dict(tickprefix="$", tickformat=",")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Sources: Fortune (Jan 2025); Accio (2025); James Allen (May 2025); "
        "Delagem (2025); Mikado Diamonds (2024)."
    )

    st.markdown("---")

    # --- SA stakeholder cards ---
    st.subheader("🇿🇦 South African Mining Stakeholders")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**De Beers Group**")
        st.metric("2024 Production", "24.7M carats",
                  delta="-22% vs 2023", delta_color="inverse")
        st.caption(
            "Lowest output since 1995. Anglo American announced plans to "
            "divest De Beers following a near-50% value writedown. "
            "*(De Beers Group, 2025; Rapaport, 2026)*"
        )

    with col2:
        st.markdown("**Petra Diamonds**")
        st.metric("Key Operations", "Cullinan & Finsch", delta=None)
        st.caption(
            "Cullinan mine — producer of some of the world's most famous "
            "diamonds including the Cullinan Diamond (3,106ct). Under "
            "sustained pressure from natural diamond price decline."
        )

    with col3:
        st.markdown("**Natural Diamond Council**")
        st.metric("Strategy", "Origin Marketing", delta=None)
        st.caption(
            "The NDC's 'Only Natural Diamonds' campaign repositions natural "
            "stones as irreplaceable heritage objects — targeting the "
            "Traditionalist and Luxury Collector archetypes identified "
            "in Phase 4 of this study."
        )

    st.markdown("---")

    # --- Global players ---
    st.subheader("🌐 Global Industry Landscape")

    global_data = pd.DataFrame({
        'Company'    : ['De Beers', 'Pandora', 'Signet Jewelers',
                        'Blue Nile', 'James Allen', 'ALROSA'],
        'Position'   : ['Natural producer & retailer',
                        '100% lab-grown committed',
                        '~40% lab-grown shelf space',
                        'Online lab-grown leader',
                        'Online natural & lab-grown',
                        'Russian natural producer'],
        'Type'       : ['Natural', 'Lab-Grown', 'Both',
                        'Lab-Grown', 'Both', 'Natural'],
        'Impact'     : [9, 8, 7, 6, 6, 7]
    })

    fig = px.bar(
        global_data, x='Company', y='Impact',
        color='Type',
        color_discrete_map={
            'Natural'  : '#2C3E7A',
            'Lab-Grown': '#1E8C5A',
            'Both'     : '#9B59B6'
        },
        title="Key Global Diamond Industry Players — Strategic Position",
        labels={'Impact': 'Industry Influence Score (1–10)'},
        text='Position'
    )
    fig.update_traces(textposition='inside', textfont_size=9)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      font_color='white',
                      xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Research summary ---
    st.subheader("📋 Study Summary")
    st.info("""
    **Diamonds Decoded — Key Findings at a Glance**

    | Finding | Value |
    |---|---|
    | Natural diamond price decline (2022–2024) | −26.7% |
    | Lab-grown diamond price decline (2020–2024) | −73.8% |
    | Natural mean retail price (dataset) | See Market Intelligence page |
    | Lab-grown mean retail price (dataset) | See Market Intelligence page |
    | Natural retailer margin | 40% |
    | Lab-grown retailer margin | 73% |
    | Best forecasting model | Random Forest (R²=0.9997, RMSE=$136) |
    | Benchmark (Xiao, 2024) | XGBoost R²=0.9821 (full dataset) |
    | Customer archetypes identified | 5 |
    | Dominant price driver | Carat weight |

    **References:** De Beers Group (2025) | Edahn Golan (Q3 2025) |
    Fortune (Jan 2025) | BriteCo (2025) | Xiao (2024) | Rapaport (2025)
    """)

    st.caption(
        "Author: Lindiwe Songelwa | Dataset: Kaggle (shivam2503) | "
        "Version 1.0 | 2026 | "
        "Portfolio: https://lindiwe-22.github.io/Portfolio-Website/"
    )
