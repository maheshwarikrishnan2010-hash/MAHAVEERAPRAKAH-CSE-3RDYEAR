import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="EV Intelligence System", layout="wide", page_icon="⚡")

# Custom CSS for premium aesthetics
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1e38 100%);
        color: #f8fafc;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.1);
        border-bottom: 2px solid #38bdf8;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.4);
    }
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ EV Intelligence Dashboard")
st.markdown("Explore comprehensive insights into the Electric Vehicle ecosystem, from geographic distributions to advanced predictive modeling.")

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ev_path = os.path.join(base_dir, "data", "ev_dataset.csv")
    feat_path = os.path.join(base_dir, "data", "ev_features.csv")
    
    df_raw = pd.DataFrame()
    df_feat = pd.DataFrame()
    
    if os.path.exists(ev_path):
        df_raw = pd.read_csv(ev_path, low_memory=False)
    if os.path.exists(feat_path):
        df_feat = pd.read_csv(feat_path, low_memory=False)
        
    return df_raw, df_feat

with st.spinner("Loading datasets..."):
    df_raw, df_feat = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Dataset & Explorer", "📈 Visualizations", "🧠 ML Models Accuracy"])

with tab1:
    st.header("Dataset Overview")
    
    if not df_feat.empty:
        st.markdown("### Engineered Features Preview")
        st.dataframe(df_feat.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Data Dictionary:**
            - `county`, `city`, `state`: Geographic location
            - `make`, `model`, `model_year`: Vehicle identification
            - `ev_type`: BEV (Battery EV) or PHEV (Plug-in Hybrid EV)
            - `electric_range`: Rated battery range in miles
            - `vehicle_age`: Age of the vehicle in years
            - `is_high_range`: Target flag indicating if range > median
            """)
        with col2:
            st.info("This dataset has been cleaned, missing values handled, and features engineered for Machine Learning models.")
            csv_feat = df_feat.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Engineered Features (CSV)", csv_feat, "ev_features.csv", "text/csv")
            
            if not df_raw.empty:
                st.download_button("⬇️ Download Raw Dataset Sample (CSV)", df_raw.head(5000).to_csv(index=False).encode('utf-8'), "ev_raw_sample.csv", "text/csv")
    else:
        st.error("No dataset found. Please run the pipeline to gather data.")

with tab2:
    st.header("Analytics & Insights")
    if not df_raw.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 EV Makes")
            make_counts = df_raw['make'].value_counts().head(10)
            st.bar_chart(make_counts)
            
        with col2:
            st.subheader("EV Adoption Over Time")
            # Filter unrealistic future years if any
            year_counts = df_raw[df_raw['model_year'] <= 2025]['model_year'].value_counts().sort_index()
            st.line_chart(year_counts)
            
        st.subheader("Electric Range Distribution by Top Makes")
        if 'electric_range' in df_feat.columns:
            top_makes = df_feat['make'].value_counts().head(5).index
            filtered_df = df_feat[df_feat['make'].isin(top_makes)]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.kdeplot(data=filtered_df, x="electric_range", hue="make", fill=True, ax=ax, palette="Set2")
            ax.set_facecolor('none')
            fig.patch.set_alpha(0)
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Change legend text color
            legend = ax.get_legend()
            if legend:
                plt.setp(legend.get_texts(), color='white')
                
            st.pyplot(fig)
            
    else:
        st.warning("Visualizations require the raw dataset.")

with tab3:
    st.header("Machine Learning Performance")
    st.markdown("Real-time inference and evaluation of regression and classification models on the EV dataset.")
    
    if st.button("🚀 Train & Evaluate Models"):
        if not df_feat.empty:
            with st.spinner("Training models... This might take a minute."):
                from src.pipeline import run_models
                # We limit the sample size to 20,000 to ensure Streamlit doesn't timeout
                sample_size = min(20000, len(df_feat))
                df_sample = df_feat.sample(sample_size, random_state=42)
                reg_results, clf_results = run_models(df_sample)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Regression Models (R² Score)")
                    st.dataframe(reg_results.sort_values(by="Accuracy (%)", ascending=False), use_container_width=True)
                    st.bar_chart(reg_results.set_index("Model")["Accuracy (%)"].sort_values(ascending=False))
                    
                with col2:
                    st.subheader("Classification Models (Accuracy)")
                    st.dataframe(clf_results.sort_values(by="Accuracy (%)", ascending=False), use_container_width=True)
                    st.bar_chart(clf_results.set_index("Model")["Accuracy (%)"].sort_values(ascending=False))
        else:
            st.error("No feature dataset found to train models.")
