import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import json
import matplotlib.pyplot as plt
import os
import base64
import requests
import time
from io import BytesIO

# --- CONFIG S3 PUBLIC ---
BUCKET_URL = "https://pac-jedha-bucket.s3.amazonaws.com/G4"

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AI FRAUD DETECTION",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS "CYBERPUNK" AVANCÉ ---
st.markdown("""
    <style>
    .stApp {
        background-color: #030508;
        background-image: 
            radial-gradient(circle at 50% 50%, #1a2c4e 0%, transparent 50%),
            linear-gradient(0deg, rgba(0,255,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,255,255,0.05) 1px, transparent 1px);
        background-size: 100% 100%, 40px 40px, 40px 40px;
        color: #e0f7fa;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    h1 {
        text-align: center; font-family: 'Orbitron', sans-serif; font-weight: 700; font-size: 3rem !important;
        background: -webkit-linear-gradient(#00e5ff, #2979ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 229, 255, 0.6); margin-bottom: 0px; text-transform: uppercase;
    }
    .subtitle { text-align: center; color: #00e5ff; font-size: 1.1rem; letter-spacing: 3px; margin-bottom: 40px; opacity: 0.8; }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-7px);}
        60% {transform: translateY(-3px);}
    }
    .bouncing-triangle {
        text-align: center; color: #00e5ff; font-size: 28px; margin-top: -10px; margin-bottom: 5px; animation: bounce 2s infinite;
    }

    .col-header {
        color: #00e5ff; font-size: 1rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 600; text-align: center; border-bottom: 1px solid rgba(0, 229, 255, 0.2); padding-bottom: 5px;
    }
    div.stButton > button {
        background: rgba(0, 20, 40, 0.5); border: 1px solid #00e5ff; color: #00e5ff; border-radius: 6px; height: 50px; width: 100%; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; transition: all 0.3s ease; box-shadow: 0 0 10px rgba(0, 229, 255, 0.1);
    }
    div.stButton > button:hover { background: rgba(0, 229, 255, 0.15); box-shadow: 0 0 20px rgba(0, 229, 255, 0.5); color: #fff; border-color: #fff; }
    div.stButton > button:disabled { border-color: rgba(0, 229, 255, 0.3); color: rgba(0, 229, 255, 0.5); background: rgba(0, 20, 40, 0.2); cursor: default; opacity: 1; }
    div.stButton > button[kind="primary"] { border-color: #ff2b2b; color: #ff2b2b; }
    .result-card { background: rgba(10, 20, 30, 0.8); border: 1px solid; border-radius: 15px; padding: 20px; margin-bottom: 20px; backdrop-filter: blur(10px); }
    .streamlit-expanderHeader { background-color: rgba(0, 229, 255, 0.05); color: #e0f7fa; border-radius: 5px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- 3. DÉFINITIONS ET TYPOLOGIES ---
FEATURE_DEFINITIONS = {
    "user_id": "Identifiant Client", "account_age_days": "Ancienneté du compte (jours)",
    "hour_local": "Heure locale", "dayofweek_local": "Jour de la semaine (0=Dim)",
    "day_local": "Jour du mois", "is_night": "Transaction nocturne (0/1)",
    "time_since_last": "Temps depuis dernière transac. (sec)", "amount": "Montant de la transaction ($)",
    "avg_amount_user": "Montant moyen habituel", "avg_amount_user_past": "Moyenne historique des montants",
    "amount_diff_user_avg": "Écart vs Moyenne habituelle", "amount_delta_prev": "Écart vs Transaction précédente",
    "total_transactions_user": "Nb total de transactions", "transaction_count_cum": "Compteur cumulé transactions",
    "tx_last_24h": "Nb transactions dernières 24h", "tx_last_7d": "Nb transactions derniers 7 jours",
    "tx_last_30d": "Nb transactions derniers 30 jours", "user_tx_count": "Compteur transactions utilisateur",
    "user_fraud_rate": "Taux de fraude historique", "user_fraud_count": "Nb fraudes passées",
    "user_has_fraud_history": "Client a déjà fraudé (Oui/Non)", "is_new_account": "Nouveau compte (< 30 jours)",
    "shipping_distance_km": "Distance Livraison <-> IP (km)", "distance_amount_ratio": "Ratio Distance / Montant",
    "country_bin_mismatch": "Pays carte vs pays déclaré différent", "avs_match": "Vérification Adresse (AVS) OK",
    "cvv_result": "Vérification CVV OK", "three_ds_flag": "Authentification 3D-Secure",
    "security_mismatch_score": "Score d'incohérence sécurité"
}

def get_typology(feature_name):
    str_feat = str(feature_name)
    if str_feat in ["hour_local", "dayofweek_local", "day_local", "is_night"]: return "Variables Temporelles"
    if str_feat in ["account_age_days", "total_transactions_user", "transaction_count_cum", "avg_amount_user", "avg_amount_user_past", "amount_diff_user_avg", "amount_delta_prev", "user_fraud_rate", "user_fraud_count", "user_has_fraud_history"]: return "Comportement Utilisateur"
    if str_feat in ["security_mismatch_score", "avs_match", "cvv_result", "three_ds_flag", "is_new_account"]: return "Indicateurs de Sécurité"
    if str_feat in ["shipping_distance_km", "distance_amount_ratio", "country_bin_mismatch"] or str_feat.startswith("country_") or str_feat.startswith("bin_country_"): return "Géographie & Logistique"
    if str_feat in ["tx_last_24h", "tx_last_7d", "tx_last_30d", "time_since_last", "user_tx_count"]: return "Dynamiques Temporelles"
    return "Autre"

def get_definition(feature_name):
    if feature_name in FEATURE_DEFINITIONS: return FEATURE_DEFINITIONS[feature_name]
    if str(feature_name).startswith("country_"): return f"Pays User : {feature_name.split('_')[1]}"
    if str(feature_name).startswith("bin_country_"): return f"Pays Banque : {feature_name.split('_')[2]}"
    if str(feature_name).startswith("merchant_category_"): return f"Cat : {feature_name.split('_')[2]}"
    return feature_name

# --- 4. ETAT ---
if 'p1_data' not in st.session_state: st.session_state['p1_data'] = None
if 'p1_tx_id' not in st.session_state: st.session_state['p1_tx_id'] = None
if 'p1_true_label' not in st.session_state: st.session_state['p1_true_label'] = None
if 'p2_reset_counter' not in st.session_state: st.session_state['p2_reset_counter'] = 0

# --- 5. FONCTIONS MÉTIER ---
def download_from_s3(filename):
    if not os.path.exists(filename):
        url = f"{BUCKET_URL}/{filename}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f: f.write(response.content)
        except Exception: pass

@st.cache_resource
def load_model_data():
    filename = "fraud_xgb_model.pkl"
    download_from_s3(filename)
    try:
        with open(filename, "rb") as f: data = pickle.load(f)
        model = data['model'] if isinstance(data, dict) and 'model' in data else data
        features = data['final_columns'] if isinstance(data, dict) and 'final_columns' in data else getattr(model, "feature_names_in_", None)
        return model, features
    except Exception: return None, None

@st.cache_data
def load_full_test_data():
    download_from_s3("X_test_app.csv"); download_from_s3("y_test_app.csv")
    try:
        X = pd.read_csv("X_test_app.csv", index_col=0); y = pd.read_csv("y_test_app.csv")
        if y.shape[1] > 1: y.set_index(y.columns[0], inplace=True)
        y = y.iloc[:, 0]; y.index = X.index
        return X, y
    except Exception: return None, None

@st.cache_data
def get_global_shap_importance(_model, X_reference):
    try:
        X_sample = X_reference.sample(min(200, len(X_reference)), random_state=42)
        explainer = shap.TreeExplainer(_model)
        shap_values = explainer(X_sample)
        global_impact = np.abs(shap_values.values).mean(axis=0)
        return pd.DataFrame({'Feat': X_sample.columns, 'Global_Avg_Impact': global_impact}).set_index('Feat')
    except Exception: return None

def align_features(row, model_features):
    if model_features is None: return row 
    if 'user_id' in model_features and 'user_id' not in row.columns:
        row = row.reset_index(); col = 'index' if 'index' in row.columns else 'level_0'
        if col in row.columns: row = row.rename(columns={col: 'user_id'})
    return row.reindex(columns=model_features, fill_value=0)

def consolidate_ohe_for_display(df):
    res = df.copy()
    def get_max_col(row, prefix):
        cols = [c for c in df.columns if str(c).startswith(prefix) and "mismatch" not in str(c) and row[c] == 1]
        return cols[0].replace(prefix, "") if cols else "Other"
    if any(c.startswith("country_") for c in df.columns):
        res['Pays Client'] = df.apply(lambda r: get_max_col(r, "country_"), axis=1)
    if any(c.startswith("bin_country_") for c in df.columns):
        res['Pays Banque'] = df.apply(lambda r: get_max_col(r, "bin_country_"), axis=1)
    if any(c.startswith("merchant_category_") for c in df.columns):
        res['Catégorie'] = df.apply(lambda r: get_max_col(r, "merchant_category_"), axis=1)
    if 'channel_app' in df.columns:
        res['Canal'] = np.where(df['channel_app'] == 1, 'App', 'Web')
    cols_to_drop = [c for c in df.columns if any(p in str(c) for p in ["country_", "bin_country_", "merchant_category_", "channel_"])]
    return res.drop(columns=cols_to_drop)

# --- 6. ACTIONS ---
def action_generate_fraud():
    X, y = load_full_test_data()
    if X is not None:
        cands = y[y == 1].index
        if len(cands) > 0:
            sel_id = np.random.choice(cands)
            st.session_state['p1_tx_id'], st.session_state['p1_data'], st.session_state['p1_true_label'] = sel_id, X.loc[[sel_id]], 1

def action_generate_legit():
    X, y = load_full_test_data()
    if X is not None:
        cands = y[y == 0].index
        if len(cands) > 0:
            sel_id = np.random.choice(cands)
            st.session_state['p1_tx_id'], st.session_state['p1_data'], st.session_state['p1_true_label'] = sel_id, X.loc[[sel_id]], 0

def action_global_reset():
    st.session_state['p1_data'] = None; st.session_state['p1_tx_id'] = None; st.session_state['p1_true_label'] = None; st.session_state['p2_reset_counter'] += 1

# --- 7. INTERFACE PRINCIPALE ---
def main():
    st.markdown("<h1>AI FRAUD DETECTION &mdash; GROUP 4</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>SYSTÈME D'ANALYSE DE DONNÉES DE FRAUDE</div>", unsafe_allow_html=True)
    model, model_features = load_model_data()
    X_ref, _ = load_full_test_data()
    
    global_shap_df = get_global_shap_importance(model, align_features(X_ref.head(200), model_features)) if X_ref is not None and model else None

    col_left, col_mid, col_right = st.columns([1, 0.2, 1])

    with col_left:
        st.markdown("<div class='col-header'>1. SIMULATOR (INTERNAL)</div>", unsafe_allow_html=True)
        st.button("GENERATE FRAUD", key="btn_fraud", type="primary", use_container_width=True, on_click=action_generate_fraud)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='col-header'>2. FILE AUDITOR (JSON)</div>", unsafe_allow_html=True)
        uploader_key = f"up_{st.session_state['p2_reset_counter']}"
        f_file = st.file_uploader("Drop Transaction JSON", type=["json"], key=f"feat_{uploader_key}", label_visibility="collapsed")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='col-header'>3. BATCH AUDITOR (CSV)</div>", unsafe_allow_html=True)
        batch_file = st.file_uploader("Upload Test Dataset CSV", type=["csv"], key=f"batch_{uploader_key}", label_visibility="collapsed")

    with col_mid:
        st.markdown("<br><br><br>", unsafe_allow_html=True) 
        img_path = "logo_G4.jpg"
        download_from_s3(img_path)
        if not os.path.exists(img_path): 
            img_path = "logo_G4_antifraude.jpg"
            download_from_s3(img_path)
        
        b64_img = ""
        if os.path.exists(img_path):
            with open(img_path, "rb") as f: b64_img = base64.b64encode(f.read()).decode()
        if b64_img:
            st.markdown(f"""<style>div[data-testid="stVerticalBlock"] > div:has(span#logo-trigger) + div button {{ background-image: url('data:image/jpeg;base64,{b64_img}'); background-size: contain; background-repeat: no-repeat; background-position: center; background-color: transparent !important; border: none !important; color: transparent !important; height: 120px; width: 100%; box-shadow: none !important; cursor: pointer; mix-blend-mode: screen; }} div[data-testid="stVerticalBlock"] > div:has(span#logo-trigger) + div button:hover {{ transform: scale(1.05); filter: drop-shadow(0 0 10px #00e5ff); }} div[data-testid="stVerticalBlock"] > div:has(span#logo-trigger) + div button p {{ display: none; }} </style>""", unsafe_allow_html=True)
        st.markdown('<span id="logo-trigger"></span>', unsafe_allow_html=True)
        st.button("LOGO_TRIGGER", key="logo_reset_btn", on_click=action_global_reset, use_container_width=True)
        st.markdown("<div class='bouncing-triangle'>▲</div>", unsafe_allow_html=True)
        st.button("GLOBAL RESET", key="label_reset_btn", disabled=True, use_container_width=True)

    with col_right:
        st.markdown("<div class='col-header'>1. SIMULATOR (INTERNAL)</div>", unsafe_allow_html=True)
        st.button("GENERATE LEGIT", key="btn_legit", use_container_width=True, on_click=action_generate_legit)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='col-header'>2. GROUND TRUTH (JSON)</div>", unsafe_allow_html=True)
        t_file = st.file_uploader("Drop Truth JSON", type=["json"], key=f"targ_{uploader_key}", label_visibility="collapsed")

    if batch_file is not None and model:
        try:
            df_batch = pd.read_csv(batch_file)
            df_ready = align_features(df_batch, model_features)
            st.divider()
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i in range(1, 11):
                time.sleep(0.05)
                progress_bar.progress(i * 10)
                status_text.text(f"Analyse des transactions : {i*10}% complété...")
            
            probs = model.predict_proba(df_ready)[:, 1]
            df_batch['Fraud_Probability'] = probs
            frauds_detected = df_batch[df_batch['Fraud_Probability'] > 0.30].copy()
            
            st.markdown("<h3 style='color:#ff2b2b;'>Résultats de l'Audit de Masse</h3>", unsafe_allow_html=True)
            with st.expander(f"🔍 Voir les {len(frauds_detected)} alertes critiques détectées", expanded=True):
                if not frauds_detected.empty:
                    display_df = consolidate_ohe_for_display(frauds_detected)
                    prio_cols = ['amount', 'Fraud_Probability', 'Pays Client', 'Pays Banque', 'Catégorie', 'Canal', 'security_mismatch_score']
                    cols_to_show = [c for c in prio_cols if c in display_df.columns]
                    st.dataframe(display_df[cols_to_show].sort_values('Fraud_Probability', ascending=False).style.background_gradient(cmap='Reds', subset=['Fraud_Probability']).format("{:.1%}", subset=['Fraud_Probability']), use_container_width=True)
                    total_loss = frauds_detected['amount'].sum()
                    st.markdown(f"""<div style='background: rgba(255, 43, 43, 0.2); border: 2px solid #ff2b2b; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;'> <h3 style='margin:0; color:#fff;'>PERTE POTENTIELLE TOTALE ÉVITÉE</h3> <h2 style='margin:0; color:#ff2b2b; font-size: 2.5rem;'>$ {total_loss:,.2f}</h2> </div>""", unsafe_allow_html=True)
                else:
                    st.success("Aucune fraude détectée dans ce dataset.")
        except Exception as e: st.error(f"Erreur Audit : {e}")

    row_data, tx_id, true_label, ctx = None, None, None, None
    if st.session_state['p1_data'] is not None:
        row_data, tx_id, true_label, ctx = st.session_state['p1_data'], st.session_state['p1_tx_id'], st.session_state['p1_true_label'], "SIMULATION"
    elif f_file:
        try:
            input_json = json.load(f_file); tx_id = input_json.get('transaction_id', 'Unknown')
            clean_input = {k:v for k,v in input_json.items() if k not in ['transaction_id', 'transaction_id_check', 'ground_truth_label']}
            row_data, ctx = pd.DataFrame([clean_input]), "AUDIT EXTERNE"
            if t_file: tj = json.load(t_file); true_label = tj.get('actual_label', tj.get('ground_truth_label'))
        except: st.error("Erreur JSON")

    if row_data is not None and model:
        st.divider(); run_analysis(row_data, model, model_features, tx_id, true_label, ctx, global_shap_df)

# --- 8. FONCTION D'ANALYSE UNITAIRE ---
def run_analysis(row_data, model, model_features, tx_id, true_label, context, global_shap_df):
    row_ready = align_features(row_data, model_features)
    try:
        proba = model.predict_proba(row_ready)[0][1]
    except: return

    is_fraud = proba > 0.30
    color_res = "#ff2b2b" if is_fraud else "#00e5ff"
    text_res = "🚨 FRAUDE DÉTECTÉE" if is_fraud else "✅ TRANSACTION LÉGITIME"

    c_data, c_viz = st.columns([1, 2])
    with c_data:
        st.markdown(f"<div class='result-card' style='border-color: {color_res};'><h2 style='text-align:center; color: {color_res}; margin:0;'>{text_res}</h2><p style='text-align:center;'>Score: <b>{proba:.1%}</b></p></div>", unsafe_allow_html=True)
        if true_label is not None:
            lbl = int(true_label)
            match = (lbl == 1 and is_fraud) or (lbl == 0 and not is_fraud)
            v_col, v_txt = ("#00e5ff", "CORRECT") if match else ("#ff2b2b", "ERREUR")
            st.markdown(f"<div style='text-align:center; border: 1px solid {v_col}; padding: 10px; border-radius: 10px; background: rgba(0,0,0,0.3);'>Vérité: {'FRAUDE' if lbl==1 else 'LÉGITIME'} vs IA: <b style='color:{v_col}'>{v_txt}</b></div>", unsafe_allow_html=True)
        st.json(row_data.iloc[0].to_dict(), expanded=False)

    with c_viz:
        try:
            explainer = shap.TreeExplainer(model); shap_values = explainer(row_ready); vals = shap_values.values[0]
            df_s = pd.DataFrame({'Feat': row_ready.columns, 'Imp': vals, 'Abs': np.abs(vals)})
            df_s['Définition'] = df_s['Feat'].apply(get_definition); df_s['Typologie'] = df_s['Feat'].apply(get_typology)

            if is_fraud:
                top_factors = df_s[df_s['Imp'] > 0].sort_values('Imp', ascending=False).head(5)
                title, col = "Facteurs de risque critiques", "#ff2b2b"
            else:
                top_factors = df_s[df_s['Imp'] < 0].sort_values('Imp', ascending=True).head(5)
                title, col = "Piliers de confiance structurels", "#00e5ff"

            if not top_factors.empty:
                st.markdown(f"<h4 style='color:{col}; font-size:1.5rem; margin-bottom:10px;'>📝 {title}</h4>", unsafe_allow_html=True)
                bullet_list = "".join([f"<li style='font-size:1.15rem; margin-bottom:5px;'><b>{r['Définition']}</b> (<span style='color:{col}'>{r['Imp']:+.3f}</span>)</li>" for _, r in top_factors.iterrows()])
                st.markdown(f"<div style='padding: 20px; border-left: 6px solid {col}; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 20px;'><p style='font-size:1.15rem; margin-top:0;'>L'algorithme a corrélé cette décision aux facteurs suivants :</p><ul style='margin-bottom:0;'>{bullet_list}</ul></div>", unsafe_allow_html=True)

            st.markdown("<h3 style='color:#00e5ff; margin-top:0;'>Facteurs d'Influence (SHAP)</h3>", unsafe_allow_html=True)
            plt.style.use('dark_background'); fig = plt.figure(figsize=(10, 3)); shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            fig.patch.set_facecolor('#030508'); st.pyplot(fig, bbox_inches='tight'); plt.close(fig)

            cols_disp = ["Feat", "Définition", "Typologie", "Imp"]
            with st.expander("1. Top 5 Global (Impact Absolu)"):
                st.dataframe(df_s.sort_values('Abs', ascending=False).head(5)[cols_disp].rename(columns={"Feat": "Feature", "Imp": "Impact"}).style.background_gradient(cmap='coolwarm', subset=['Impact']), use_container_width=True, hide_index=True)
            with st.expander("🚨 2. Top 5 Facteurs de risque"):
                top_risk_df = df_s[df_s['Imp'] > 0].sort_values('Imp', ascending=False).head(5)
                if not top_risk_df.empty: st.dataframe(top_risk_df[cols_disp].rename(columns={"Feat": "Feature", "Imp": "Impact"}).style.background_gradient(cmap='Reds', subset=['Impact']), use_container_width=True, hide_index=True)
            with st.expander("🛡️ 3. Top 5 Facteurs de confiance"):
                top_conf_df = df_s[df_s['Imp'] < 0].sort_values('Imp', ascending=True).head(5)
                if not top_conf_df.empty: st.dataframe(top_conf_df[cols_disp].rename(columns={"Feat": "Feature", "Imp": "Impact"}).style.background_gradient(cmap='Blues_r', subset=['Impact']), use_container_width=True, hide_index=True)

            if global_shap_df is not None:
                st.markdown("---")
                with st.expander("🌍 Comparaison vs Modèle Global"):
                    # REPRISE DANS L'ORDRE : TOP 5 RISQUES PUIS TOP 5 CONFIANCE
                    top_risk_local = df_s[df_s['Imp'] > 0].sort_values('Imp', ascending=False).head(5)
                    top_conf_local = df_s[df_s['Imp'] < 0].sort_values('Imp', ascending=True).head(5)
                    
                    combo = pd.concat([top_risk_local, top_conf_local])
                    final = combo.set_index('Feat').join(global_shap_df, how='inner')
                    final['Sur-Imp'] = final['Abs'] - final['Global_Avg_Impact']
                    final['Nature'] = np.where(final['Imp'] > 0, '🚨 Risque', '🛡️ Confiance')
                    
                    st.dataframe(
                        final[['Nature', 'Définition', 'Typologie', 'Abs', 'Global_Avg_Impact', 'Sur-Imp']]
                        .rename(columns={'Abs': 'Local', 'Global_Avg_Impact': 'Global', 'Sur-Imp': 'Écart'})
                        .style.background_gradient(cmap='Purples', subset=['Écart'])
                        .format("{:.2f}", subset=['Local', 'Global', 'Écart']), 
                        use_container_width=True
                    )

        except Exception as e: st.warning(f"Analyse SHAP limitée: {e}")

if __name__ == "__main__":
    main()
