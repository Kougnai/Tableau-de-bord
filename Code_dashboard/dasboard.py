import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import requests

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Analytics Restaurant GS", layout="wide")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data 
def load_data():
    path = 'Data/Master_dater_GS_clean.csv'
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["n_jour"] = df["date"].dt.dayofweek
    df["année_str"] = df["année"].astype(str)
    return df

df = load_data()

# --- SIDEBAR (FILTRES) ---
st.sidebar.header("🎯 Filtres Globaux")
years = sorted(df["année"].unique())
selected_year = st.sidebar.multiselect("Années", options=years, default=[2023, 2024, 2025])
selected_pv = st.sidebar.selectbox("Point de Vente", options=["Tous"] + list(df["point de vente"].unique()))

# Filtrage du dataframe
df_filtered = df[df["année"].isin(selected_year)].copy()
if selected_pv != "Tous":
    df_filtered = df_filtered[df_filtered["point de vente"] == selected_pv]

# --- TITRE PRINCIPAL ---
st.title("📊 Rapport de Gestion Les Cassines")
st.markdown(f"Analyse pour : **{selected_pv}** | Période : **{', '.join(map(str, selected_year))}**")

# --- ONGLETS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Performance CA", "🗺️ Répartition & Heatmaps", "☁️ Impact Météo", "📉 Corrélations", "🚀 Prévisions IA"])
# CREER LES TABSET POUR CHAQUE ONGLET
# ---------------------------------------------------------
# TAB 1 : PERFORMANCE CA
# ---------------------------------------------------------
with tab1:
    st.header("Analyse Comparative Mensuelle (YoY)")
    
    # Agrégation (Attention : "total couverts" avec espace)
    ca_agg = df_filtered.groupby(["année_str", "mois", "mois_nom"]).agg({
        "ca ht": 'sum', 
        "food ca": 'sum', 
        "bev ca": 'sum', 
        "total couverts": 'sum',
        "ms/c" : 'sum'
    }).reset_index().sort_values(by=["année_str", "mois"])

    fig_bar = px.bar(
        ca_agg, x="mois_nom", y="ca ht", color="année_str",
        barmode='group', text_auto=".3s",
        color_discrete_map={"2023": "#90C5DA", "2024": "#ABB2B9", "2025": "#2E86C1"},
        template="simple_white",
        labels={"ca ht": "CA HT (€)", "mois_nom": "Mois", "année_str": "Année"}
    )
    fig_bar.update_layout(hovermode="x unified", yaxis=dict(range=[0, ca_agg["ca ht"].max() * 1.2]))
    fig_bar.update_traces(textposition='outside')

    st.plotly_chart(fig_bar, use_container_width=True)

    # --- SECTION MÉTRIQUES COMPARATIVES ---
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    # 1. Identification des années
    latest_year_int = max(selected_year)
    prev_year_int = latest_year_int - 1
    
    # 2. Extraction des données (N et N-1)
    df_n = ca_agg[ca_agg["année_str"] == str(latest_year_int)]
    df_n1 = ca_agg[ca_agg["année_str"] == str(prev_year_int)]
    
    if not df_n.empty:
        # Valeurs Année N
        val_ca_n = df_n["ca ht"].sum()
        val_cvts_n = df_n["total couverts"].sum()
        val_tm_n = val_ca_n / val_cvts_n if val_cvts_n > 0 else 0
        val_msc_n = df_n["ms/c"].sum() /val_ca_n
        
        # Initialisation des deltas
        d_ca, d_cvts, d_tm, d_ms = None, None, None, None
        
        # 3. Calcul des deltas si l'année N-1 existe dans les données
        if not df_n1.empty:
            val_ca_n1 = df_n1["ca ht"].sum()
            val_cvts_n1 = df_n1["total couverts"].sum()
            val_tm_n1 = val_ca_n1 / val_cvts_n1 if val_cvts_n1 > 0 else 0.1
            val_msc_n1 = df_n1["ms/c"].sum()
            val_msc_n1 = val_msc_n1 / val_ca_n1 if val_msc_n1 > 0 else 0 
            
            d_ca = f"{((val_ca_n - val_ca_n1) / val_ca_n1):.1%}"
            d_cvts = f"{((val_cvts_n - val_cvts_n1) / val_cvts_n1):.1%}"
            d_tm = f"{((val_tm_n - val_tm_n1) / val_tm_n1):.1%}" 
            d_ms = f"{((val_msc_n - val_msc_n1) / val_msc_n1):.1%}"


        # 4. Affichage dans Streamlit
        col1.metric(
            label=f"CA HT {latest_year_int}", 
            value=f"{val_ca_n:,.0f} €".replace(",", " "),
            delta=f"{d_ca} vs {prev_year_int}" if d_ca else None
        )
        
        col2.metric(
            label=f"Couverts {latest_year_int}", 
            value=f"{val_cvts_n:,.0f}".replace(",", " "),
            delta=f"{d_cvts} vs {prev_year_int}" if d_cvts else None
        )
        
        col3.metric(
            label=f"Ticket Moyen {latest_year_int}", 
            value=f"{val_tm_n:.2f} €",
            delta=f"{d_tm} vs {prev_year_int}" if d_tm else None
        )

        col4.metric(
            label=f"MS/C {latest_year_int}",
            value=f"{val_msc_n:.1%}",
            delta=f"{d_ms} vs {prev_year_int}" if d_ms else None
        )
    else:
        st.info("Sélectionnez l'année en cours dans les filtres pour voir les performances.")

# ---------------------------------------------------------
# TAB 2 : TREEMAPS & HEATMAPS
# ---------------------------------------------------------
with tab2:
    df_2025 = df_filtered[df_filtered["année"] == 2025].copy()
    df_2025_gr = df_2025.groupby(["mois_nom","semaine_iso", 'point de vente']).agg({
        "ca ht" : 'sum',
        "ms/c" : 'sum',
        'total couverts' : 'sum'
    }).round().reset_index()
    df_2025_gr["Ms/c"] = df_2025_gr["ms/c"] / df_2025_gr["ca ht"]

    if df_2025.empty:
        st.warning(f"Pas de données pour {selected_year} pour l'affichage des cartes.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Structure du CA (2025)")
            path_tree = ["point de vente", "semaine_iso"] if selected_pv != "Tous" else ["point de vente", "semaine_iso"]
            fig_tree = px.treemap(
                df_2025_gr, path=path_tree, values='ca ht',
                color='ca ht', color_continuous_scale='blues',
                template='simple_white', custom_data=["Ms/c", "total couverts"]
            )
            fig_tree.update_traces(
            textinfo = "label+value+percent root",
            texttemplate = "<b>Semaine N° </b>: %{label}<br><b>CA</b> : %{value:.3s} €<br><b>MS/C</b> : %{customdata[0]:.2%}<br><b>Nb cvts</b> : %{customdata[1]:,.2s}<br><b>Part CA %</b> : %{percentRoot:.0%}",
            textfont = dict(size = 12, family='Arial')
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_b:
            st.subheader("Saisonnalité : Jour vs Mois")
            ordre_jours = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            ordre_mois = ["April", 'May', 'June', 'July', 'August', 'September', 'October']
            
            fig_heat = px.density_heatmap(
                df_2025, x="mois_nom", y="day_name", z='ca ht', histfunc='avg',
                category_orders={"day_name": ordre_jours, "mois_nom": ordre_mois},
                text_auto='.2s', color_continuous_scale="blues", template="simple_white",
                labels={"mois_nom" : '', "day_name" : ''}
            )
            fig_heat.update_traces(
            texttemplate = "%{z:,.2s} €",
            hovertemplate = "<br>Mois : %{x}<br>Jour : %{y}<br>CA HT : %{z:,.0f} €"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------
# TAB 3 : IMPACT MÉTÉO
# ---------------------------------------------------------
with tab3:
    st.header("Impact de la météo sur le CA Moyen (2025)")
    st.markdown("Ne pas prendre en compte le nombre de jour pour : TOUS")
    if not df_2025.empty:
        c1, c2 = st.columns(2)
        with c1:
            bins_pluie = [-0.1, 0, 5, 10, 20, float('inf')]
            labels_pluie = ["Sec < 0 mm", "0 - 5 mm", "5 - 10 mm", "10 - 20 mm", "Vigilance > 20 mm"]
            df_2025["Group pluie"] = pd.cut(df_2025["précipitation"], bins=bins_pluie, labels=labels_pluie)
            pluie_agg = df_2025.groupby("Group pluie", observed=False).agg({"ca ht": 'mean', 'précipitation' : 'count'}).round().reset_index()
            fig_pluie = px.bar(pluie_agg, x="Group pluie",
                                y="ca ht", color="ca ht",text_auto ='.2s', hover_data='précipitation',
                                color_continuous_scale="reds_r", labels={"ca ht" : "<b>Chiffre d'affaire HT (€)</b>", "Group pluie" : "<b>Précipitation en mm</b>", "précipitation" : "Nb de jour"},
                                title="Impact Précipitations")
            fig_pluie.update_traces(textposition='outside')
            fig_pluie.update_layout(yaxis = dict(range = [0, pluie_agg["ca ht"].max() * 1.2]))
            st.plotly_chart(fig_pluie, use_container_width=True)

        with c2:
            bins_temp = [-float('inf'), 15, 20, 25, 30, float('inf')]
            labels_temp = ["<15°C", "15-20°C", "20-25°C", "25-30°C", ">30°C"]
            df_2025["Group temp"] = pd.cut(df_2025["température"], bins=bins_temp, labels=labels_temp)
            temp_agg = df_2025.groupby("Group temp", observed=False).agg({"ca ht":'mean', 'précipitation' : 'count'}).round().reset_index()
            fig_temp = px.bar(temp_agg, x="Group temp", 
                              y="ca ht", color="ca ht", text_auto='.2s', hover_data='précipitation',
                              color_continuous_scale="reds_r",labels={"ca ht" : "<b>Chiffre d'affaire HT (€)</b>", "Group temp" : "<b>Température °C</b>", "précipitation" : "Nb de jour"}, 
                              title="Impact Température")
            fig_temp.update_traces(textposition='outside')
            fig_temp.update_layout(yaxis = dict(range = [0, temp_agg["ca ht"].max() * 1.2]))
            st.plotly_chart(fig_temp, use_container_width=True)

# ---------------------------------------------------------
# TAB 4 : CORRÉLATIONS
# ---------------------------------------------------------
with tab4:
    st.header("🔍 Ticket Moyen vs Volume")
    if not df_2025.empty:
        # 1. On prépare les données et on vire les lignes vides
        cvts_tm = df_2025.query("`ticket moyen`< 90").groupby("date").agg({
            "ca ht": 'sum', 
            "total couverts": 'sum'
        }).reset_index().dropna()

        # 2. Calcul du ticket moyen
        cvts_tm["ticket_moyen"] = cvts_tm["ca ht"] / cvts_tm["total couverts"]
        
        # 3. On s'assure qu'on n'a pas de divisions par zéro ou de données aberrantes
        cvts_tm = cvts_tm[cvts_tm["total couverts"] > 0]

        # 4. Le graphique : On utilise le Nb de couverts en X (chiffre) et non la Date
        fig_corr = px.scatter(
            cvts_tm, 
            x="total couverts", # X doit être numérique pour l'OLS
            y="ticket_moyen", 
            trendline="ols", 
            trendline_color_override="red", # On la met en rouge pour bien la voir
            template="simple_white",
            title="Corrélation : Plus il y a de monde, plus le ticket moyen baisse ?",
            labels={"total couverts" : 'Nombre de couverts', "ticket_moyen" : 'Ticket moyen'}
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)

        # 5. Bonus : Afficher les résultats de la régression (R²)
        model = px.get_trendline_results(fig_corr)
        results = model.iloc[0]["px_fit_results"]
        st.write(f"**Coefficient de détermination (R²) :** {results.rsquared:.2f}")
# ---------------------------------------------------------
# TAB 5 : Prévisions IA
# ---------------------------------------------------------
with tab5:
    st.header("🔮 Prévisions de CA à 7 jours")
    
    if selected_pv == "Tous":
        st.warning("⚠️ Veuillez sélectionner un Point de Vente spécifique dans la barre latérale pour lancer la prédiction.")
    else:
        # --- FONCTION CACHE POUR LE MODÈLE ---
        @st.cache_resource
        def train_prophet_model(data):
            # Préparation spécifique Prophet
            df_p = data[['date', 'ca ht', 'température', 'précipitation']].copy()
            df_p.columns = ['ds', 'y', 'temp', 'pluie']
            
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.add_regressor('temp')
            m.add_regressor('pluie')
            m.add_country_holidays(country_name='FR')
            m.fit(df_p)
            return m

       # --- RÉCUPÉRATION MÉTÉO RÉELLE (7j) ---
        @st.cache_data(ttl=3600) # Cache de 1h pour la météo
        def get_forecast_weather():
            url = "https://api.open-meteo.com/v1/forecast?latitude=45.89&longitude=6.12&daily=temperature_2m_max,precipitation_sum&timezone=Europe/Berlin"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status() # Vérifie si la requête a réussi (code 200)
                r = response.json()
                
                # Vérification de la présence de la clé 'daily'
                if 'daily' not in r:
                    st.error("L'API météo n'a pas renvoyé de données 'daily'. Vérifiez l'URL ou le service.")
                    return pd.DataFrame() # Retourne un DF vide pour éviter le crash

                return pd.DataFrame({
                    'ds': pd.to_datetime(r['daily']['time']),
                    'temp': r['daily']['temperature_2m_max'],
                    'pluie': r['daily']['precipitation_sum']
                })
            except Exception as e:
                st.error(f"Erreur lors de la récupération météo : {e}")
                return pd.DataFrame()
        with st.spinner('L\'IA analyse vos données...'):
            # 1. Entraînement sur les données filtrées du PV choisi
            model = train_prophet_model(df_filtered)
            
            # 2. Préparation du futur
            future = model.make_future_dataframe(periods=7)
            weather_futur = get_forecast_weather()
            
            # 3. Fusion météo passée + future
            hist_weather = df_filtered[['date', 'température', 'précipitation']].rename(columns={'date':'ds','température':'temp','précipitation':'pluie'})
            future = pd.merge(future, hist_weather, on='ds', how='left')
            future.update(weather_futur)
            future['temp'] = future['temp'].fillna(df_filtered['température'].mean())
            future['pluie'] = future['pluie'].fillna(0)
            
            # 4. Prédiction
            forecast = model.predict(future)
            
            # 5. Masque de fermeture (Mai à Septembre)
            forecast['yhat'] = forecast.apply(lambda x: x['yhat'] if 5 <= x['ds'].month <= 9 else 0, axis=1)
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0)

        # --- AFFICHAGE ---
        res = forecast.tail(7)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig_pred = px.line(forecast.tail(30), x='ds', y='yhat', title="Prévisions (30 derniers jours + 7 futurs)")
            fig_pred.add_scatter(x=forecast.tail(7)['ds'], y=forecast.tail(7)['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False)
            fig_pred.add_scatter(x=forecast.tail(7)['ds'], y=forecast.tail(7)['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name="Incertitude")
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with col_right:
            st.subheader("📋 Planning suggéré")
            res['ds'] = res['ds'].dt.strftime('%A %d %b')
            st.dataframe(res.rename(columns={'ds':'Date', 'yhat':'CA Prévu (€)'}).set_index('Date').style.format("{:.0f}"))

        # --- CALCUL MS/C CIBLE ---
        st.divider()
        st.subheader("💰 Estimation du besoin Staff")
        target_msc = st.slider("Cible MS/C (%)", 15, 40, 25) / 100
        res['Budget Staff Max'] = res['yhat'] * target_msc
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Potentiel CA (7j)", f"{res['yhat'].sum():,.0f} €")
        with c2:
            st.metric("Budget Staff total (7j)", f"{res['Budget Staff Max'].sum():,.0f} €")
        with c3:
            st.info(f"Basé sur un ratio de {target_msc:.0%}")
