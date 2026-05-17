import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse Sentiment Avis Produits",
    page_icon="💬",
    layout="wide"
)

# ─── Chargement des ressources ──────────────────────────────────────────────
@st.cache_resource
def charger_modele():
    return joblib.load("outputs/modele_sentiment.pkl")

@st.cache_data
def charger_donnees():
    return pd.read_csv("outputs/data_dashboard.csv")

@st.cache_data
def charger_mots_cles():
    return pd.read_csv("outputs/04_mots_cles.csv")

@st.cache_data
def charger_topics():
    return pd.read_csv("outputs/04_topics_lda.csv")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

modele     = charger_modele()
df         = charger_donnees()
df_mots    = charger_mots_cles()
df_topics  = charger_topics()

# ─── Menu en haut ───────────────────────────────────────────────────────────
page = option_menu(
    menu_title=None,
    options=["Accueil", "Visualisations", "Prédiction", "Performance"],
    icons=["house", "bar-chart", "search", "graph-up"],
    orientation="horizontal"
)

# ─── Pages ──────────────────────────────────────────────────────────────────
if page == "Accueil":
    st.title("💬 Analyse de Sentiment : Avis Produits Amazon")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("📝 Nombre d'avis analysés", f"{len(df):,}")
    col2.metric("🎯 Accuracy du modèle",     "85.67%")
    col3.metric("🤖 Modèle utilisé",         "SVM LinearSVC")

    st.markdown("---")
    st.markdown("""
    ### À propos du projet
    Ce dashboard analyse automatiquement **568 453 avis clients** Amazon
    pour en extraire le sentiment (positif, négatif, neutre) et les thèmes principaux.

    ### Navigation
    - 📊 **Visualisations** : Répartition des sentiments et mots clés
    - 🔍 **Prédiction** : Testez le modèle sur votre propre avis
    - 📈 **Performance** : Métriques détaillées du modèle
    """)

elif page == "Visualisations":
    st.title("📊 Visualisations")
    st.markdown("---")

    # ─── Répartition des sentiments ─────────────────────────────────────
    st.subheader("Répartition des sentiments")

    col1, col2 = st.columns(2)

    repartition = df["sentiment"].value_counts()
    couleurs    = ["#2ecc71", "#e74c3c", "#f39c12"]

    # Camembert
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie(
            repartition.values,
            labels=repartition.index,
            colors=couleurs,
            autopct="%1.1f%%",
            startangle=90
        )
        ax1.set_title("Répartition en pourcentage")
        st.pyplot(fig1)

    # Barres
    with col2:
        fig2, ax2 = plt.subplots()
        ax2.bar(
            repartition.index,
            repartition.values,
            color=couleurs
        )
        ax2.set_title("Nombre d'avis par sentiment")
        ax2.set_ylabel("Nombre d'avis")
        for i, v in enumerate(repartition.values):
            ax2.text(i, v + 1000, f"{v:,}", ha="center", fontsize=9)
        st.pyplot(fig2)

    st.markdown("---")

    # ─── Wordcloud ───────────────────────────────────────────────────────
    st.subheader("Nuages de mots")

    sentiment_choisi = st.selectbox(
        "Choisir un sentiment :",
        ["positif", "negatif", "neutre"]
    )

    couleur_map = {
        "positif": "Greens",
        "negatif": "Reds",
        "neutre" : "Oranges"
    }

    # Limiter à 10 000 avis pour la vitesse
    # Mots parasites à exclure du wordcloud
    mots_exclus = {"br", "ive", "ve", "don", "doesn", "didn",
               "isn", "wasn", "wouldn", "couldn", "one",
               "get", "make", "would", "really", "much",
               "im", "dont", "doesnt", "u", "alway"}

    texte_combine = " ".join(
    df[df["sentiment"] == sentiment_choisi]["text_nettoye"]
    .dropna()
    .sample(n=min(10000, len(df[df["sentiment"] == sentiment_choisi])),
            random_state=42)
)

# Supprimer les mots parasites directement du texte
    texte_filtre = " ".join([
    mot for mot in texte_combine.split()
    if mot not in mots_exclus and len(mot) > 2
    ])

    wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap=couleur_map[sentiment_choisi],
    max_words=100
    ).generate(texte_filtre)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.imshow(wordcloud, interpolation="bilinear")
    ax3.axis("off")
    st.pyplot(fig3)

    st.markdown("---")

    # ─── Topics LDA ──────────────────────────────────────────────────────
    st.subheader("Thèmes principaux (LDA)")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### 🟢 Topics positifs")
        topics_pos = df_topics[df_topics["sentiment"] == "positif"]
        for _, row in topics_pos.iterrows():
            st.markdown(f"**{row['topic']}** : {row['mots']}")

    with col4:
        st.markdown("#### 🔴 Topics négatifs")
        topics_neg = df_topics[df_topics["sentiment"] == "negatif"]
        for _, row in topics_neg.iterrows():
            st.markdown(f"**{row['topic']}** : {row['mots']}")

elif page == "Prédiction":
    st.title("🔍 Prédiction en temps réel")
    st.markdown("---")

    st.markdown("### Testez le modèle sur votre propre avis")
    avis = st.text_area("Entrez un avis en anglais :", height=150,
                        placeholder="Ex: This product is amazing, I love it!")

    if st.button("Analyser"):
        if avis.strip() == "":
            st.warning("Veuillez entrer un avis avant d'analyser.")
        else:
            # Prédiction du sentiment
            sentiment = modele.predict([avis])[0]

            # Score de confiance
            if hasattr(modele, "decision_function"):
                scores   = modele.decision_function([avis])[0]
                classes  = modele.classes_
                score_max = max(scores)
                confiance = round(
                    100 * (score_max - min(scores)) /
                    (max(scores) - min(scores) + 1e-9), 1
                )
            else:
                confiance = None

            # Affichage du résultat
            couleur_map = {
                "positif": "🟢",
                "negatif": "🔴",
                "neutre" : "🟡"
            }

            st.markdown("---")
            st.markdown(f"### Résultat : {couleur_map[sentiment]} **{sentiment.upper()}**")

            if confiance:
                st.progress(int(confiance))
                st.caption(f"Score de confiance : {confiance}%")

            st.markdown("---")

            # Mots clés détectés
            st.markdown("### Mots clés détectés dans votre avis")

            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words("english"))

            mots_avis = [
                lemmatizer.lemmatize(mot)
                for mot in re.sub(r"[^a-z\s]", "", avis.lower()).split()
                if mot not in stop_words and len(mot) > 2
            ]

            mots_positifs = set(df_mots["mot_positif"].tolist())
            mots_negatifs = set(df_mots["mot_negatif"].tolist())

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🟢 Mots positifs détectés")
                detectes_pos = [m for m in mots_avis if m in mots_positifs]
                if detectes_pos:
                    st.success(", ".join(detectes_pos))
                else:
                    st.info("Aucun mot clé positif détecté")

            with col2:
                st.markdown("#### 🔴 Mots négatifs détectés")
                detectes_neg = [m for m in mots_avis if m in mots_negatifs]
                if detectes_neg:
                    st.error(", ".join(detectes_neg))
                else:
                    st.info("Aucun mot clé négatif détecté")

elif page == "Performance":
    st.title("📈 Performance du modèle")
    st.markdown("---")

    # ─── Métriques globales ──────────────────────────────────────────────
    st.subheader("Métriques globales")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Accuracy",  "85.67%")
    col2.metric("🤖 Modèle",    "SVM LinearSVC")
    col3.metric("⚙️ Paramètre C", "0.1")

    st.markdown("---")

    # ─── Rapport détaillé ────────────────────────────────────────────────
    st.subheader("Rapport détaillé par classe")

    df_rapport = pd.DataFrame({
        "Classe"    : ["positif", "negatif", "neutre"],
        "Precision" : ["94%", "69%", "40%"],
        "Recall"    : ["92%", "75%", "43%"],
        "F1-score"  : ["93%", "72%", "41%"],
        "Support"   : [66566, 12306, 6396]
    })
    st.dataframe(df_rapport, use_container_width=True)

    st.markdown("---")

    # ─── Explication des métriques ───────────────────────────────────────
    st.subheader("Comprendre les métriques")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.info("""
        **Precision :**
        Sur tous les avis classés "positif",
        combien sont vraiment positifs ?

        Ex: 94% → 94 vrais positifs
        sur 100 prédits positifs
        """)

    with col5:
        st.info("""
        **Recall :**
        Sur tous les vrais avis positifs,
        combien le modèle en détecte ?

        Ex: 92% → le modèle détecte
        92 avis sur 100 vrais positifs
        """)

    with col6:
        st.info("""
        **F1-score :**
        Moyenne entre Precision et Recall.
        C'est le score le plus honnête.

        Ex: 93% → excellent équilibre
        entre precision et recall
        """)

    st.markdown("---")

    # ─── Choix du modèle ─────────────────────────────────────────────────
    st.subheader("Comparaison des modèles testés")

    df_comparaison = pd.DataFrame({
        "Modèle"          : ["Régression Logistique", "SVM LinearSVC"],
        "Accuracy (CV=5)" : ["78.93%", "85.48%"],
        "Paramètre"       : ["défaut", "C=0.1"],
        "Statut"          : ["❌ Rejeté", "✅ Sélectionné"]
    })
    st.dataframe(df_comparaison, use_container_width=True)