# 💬 Analyse de Sentiment — Avis Produits Amazon

> Pipeline NLP end-to-end pour extraire automatiquement les points
> positifs et négatifs de 568 454 avis clients Amazon.

## 🚀 Demo en ligne
👉 [Accéder au dashboard](https://analyse-sentiment-nlp-9hb4rudvsmrpva6wjswohn.streamlit.app/)

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Accuracy globale | 85.13% |
| F1-score positif | 93% |
| F1-score négatif | 72% |
| Modèle | SVM LinearSVC |

## 🗂️ Dataset

- **Source** : [Amazon Fine Food Reviews — Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Volume** : 568 454 avis clients
- **Période** : 1999 — 2012

## 🛠️ Stack technique

- **Langage** : Python 3.11
- **NLP** : NLTK, scikit-learn (TF-IDF, LinearSVC, LDA)
- **Dashboard** : Streamlit
- **Versioning** : Git / GitHub

## 📁 Structure du projet
Analyse-sentiment-NLP/
├── notebooks/
│   ├── 01_exploration.ipynb     ← Exploration du dataset
│   ├── 02_nettoyage.ipynb       ← Nettoyage du texte (7 étapes)
│   ├── 03_modelisation.ipynb    ← Entraînement SVM + cross-validation
│   └── 04_topics.ipynb          ← Extraction topics TF-IDF + LDA
├── output/                      ← Données générées (non versionnées)
├── app.py                       ← Dashboard Streamlit
├── requirements.txt
└── README.md

## 🚀 Lancer le projet

**1. Cloner le dépôt**
```bash
git clone https://github.com/Dboy003/Analyse-sentiment-NLP.git
cd Analyse-sentiment-NLP
```

**2. Créer un environnement virtuel**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Installer les dépendances**
```bash
pip install -r requirements.txt
```

**4. Télécharger le dataset**

Télécharger `Reviews.csv` depuis [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
et le placer dans le dossier `data/`.

**5. Lancer les notebooks dans l'ordre**
01_exploration → 02_nettoyage → 03_modelisation → 04_topics

**6. Lancer le dashboard**
```bash
streamlit run app.py
```

## 📈 Pipeline NLP
Données brutes (Reviews.csv)
↓
Exploration & labélisation (01)
↓
Nettoyage texte 7 étapes (02)
↓
Modélisation SVM (03)
↓
Extraction topics TF-IDF + LDA (04)
↓
Dashboard Streamlit (app.py)

## 🔍 Fonctionnalités du dashboard

- 📊 Visualisation de la répartition des sentiments
- ☁️ Nuage de mots par sentiment
- 🔍 Prédiction en temps réel sur un avis saisi
- 📈 Métriques détaillées du modèle

## 👤 Auteur

**Mourad** — [GitHub](https://github.com/Dboy003)
