# 📊 Supplier Clustering Dashboard

Ce projet est une application Streamlit interactive qui permet de **visualiser des clusters de fournisseurs** à partir de leurs comportements de paiement et de litiges. L'application utilise l'algorithme de **KMeans** avec une réduction de dimension via **PCA** pour faciliter la visualisation. Elle permet aussi d'évaluer un nouveau fournisseur en ligne, en l'intégrant dynamiquement dans les clusters existants.

---

## 🚀 Fonctionnalités

- **Visualisation des clusters de fournisseurs** 
- **Ajout dynamique** d’un nouveau fournisseur à évaluer.
- **Indicateurs de performance du clustering** :
  - Silhouette Score
  - Calinski-Harabasz Index
  - Davies-Bouldin Score
- **Vue détaillée** des fournisseurs et statistiques des clusters.
- **Mise en évidence personnalisée** d’un fournisseur sélectionné.

---

## 🧠 Technologies utilisées

- Python
- Streamlit
- scikit-learn (KMeans, PCA, preprocessing)
- Pandas, NumPy
- Matplotlib, Seaborn
- joblib (chargement de modèles)

---



