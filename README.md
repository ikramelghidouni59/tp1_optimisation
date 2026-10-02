TP d'optimisation avec NumPy

TP du Master IAA FSA Ouarzazate, Université Ibnou Zohr, 2025/2026. Réalisé par Ikram El Ghidouni, encadré par Dr. A. HADRI.

Dans ce TP, j'ai codé avec NumPy sept méthodes d'optimisation : GD, GD avec recherche de pas, SGD, Momentum, AdaGrad, RMSprop et Adam. Je n'ai pas utilisé scipy.optimize ni torch.optim.

Les trois exercices
Régression (OLS, Ridge, Ridge à noyau RBF) sur les jeux nonlinear et diabetes.
Classification de classification_spiral3.csv avec un petit réseau de neurones (2 → 16 → 3, softmax).
Clustering en minimisant une fonction soft-min J_tau, comparé à la méthode de Lloyd, sur clustering_blobs3.csv, clustering_wine.csv, Fashion-MNIST et MNIST.
Dossiers
data/ : les fichiers CSV
results/ : les tableaux de résultats (CSV et Markdown)
figures/ : les courbes et les graphiques
rapport/ : le rapport en LaTeX et en PDF
Utilisation

Il faut Python avec numpy, pandas, matplotlib et scikit-learn.

pip install numpy pandas matplotlib scikit-learn

Les expériences ont été lancées sur Google Colab. TODO : ajouter ici les noms des fichiers Python et comment les lancer.

Méthode
Les données sont séparées en train, validation et test. La standardisation est calculée sur le train seulement.
Les hyperparamètres sont choisis sur la validation. Le test sert uniquement à la fin.
Les graines utilisées sont 42, 123 et 2024. Les résultats sont donnés en moyenne ± écart type.
Les gradients sont vérifiés avec des différences finies centrales (h = 1e-5).
Résultats en bref
Exercice 1 : sur nonlinear, le Ridge à noyau RBF avec Adam est le meilleur (RMSE test 0.1739, R² 0.9552). Sur diabetes, OLS avec SGD est le meilleur (RMSE test 58.3477).
Exercice 2 : tous les optimiseurs dépassent 98 % d'accuracy. Adam est le meilleur (0.9963).
Exercice 3 : sur blobs, toutes les méthodes donnent le même résultat. Sur wine, l'ARI est autour de 0.90. Sur Fashion-MNIST et MNIST, l'ARI reste entre 0.33 et 0.36.

Les tableaux complets sont dans results/ et dans le rapport.
