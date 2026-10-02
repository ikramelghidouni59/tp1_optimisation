TP d'optimisation avec NumPy

TP du Master IAA FSA Ouarzazate, Université Ibnou Zohr, 2025/2026. Réalisé par Ikram El Ghidouni, encadré par Dr. A. HADRI.

Dans ce TP, j'ai codé avec NumPy sept méthodes d'optimisation : GD, GD avec recherche de pas, SGD, Momentum, AdaGrad, RMSprop et Adam. Je n'ai pas utilisé scipy.optimize ni torch.optim.

Les trois exercices
Régression (OLS, Ridge, Ridge à noyau RBF) sur les jeux nonlinear et diabetes.
Classification de classification_spiral3.csv avec un petit réseau de neurones (2 → 16 → 3, softmax).
Clustering en minimisant une fonction soft-min J_tau, comparé à la méthode de Lloyd, sur clustering_blobs3.csv, clustering_wine.csv, Fashion-MNIST et MNIST.

Utilisation

Il faut Python avec numpy, pandas, matplotlib et scikit-learn.

pip install numpy pandas matplotlib scikit-learn


Méthode
Les données sont séparées en train, validation et test. La standardisation est calculée sur le train seulement.
Les hyperparamètres sont choisis sur la validation. Le test sert uniquement à la fin.
Les graines utilisées sont 42, 123 et 2024. Les résultats sont donnés en moyenne + ou - écart type.
Les gradients sont vérifiés avec des différences finies centrales .
Résultats en bref
Exercice 1 : sur nonlinear, le Ridge à noyau RBF avec Adam est le meilleur . Sur diabetes, OLS avec SGD est le meilleur .
Exercice 2 : tous les optimiseurs dépassent 98 % d'accuracy. Adam est le meilleur .
Exercice 3 : sur blobs, toutes les méthodes donnent le même résultat. Sur wine, l'ARI est autour de 0.90. Sur Fashion-MNIST et MNIST, l'ARI reste entre 0.33 et 0.36.

Les tableaux complets sont  dans le rapport.
