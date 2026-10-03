# TP d'optimisation avec NumPy

TP du Master IAA, FSA Ouarzazate, Université Ibnou Zohr, 2025/2026.
Réalisé par Ikram El Ghidouni, encadré par Dr. A. HADRI.

Sept méthodes d'optimisation codées avec NumPy : GD, GD avec recherche de pas, SGD,
Momentum, AdaGrad, RMSprop et Adam (plus Lloyd pour le clustering). Aucun optimiseur
préfabriqué (`scipy.optimize`, `torch.optim`) ni différentiation automatique n'est utilisé.

## Contenu
- `exercice_1/` : régression (OLS, Ridge, Ridge à noyau RBF) sur nonlinear et diabetes.
- `exercice_2/` : classification des spirales avec un réseau 2 → 16 → 3 (ReLU, softmax).
- `exercice_3/` : clustering par optimisation d'une perte soft-min, comparé à Lloyd
  (blobs, wine, Fashion-MNIST, MNIST), dans un notebook.

## Installation
```bash
pip install -r requirements.txt
```
Versions utilisées : Python 3.13, NumPy 2.1.3, pandas 2.2.3.

## Vérification (gradients et règles de mise à jour)
```bash
cd exercice_1 && python verify_ex1.py --data-dir data
cd ../exercice_2 && python verify_ex2.py --data data/classification_spiral3.csv
```

## Exécution
Exercice 1 (budget complet : 2000 époques en lot complet, 300 en mini-lots) :
```bash
cd exercice_1
python ex1_regression.py --data-dir data --out results/full
```
Exercice 2 :
```bash
cd exercice_2
python ex2_classification.py --data data/classification_spiral3.csv --out results/ex2_full
```
Exercice 3 : ouvrir `exercice_3/TP_optimisation_ex3.ipynb` (Jupyter ou Google Colab) et
exécuter toutes les cellules. Les fichiers `results_raw.csv` et `results_summary.csv` sont
écrits dans le dossier de sortie du notebook.

Un test rapide (budgets réduits, une graine) est possible avec l'option `--quick`.

## Données
- CSV fournis dans `exercice_1/data`, `exercice_2/data`, `exercice_3/data`
  (`clustering_wine.csv` est généré par le notebook via scikit-learn s'il est absent).
- Pour les images (exercice 3) : placer `mnist_50k_10k_10k.npz` et
  `fashion_mnist_50k_10k_10k.npz` dans `exercice_3/data/` (fichiers de l'archive de la fiche).
  À défaut, le notebook tente de les reconstruire avec `tensorflow.keras.datasets`.

## Méthode
- Séparation train / validation / test ; standardisation calculée sur le train seulement.
- Hyperparamètres choisis sur la validation ; le test sert uniquement à l'évaluation finale.
- Graines 42, 123 et 2024 ; résultats en moyenne ± écart type.
- Gradients vérifiés par différences finies centrales (h = 1e-5).

## Résultats principaux
Voir le rapport PDF pour les tableaux complets.
- Exercice 1 : Ridge à noyau RBF avec Adam retenu sur la validation pour nonlinear et diabetes.
- Exercice 2 : spirales uniquement ; toutes les méthodes dépassent 0.98 d'accuracy ;
  Adam a la meilleure moyenne, mais l'écart est d'environ une observation sur 90.
- Exercice 3 : blobs identique pour toutes les méthodes ; wine ARI ≈ 0.90 ; images :
  résultats préliminaires (≈ 8 époques), ARI entre 0.33 et 0.36.

## Limites
Exercice 2 limité aux spirales avec une architecture différente de la fiche ;
exercice 3 sur images à budget court ; trois graines seulement.
