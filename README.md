# BEC_in_optical_lattice
Implémentation Python d'algorithmes GRAPE pour le contrôle optimal quantique, appliqué à un condensat de Bose-Einstein dans un réseau optique à une ou deux dimensions. Ce travail a été effectué dans le cadre d'un stage de recherche à l'Institut Carnot de Bourgogne. 


Le langage utilisé est Python. Les bibliothèques utilisées sont *numpy*, *scipy*, *matplotlib*, et parfois *cupy*, *qutip*. L'affichage utilise également certains packages Latex. Pour les algorithmes dans un réseau 2D, le dossier *data* contient un contrôle optimal pré-calculé - qui nécessite environ 6h d'optimisation - ainsi que les populations finales simulées pour différentes valeurs de $\lambda$ et $\theta$.

* **`1D_fingerprinting.py`** : Fingerprinting de condensats de Bose-Einstein dans un réseau à une dimension, selon l'algorithme GRAPE.  
  Le programme est composé de trois classes : une classe System définissant un système BEC, une classe Propagation liée à une échelle de temps et permettant de calculer la fonction d'onde, l'adjoint et les gradients pour chaque pas de temps. La classe OptimalControl définit la fonction coût, gradient et accepte un nombre quelconque de système. Le programme permet de définir un BEC puis de trouver le contrôle optimal en suivant l'algorithme GRAPE.    
  Utilise le module *qutip* pour la figure représentant la sphère de Bloch.

* **`1D_maxQFI.py`** : Maximisation de la QFI dans un réseau 1D.  
  A la différence du fingerprinting, on propage la fonction d'onde étendue $(\vert{}\psi\rangle, \partial\vert{}\psi\rangle/\partial\lambda)$. Les fonctions de coût et gradients sont également généralisés.

* **`2D_state_to_state.py`** : Transfert d'état à état dans un réseau à deux dimensions.  
  Ce programme adapte les 3 classes définissant le système, la propagation et le contrôle en utilisant des matrices sparses et un propagateur exact. Il peut aisément se développer en programme implémentant le fingerprinting sur CPU en ajoutant une force $\vec\lambda$.

* **`2D_fingerprinting_3systems.py`** : Algorithme GRAPE appliqué à 3 systèmes BEC pour implémenter la méthode de fingerprinting dans un réseau bidimensionnel.  
  Le code est conçu pour être exécuter à l'aide d'une carte graphique nvidia (avec l'outil cuda), et est vectorisé au maximum. Réduire la précision des flottants et des complexes détériorent les résultats car les fonctions d'onde ne sont plus normées.  
  Les simulations sont faites avec max_iter$=10000$, d'une durée de $5$ à $10$h, et le contrôle optimal est sauvegardé.  
  Le calcul des états finaux pour différents $\lambda$ et $\theta$ est ensuite effectué puis sauvegardé.  
  Les variables *new_optimisation* et *new_plots* permettent de choisir si une nouvelle optimisation est lancée.  
  Depuis ce code, il est très facile d'implémenter des configurations à deux systèmes (il suffit de n'en définir que deux), de faire du transfert d'état à état (il suffit de définir un seul système et mettre $\lambda=0$).  
  Ce programme nécessite CUDA (i.e. une carte graphique nvidia).

* **`2D_fingerprinting_4systems.py`** : Algorithme GRAPE appliqué à 4 systèmes BEC pour implémenter la méthode de fingerprinting dans un réseau bidimensionnel.
  Légère adaptation (notamment pour la partie affichage) pour le fingerprinting à 4 systèmes.  
    Ce programme nécessite CUDA (i.e. une carte graphique nvidia).
