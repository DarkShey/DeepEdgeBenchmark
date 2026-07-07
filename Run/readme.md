Pour chaque Run, dans ce folder créer un sous-folder avec comme format <Date YYYYMMDD>-<nom du modèle>-<asset-horizon> (rappel: si on veut faire une prédiction en daily, le training est en daily)
Là-dedans, copier une copie de:
- output de trainings (le plus important) -> cf le doc de Kyrio (dans le suite 'Data REadiness')
- les tests effectués en format XLS. 2 types de test:
               - ceux effectués par Claude lors du Run (smoke test essentiellement) -> demander à Claude de documenter ces tests dans un fichier xls
               - ceux décris par vous -> Test Cases
      . ajouter les autres outputs: graph (.png) etc...