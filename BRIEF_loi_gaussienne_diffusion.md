# BRIEF — La loi gaussienne dans le modèle de diffusion (TSDiff)

> Objectif : préparer un **tableau récap léger, sans données parasites**, à envoyer au tuteur,
> répondant à : *quelle librairie fournit la loi gaussienne, où (ligne de code), et est-elle
> appelée en boucle / au training / à la validation ?* — avec un **focus sur le modèle de
> diffusion TSDiff**.
>
> Document de référence : `Stage Unifox/Loi_Gaussienne_vs_Empirique.pdf` (22 + 24 juillet 2026).
> Code de référence : `models/tsdiff_model.py` (repo DeepEdgeBenchmark).

## 0. Le point clé à retenir

Pour TSDiff, la gaussienne **n'est pas la loi de sortie**. C'est le **seul modèle qui échappe
complètement à la gaussienne** pour son intervalle de prévision (cf. §2.3 du PDF). La gaussienne
n'apparaît que **à l'intérieur du mécanisme de diffusion** (le bruit DDPM/DDIM), et pas du tout
comme hypothèse sur la distribution prédictive finale, qui est **empirique** (nuage de
trajectoires simulées, quantiles 2,5 % / 97,5 %).

C'est exactement ce qui distingue TSDiff des 4 autres modèles (ARIMA-GARCH, SARIMA, Prophet, LSTM,
Naive), où la gaussienne **est** l'hypothèse qui produit l'intervalle.

## 1. Est-ce que le doc `Loi_Gaussienne_vs_Empirique` « s'applique » à la diffusion ?

Oui, il **traite** explicitement le cas TSDiff, mais sa conclusion est qu'il en est **l'exception** :

- §2.3 « TSDiff : le seul cas qui échappe complètement à la gaussienne ».
- La mention « Gaussian noise » dans la docstring (`tsdiff_model.py`, l. 12) décrit le **bruit de
  diffusion** que le denoiser apprend à prédire pendant l'entraînement (formulation DDPM standard),
  **pas** la loi de sortie.
- La distribution prédictive finale est explicitement **non gaussienne** (l. 18-19 : *« a genuine
  predictive distribution rather than a Gaussian residual band »*).

Autrement dit : l'« Usage 1 » du doc (gaussienne → IC95) **ne s'applique pas** à TSDiff. Seule une
note de vigilance s'y applique : ne pas confondre le bruit interne de diffusion avec une hypothèse
gaussienne sur la sortie.

## 2. Quelle librairie fournit la loi gaussienne pour TSDiff ?

**PyTorch** (`torch`), et non `scipy` / `statsmodels` / `arch` / `prophet` comme pour les autres
modèles, ni une constante `1.96` codée en dur comme LSTM/Naive.

- La gaussienne = **loi normale standard N(0,1)** tirée par `torch.randn` / `torch.randn_like`.
- Elle sert de **bruit du processus de diffusion**, à deux endroits : l'entraînement (le bruit ε
  que le réseau apprend à prédire) et l'échantillonnage DDIM (le latent initial + le bruit
  stochastique réinjecté à chaque pas).
- La **sortie** (l'intervalle) n'est **pas** gaussienne : elle vient de `numpy.quantile` sur le
  nuage de prix simulés.

> À ne PAS mettre dans le tableau (données parasites) : `torch.randn` aux lignes 219-220
> (`trend_w`, `season_w`) — ce sont de simples **initialisations de poids**, sans rapport avec la
> loi gaussienne de prévision.

## 3. Tableau récap (focus TSDiff) — à faire vérifier ligne par ligne

Numéros de ligne **indicatifs** (relevés le 29/07/2026) ; à revalider dans le code actuel avant
envoi (le PDF datant du 22-24/07 a déjà des lignes légèrement décalées).

| Rôle de la gaussienne | Librairie / appel | Fichier : ligne | Boucle ? | Training / Validation |
|---|---|---|---|---|
| Bruit de diffusion ε appris par le réseau | `torch.randn_like(residual)` | `tsdiff_model.py:336` | **Oui** — double boucle epochs × batches (l. 328 & 330) | **Training** |
| Latent initial (bruit pur) du sampling DDIM | `torch.randn(n_samples, horizon, 1)` | `tsdiff_model.py:363` | Une fois par appel de `sample_paths` | **Validation** (et prévision live) |
| Bruit stochastique réinjecté à chaque pas DDIM | `torch.randn_like(x)` | `tsdiff_model.py:377` | **Oui** — boucle de débruitage `k_denoise` (l. 365) | **Validation** (et prévision live) |
| Intervalle final — **non gaussien** | `np.quantile(price_samples, 0.025 / 0.975)` | `tsdiff_model.py:498-499` (et `542-543`) | **Oui** — boucle walk-forward (l. 493) | **Validation** |

Lecture : la seule vraie « loi gaussienne » de TSDiff est du **bruit torch**, présent au training
(l. 336) et pendant le sampling (l. 363, 377), lui-même appelé dans la boucle de validation
walk-forward (l. 493 → `sample_next` l. 496). L'intervalle envoyé au dashboard, lui, est
**empirique** (quantiles numpy), **sans hypothèse de forme**.

## 4. Est-ce appelé en boucle ? Au training ? À la validation ?

Oui aux trois, mais à des niveaux distincts :

- **Training** — boucle imbriquée `for epoch` (l. 328) × `for batch` (l. 330) : `torch.randn_like`
  (l. 336) tire un nouveau bruit gaussien **à chaque batch de chaque epoch**.
- **Sampling / inférence** — boucle de débruitage DDIM `for i in range(k_denoise)` (l. 365) :
  bruit réinjecté à chaque pas (l. 377), après un tirage initial (l. 363).
- **Validation** — boucle walk-forward `for i in range(len(test))` (l. 493) : à chaque pas de
  validation on rappelle le sampling (l. 496), donc les tirages gaussiens ci-dessus sont rejoués
  **à chaque jour** de la fenêtre de validation. Mais le résultat exposé (l'intervalle) est
  reconstruit par **quantiles empiriques** (l. 498-499), pas par la gaussienne.

## 5. Contraste utile pour le tuteur (contexte, hors tableau principal)

Si le tuteur veut situer TSDiff par rapport aux autres, une ligne suffit :

- 4 modèles gaussiens : la gaussienne **est** la sortie (via `arch`, `statsmodels`, `prophet`, ou
  `1.96×σ` codé main) → intervalle analytique.
- TSDiff : la gaussienne est **un ingrédient interne** (bruit torch de diffusion) → intervalle
  **empirique** (quantiles numpy), sans hypothèse de forme.

## 6. Points de vigilance pour la vérification

1. **Revérifier les numéros de ligne** dans `models/tsdiff_model.py` actuel — ceux du PDF ont
   déjà bougé (ex. Naive : PDF donne l. 130-131, le code réel est l. 138-139).
2. **Exclure les faux positifs** : `torch.randn` l. 219-220 = init de poids, à ne pas confondre
   avec le bruit de diffusion.
3. **Ne pas surcharger** : le tuteur veut un tableau léger. 4 lignes suffisent (les 3 tirages
   gaussiens + la sortie empirique). Pas de RMSE, pas de jargon DDPM au-delà du strict nécessaire.

## 7. Ce qu'on demande à Claude Code

Voir `MESSAGE_claude_code_loi_gaussienne_diffusion.md` : vérifier ce tableau ligne par ligne
contre le code actuel, corriger tout numéro décalé, et sortir **une seule** version propre et
légère du tableau (Markdown) prête à envoyer.
