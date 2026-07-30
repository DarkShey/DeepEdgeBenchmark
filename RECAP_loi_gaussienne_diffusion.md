# La loi gaussienne dans TSDiff — récap vérifié

| Rôle | Librairie / appel | Fichier:ligne | En boucle ? | Training/Validation |
|---|---|---|---|---|
| Bruit de diffusion ε appris par le réseau | `torch.randn_like(residual)` | `models/tsdiff_model.py:336` | Oui — `for epoch` (l.328) × `for batch` (l.330) | Training |
| Latent initial du sampling DDIM | `torch.randn(n_samples, horizon, 1)` | `models/tsdiff_model.py:363` | Non — un seul tirage avant la boucle DDIM | Validation |
| Bruit réinjecté à chaque pas DDIM | `torch.randn_like(x)` | `models/tsdiff_model.py:377` | Oui — boucle `for i in range(k_denoise)` (l.365) | Validation |
| Intervalle final (non gaussien) | `np.quantile(price_samples, 0.025 / 0.975)` | `models/tsdiff_model.py:498-499` | Oui — boucle walk-forward `for i in range(len(test_p))` (l.493) | Validation |

**Conclusion :** pour TSDiff, la gaussienne (`torch`) n'est qu'un bruit interne au training et au sampling DDIM ; l'intervalle de prévision livré est **empirique** (quantiles `numpy`), sans hypothèse gaussienne.

---

**Écart relevé (à confirmer avec le tuteur, non corrigé ici) :** le §3 du brief (phrase sous le tableau) indique que la boucle walk-forward (l.493) appelle « `sample_next` l.496 ». En réalité, la ligne 496 fait partie de l'appel à `forecast_from_fitted(...)` (l.494-496), qui appelle en interne `model.sample_paths` (l.453) — pas `sample_next` directement (`sample_next` n'est utilisé que par `next_step_tsdiff`, hors boucle walk-forward). Le fond reste correct (le bruit gaussien est bien rejoué à chaque pas de validation), seul le nom de fonction cité est imprécis.
