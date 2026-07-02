from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict


@dataclass
class RegimeState:
    """
    Contrat de sortie du moteur de régime DEITA.
    Produit par RegimeHMM (ce module) et complété par BOCPD (Kyrio).
    Consommé par : calibration de prédiction (conformal), calibration de risque (sizing).
    """

    # ── Produit par HMM (ce module) ──────────────────────────────────────────
    probs: Dict[str, float]
    # Distribution a posteriori sur les états. Clés exactes : calm, bull, bear, stress.
    # Somme = 1.0 à 1e-6 près.
    # Exemple : {"calm": 0.55, "bull": 0.10, "bear": 0.10, "stress": 0.25}

    vol_bucket: int
    # Entier 0, 1 ou 2.
    # Calculé par terciles de σ_t GARCH sur l'historique d'entraînement.
    # 0 = volatilité faible (tiers inférieur), 1 = modérée, 2 = élevée (tiers supérieur).
    # MÊME bucket consommé par conformal et sizing — ne jamais redéfinir ailleurs.

    stress_score: float
    # = probs["stress"]. Raccourci pour les couches de risque. Valeur dans [0, 1].

    expected_duration_days: float
    # Durée moyenne estimée du régime dominant courant, en jours.
    # Calculée depuis la matrice de transition HMM :
    #   state_idx = argmax(probs.values())
    #   expected_duration_days = 1 / (1 - transmat_[state_idx, state_idx])

    as_of: datetime
    # Date de calcul. Garantit la contrainte point-in-time :
    # toutes les données utilisées sont strictement antérieures à as_of.

    version: str
    # Identifiant de version de l'algorithme. Valeur fixe : "hmm-garch-adx-v2"

    # ── Complété par BOCPD (Kyrio) — valeurs par défaut ──────────────────────
    changepoint_prob: float = 0.0
    # Probabilité bayésienne qu'un changement de régime se produise à as_of.
    # Rempli par le module BOCPD de Kyrio. Par défaut 0.0.

    is_transitioning: bool = False
    # True si changepoint_prob > seuil (défini par Kyrio, typiquement 0.5).
    # Rempli par le module BOCPD de Kyrio. Par défaut False.

    def dominant_regime(self) -> str:
        """Retourne le nom du régime dominant (argmax des probs)."""
        return max(self.probs, key=self.probs.get)

    def validate(self) -> None:
        """
        Vérifie la cohérence interne du RegimeState.
        Lève ValueError si une contrainte est violée.
        """
        if set(self.probs.keys()) != {"calm", "bull", "bear", "stress"}:
            raise ValueError(f"probs doit contenir exactement calm/bull/bear/stress, got {set(self.probs.keys())}")
        if abs(sum(self.probs.values()) - 1.0) > 1e-5:
            raise ValueError(f"probs ne somme pas à 1 : {sum(self.probs.values())}")
        if self.vol_bucket not in (0, 1, 2):
            raise ValueError(f"vol_bucket doit être 0, 1 ou 2, got {self.vol_bucket}")
        if not (0.0 <= self.stress_score <= 1.0):
            raise ValueError(f"stress_score hors [0,1] : {self.stress_score}")
        if not (0.0 <= self.changepoint_prob <= 1.0):
            raise ValueError(f"changepoint_prob hors [0,1] : {self.changepoint_prob}")
        if self.expected_duration_days <= 0:
            raise ValueError(f"expected_duration_days doit être > 0")
        if abs(self.stress_score - self.probs["stress"]) > 1e-6:
            raise ValueError("stress_score doit être égal à probs['stress']")
