"""
Script simple pour tester la Reactive Layer sur le dataset fusion
Sans modifier la structure existante

CONFIGURATION DU SEUIL DE CONFIANCE:
====================================
Par défaut, la ReactiveLayer utilise un seuil de 0.75.
Pour changer le seuil, modifier la ligne suivante dans __main__:

    reactive = ReactiveLayer(confidence_threshold=0.80)  # ← Changer ici

Seuils recommandés:
- 0.70: Maximum de sécurité (100% détection, 48% fausses alertes)
- 0.80: ÉQUILIBRE OPTIMAL (98% détection, 35% fausses alertes) ← RECOMMANDÉ
- 0.85: Moins de fausses alertes (95% détection, 25% fausses alertes)

Voir: REACTIVE_LAYER_OPTIMIZATION.md pour plus de détails
"""
from pathlib import Path

import pandas as pd
from core.agent.ReactiveLayer import ReactiveLayer, CriticalAlert


class ReactiveTester:
    """
    Testeur simple pour la Reactive Layer
    """

    def __init__(self, reactive_layer):
        self.reactive = reactive_layer
        self.results = []
        self.results_df = None  # Pour stocker les résultats après test

    def convert_row(self, row):
        """
        Convertit une ligne du dataset en format vehicle_state
        Adapté à vos colonnes exactes
        """
        # Structure de base
        vehicle_state = {
            "speed": float(row['Speed']),
            "speed_limit": 90,  # Valeur par défaut si non disponible
            "hard_brake": row.get('hard_brake', 0),  # Ajout pour danger_zone
            "leader_stopped": row.get('leader_stopped', 0),  # Ajout pour danger_zone
            "danger_accel_toward_leader": row.get('danger_accel_toward_leader', 0),  # Ajout
            "no_reaction_to_stopped_leader": row.get('no_reaction_to_stopped_leader', 0),  # Ajout
            "TTC": row.get('TTC', float('inf')),  # Ajout
            "LeaderGap": row.get('LeaderGap', float('inf')),  # Ajout
        }

        # Vérifier si on a un leader (LeaderGap existe et > 0)
        if pd.notna(row['LeaderGap']) and row['LeaderGap'] > 0:
            vehicle_state["leading_vehicle"] = {
                "distance": float(row['LeaderGap']),
                "speed": float(row['LeaderSpeed']) if pd.notna(row['LeaderSpeed']) else float(row['Speed']),
                "acceleration": float(row['Acceleration']) if pd.notna(row['Acceleration']) else 0
            }
        else:
            vehicle_state["leading_vehicle"] = {}

        return vehicle_state

    def test_on_dataset(self, df, max_rows=None):
        """
        Teste la reactive layer sur le dataset
        """
        # Prendre un échantillon si demandé
        if max_rows and max_rows < len(df):
            df_test = df.sample(n=max_rows, random_state=42)
            print(f"Test sur {max_rows} lignes (échantillon aléatoire)")
        else:
            df_test = df
            print(f"Test sur la totalité du dataset ({len(df)} lignes)")

        # Réinitialiser les stats
        self.reactive.reset_stats()
        self.results = []

        # Tester chaque ligne
        total = len(df_test)
        print(f"Progression: 0/{total}", end='')

        for idx, (_, row) in enumerate(df_test.iterrows()):
            # Afficher progression
            if (idx + 1) % 1000 == 0:  # Modifié pour moins de messages
                print(f"\rProgression: {idx + 1}/{total}", end='')

            # Convertir et tester
            state = self.convert_row(row)
            alert = self.reactive.check_immediate_danger(state)

            # Enregistrer avec toutes les colonnes utiles
            self.results.append({
                'index': idx,
                'has_alert': alert is not None,
                'alert_type': alert.type if alert else None,
                'severity': alert.severity if alert else None,
                'speed': row['Speed'],
                'leader_gap': row['LeaderGap'] if pd.notna(row['LeaderGap']) else None,
                'ttc': row['TTC'] if pd.notna(row['TTC']) else None,
                'hard_brake': row['hard_brake'] if 'hard_brake' in row else 0,
                'safe_danger': row['SafeDangerZone'] if 'SafeDangerZone' in row else 0,
                'is_accident': row['IsAccidentCase'] if 'IsAccidentCase' in row else 0,  # AJOUT IMPORTANT
                'leader_stopped': row['leader_stopped'] if 'leader_stopped' in row else 0,
                'danger_accel': row['danger_accel_toward_leader'] if 'danger_accel_toward_leader' in row else 0,
            })

        print(f"\rProgression: {total}/{total} - Terminé!")

        # Convertir en DataFrame pour analyse
        self.results_df = pd.DataFrame(self.results)
        return self.results_df

    def show_results(self):
        """
        Affiche les résultats de façon simple
        """
        if not hasattr(self, 'results_df') or self.results_df is None:
            print("Lancez d'abord test_on_dataset()")
            return

        df = self.results_df
        alerts = df[df['has_alert']]

        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DU TEST")
        print("=" * 60)

        # Stats générales
        print(f"\n📈 Statistiques:")
        print(f"  • Total tests: {len(df)}")
        print(f"  • Alertes générées: {len(alerts)}")
        print(f"  • Taux d'alerte: {len(alerts) / len(df) * 100:.2f}%")

        # Types d'alertes
        if len(alerts) > 0:
            print(f"\n🚨 Types d'alertes:")
            alert_counts = alerts['alert_type'].value_counts()
            for alert_type, count in alert_counts.items():
                print(f"  • {alert_type}: {count} ({count/len(alerts)*100:.1f}%)")

        # Performance
        report = self.reactive.get_performance_report()
        print(f"\n⏱️ Performance:")
        print(f"  • Temps moyen: {report['avg_response_time']:.2f} ms")
        print(f"  • Temps max: {report['max_response_time']:.2f} ms")
        print(f"  • SLA (<50ms): {'✅' if report['meets_sla'] else '❌'}")

        # Comparaison avec SafeDangerZone
        if 'safe_danger' in df.columns:
            self._compare_with_safe_danger(df)

        # Comparaison avec IsAccidentCase (NOUVEAU)
        if 'is_accident' in df.columns:
            self.compare_with_accidents()
        else:
            print("\n❌ Colonne IsAccidentCase non trouvée dans les résultats")

    def _compare_with_safe_danger(self, df):
        """Compare avec SafeDangerZone"""
        if df['safe_danger'].notna().any():
            print(f"\n🔍 Comparaison avec SafeDangerZone:")
            true_danger = df[df['safe_danger'] == 1]
            if len(true_danger) > 0:
                detected = len(true_danger[true_danger['has_alert']])
                print(f"  • Zones dangereuses: {len(true_danger)}")
                print(f"  • Détectées: {detected}")
                print(f"  • Non détectées: {len(true_danger) - detected}")
                print(f"  • Taux de détection: {detected / len(true_danger) * 100:.1f}%")

    def compare_with_accidents(self):
        """
        Compare les alertes de la reactive layer avec les vrais cas d'accidents
        Version simplifiée - sans affichage détaillé des cas
        """
        if not hasattr(self, 'results_df') or self.results_df is None:
            print("❌ Lancez d'abord test_on_dataset()")
            return

        df = self.results_df

        print("\n" + "=" * 60)
        print("📊 COMPARAISON AVEC LES CAS D'ACCIDENTS RÉELS (IsAccidentCase)")
        print("=" * 60)

        # Vérifier si IsAccidentCase existe
        if 'is_accident' not in df.columns:
            print("❌ La colonne is_accident n'est pas dans les résultats")
            return

        # Statistiques générales
        total_accidents = df['is_accident'].sum()
        total_non_accidents = len(df) - total_accidents

        print(f"\n📈 Statistiques générales:")
        print(f"  • Total lignes: {len(df)}")
        print(f"  • Cas d'accidents (IsAccidentCase=1): {int(total_accidents)}")
        print(f"  • Cas non-accidents: {int(total_non_accidents)}")

        if total_accidents == 0:
            print("\n⚠️ Aucun cas d'accident dans le dataset testé")
            return

        # Séparer les données
        accidents = df[df['is_accident'] == 1]
        non_accidents = df[df['is_accident'] == 0]

        # Pour les accidents
        accidents_alertes = accidents[accidents['has_alert'] == True]
        accidents_sans_alerte = accidents[accidents['has_alert'] == False]

        # Pour les non-accidents
        non_accidents_alertes = non_accidents[non_accidents['has_alert'] == True]
        non_accidents_sans_alerte = non_accidents[non_accidents['has_alert'] == False]

        # Calcul des métriques
        tp = len(accidents_alertes)  # Vrais positifs
        fn = len(accidents_sans_alerte)  # Faux négatifs (accident non détecté)
        fp = len(non_accidents_alertes)  # Faux positifs (fausse alerte)
        tn = len(non_accidents_sans_alerte)  # Vrais négatifs

        print(f"\n🔍 MATRICE DE CONFUSION:")
        print(f"  {'':25} {'Accident réel':>15} {'Non-accident':>15}")
        print(f"  {'-' * 55}")
        print(f"  {'Alerte générée':25} {tp:>15} {fp:>15}")
        print(f"  {'Pas d\'alerte':25} {fn:>15} {tn:>15}")

        # Taux de détection (Recall) - Le plus important pour la sécurité
        recall = (tp / total_accidents) * 100 if total_accidents > 0 else 0
        print(f"\n🎯 TAUX DE DÉTECTION DES ACCIDENTS: {recall:.2f}%")

        # Précision
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        print(f"🎯 PRÉCISION: {precision:.2f}%")

        # F1-Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"📊 F1-SCORE: {f1:.2f}%")

        # Recommandation simple
        print(f"\n💡 RECOMMANDATION:")
        if recall < 90:
            print(f"  • 🔴 Taux de détection trop faible ({recall:.1f}%) → Rendre plus sensible")
        elif fp > 1000:
            print(f"  • 🟡 Trop de fausses alertes ({fp}) → Rendre plus sélectif")
        else:
            print(f"  • ✅ Bon équilibre détection/précision")

        return {
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'recall': recall, 'precision': precision, 'f1': f1
        }
    def show_examples(self, n=5):
        """
        Montre quelques exemples d'alertes
        """
        if not hasattr(self, 'results_df') or self.results_df is None:
            return

        alerts = self.results_df[self.results_df['has_alert']]
        if len(alerts) == 0:
            print("\n❌ Aucune alerte à afficher")
            return

        print(f"\n🔍 Exemples d'alertes (sur {len(alerts)}):")
        print("-" * 80)

        for i, (_, row) in enumerate(alerts.head(n).iterrows()):
            print(f"Exemple {i + 1}:")
            print(f"  • Type: {row['alert_type']}")
            print(f"  • Sévérité: {row['severity']}")
            print(f"  • Vitesse: {row['speed']:.1f} km/h")
            if pd.notna(row['leader_gap']):
                print(f"  • Distance leader: {row['leader_gap']:.1f} m")
            if pd.notna(row['ttc']):
                print(f"  • TTC: {row['ttc']:.2f} s")
            print(f"  • Est un accident: {'✅' if row['is_accident'] == 1 else '❌'}")
            print("-" * 40)


def resolve_dataset_path(file_arg):
    """
    Résout le chemin du dataset de façon robuste.
    Priorité :
    1. chemin fourni tel quel s'il existe
    2. chemin relatif au dossier du script
    """
    candidate = Path(file_arg)

    if candidate.is_file():
        return candidate.resolve()

    script_dir = Path(__file__).resolve().parent
    candidate_from_script = script_dir / file_arg
    if candidate_from_script.is_file():
        return candidate_from_script.resolve()

    raise FileNotFoundError(
        f"Dataset introuvable: '{file_arg}'. "
        f"Chemins testés: '{candidate.resolve()}' et '{candidate_from_script.resolve()}'."
    )


# ============================================
# EXÉCUTION PRINCIPALE
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Reactive Layer')
    parser.add_argument('--file', type=str, default='Fusion_dataset.csv',
                        help='Chemin vers le dataset')
    parser.add_argument('--rows', type=int, default=None,
                        help='Nombre de lignes à tester (optionnel)')

    args = parser.parse_args()

    # 1. Charger le dataset
    try:
        dataset_path = resolve_dataset_path(args.file)
        print(f"\n📂 Chargement de {dataset_path}...")
        df = pd.read_csv(dataset_path)
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}")
        raise SystemExit(1)

    print(f"✅ Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"Colonnes: {list(df.columns)}")

    # 2. Créer la reactive layer
    print(f"\n🤖 Initialisation de la Reactive Layer...")
    reactive = ReactiveLayer()

    # 3. Créer le testeur
    tester = ReactiveTester(reactive)

    # 4. Lancer les tests
    print(f"\n🚀 Lancement des tests...")
    results = tester.test_on_dataset(df, max_rows=args.rows)

    # 5. Afficher les résultats
    tester.show_results()
    tester.show_examples()