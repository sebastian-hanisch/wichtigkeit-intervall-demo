"""Konstanten: Generator (kopiert aus verweilzeit-demo, erfunden), Raster, Regler-Grenzen, Namen, Farben und Presets.

Alle Merkmale, die Verweilzeit und die Größenordnungen sind ERFUNDEN (synthetisch, keine reale Datenquelle)."""

# --- Generator (Formeln und Anteile aus verweilzeit-demo/vwz_constants.py kopiert; die Kopie prüft tests/test_generator.py) -----------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
# Kundeneffekte: numpy.random.default_rng(12345).normal(0, 0,30, 12) aus verweilzeit-demo, hier als feste Werte (der Test vergleicht mit numpy)
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
U_SCALE = (1.0, 0.3, 0.3, 1.0)               # Kundeneffekt je Art
S_BASE = (0.55, 0.25, 0.35, 0.90)            # Streuung der log-Verweilzeit je Art
P_ZOLL = 0.25                                # Zollhalt bei Import
P_WOCHENENDE = 2 / 7
VORLAUF_EXPORT = (1.0, 10.0)                 # Träger des Vorlaufs je Art
VORLAUF_UMSCHLAG = (0.5, 6.0)
FEATURES = ("art", "kunde", "zoll", "wochenende", "vorlauf", "gewicht", "zwilling")
FEATURE_LABELS = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht", "Zwilling")
D = 6                                        # Merkmale des Hauptexperiments (Spalten 0 bis 5)
D_TWIN = 7                                   # mit dem Zwilling des Vorlaufs (Spalte 6: Vorlauf + N(0; 0,05²), von der wahren Funktion nicht gelesen)
TWIN_SD = 0.05
COL_ART, COL_VORLAUF, COL_GEWICHT, COL_TWIN = 0, 4, 5, 6

COL_NOISE_FEATURES = (5,)                    # Gewicht: reines Rauschen, Wirkung null
EFFECT_FEATURES = (0, 1, 2, 3, 4)            # Merkmale mit Wirkung auf mu (Mittel über diese fünf in den Abdeckungstabellen)

# --- Verfahren, Klassen, Raster (tools/wik_make_grid.py rechnet es, data/wik_grid.json friert es ein) ------------------------------------------
METHODS = ("perm", "cperm", "loco")
METHOD_LABELS = {"perm": "Permutation", "cperm": "Bedingte Permutation", "loco": "LOCO (Merkmal weglassen, neu trainieren)"}
CLASSES = ("gbm", "rf", "mlp")
CLASS_LABELS = {"gbm": "Boosting", "rf": "Random Forest", "mlp": "Kleines Netz"}
N_TRAIN = 1000                               # Trainingszeilen je Zufalls-Seed (nur diese Datenmenge ist gerechnet)
TEST_SIZES = (300, 1000, 3000)               # Testzeilen: Anfänge desselben, je Zufalls-Seed neu gezogenen Testsatzes
N_SEEDS = 100                                # Zufalls-Seeds (Neu-Trainings mit eigenem Testsatz) je Klasse
R_PERM = 3                                   # Umordnungen je Merkmal
N_BOOT = 300                                 # Bootstrap-Neuziehungen der Testzeilen
POP_ROWS = 30_000                            # Bevölkerung für das Ziel „Wichtigkeit des Modells“ (feste Zeilen, für alle Zufalls-Seeds gleich)
TRUTH_ROWS = 300_000                         # Bevölkerung der Wahrheit je Verfahren (Permutation, bedingte Permutation)
LOCO_TRUTH_ROWS = 20_000                     # LOCO-Wahrheit exakt aus der bedingten Erwartung, gemittelt über so viele Zeilen
INTERVAL_KINDS = ("B", "N")
INTERVAL_LABELS = {"B": "Bootstrap-Perzentile (B)", "N": "Normalnäherung (N)"}
TARGETS = ("pop", "tmean", "truth")
TARGET_LABELS = {"pop": "Wichtigkeit des Modells", "tmean": "Mitte über Neu-Trainings", "truth": "Wahrheit je Verfahren"}
NOMINAL = 0.90
TWIN_STATS = ("perm", "cperm", "loco1", "loco_group")   # Zwillings-Experiment (loco_group lässt Vorlauf und Zwilling zusammen weg)
TWIN_LABELS = {"perm": "Permutation", "cperm": "Bedingte Permutation (Schicht Art)", "loco1": "LOCO einzeln", "loco_group": "LOCO ohne beide (Gruppe)"}

# --- Zufallsströme (SplitMix64, je Verwendung ein eigener Bereich) -------------------------------------------------------------------------------
SEED_TRAIN, SEED_TEST = 600_000, 700_000     # + Zufalls-Seed der Zelle
SEED_POP, SEED_TRUTH = 800_000, 810_000
SEED_PERM, SEED_BOOT = 830_000, 840_000

# --- Regler und Presets ------------------------------------------------------------------------------------------------------------------------
SEED_MAX = N_SEEDS
COLOR_MODEL = "#4c78a8"
COLOR_TRUTH = "#2F6B65"
COLOR_BAD = "#e45756"
COLOR_GRAY = "rgba(150,150,150,0.55)"
PRESETS = {
    "Boosting, 1000 Zeilen, Permutation": {"klasse": "gbm", "verfahren": "perm", "merkmal": 0, "art": "B", "testzeilen": 1000, "seed": 1},
    "Random Forest, bedingte Permutation": {"klasse": "rf", "verfahren": "cperm", "merkmal": 0, "art": "B", "testzeilen": 1000, "seed": 1},
    "Netz, LOCO": {"klasse": "mlp", "verfahren": "loco", "merkmal": 0, "art": "B", "testzeilen": 1000, "seed": 1},
    "3000 Testzeilen (enges Intervall)": {"klasse": "gbm", "verfahren": "perm", "merkmal": 4, "art": "B", "testzeilen": 3000, "seed": 1},
    "Zwillingsmerkmal": {"klasse": "gbm", "verfahren": "perm", "merkmal": 4, "art": "N", "testzeilen": 1000, "seed": 1},
}
PRESET_HELP = {
    "Boosting, 1000 Zeilen, Permutation": "Boosting mit 1000 Trainingszeilen: das Intervall um die Permutationswichtigkeit der Art, gegen drei Ziele.",
    "Random Forest, bedingte Permutation": "Random Forest, bedingte Permutation (Umordnen innerhalb der Schichten): deckt das Intervall die wahre Wichtigkeit?",
    "Netz, LOCO": "Kleines Netz, LOCO (Merkmal weglassen, neu trainieren).",
    "3000 Testzeilen (enges Intervall)": "Mehr Testzeilen machen das Intervall enger, aber nicht verlässlicher für die Frage nach einem Neu-Training (Vorlauf, Permutation).",
    "Zwillingsmerkmal": "Vorlauf mit Normalnäherung; das Experiment zu zwei Spalten, die dasselbe messen, steht weiter unten.",
}
