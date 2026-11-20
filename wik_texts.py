"""Texte der App, die auch in der README stehen (ein Test vergleicht beide Fassungen) oder Zahlen aus dem Raster einsetzen."""

import numpy as np

import wik_constants as C
from wik_evaluation import col, est_array, fmt, interval, pct

TITEL = "Wichtigkeit der Merkmale mit Intervall: wen deckt das Intervall?"

EINLEITUNG = """
Die **Wichtigkeit eines Merkmals** (global: um wie viel der Fehler steigt, wenn man das Merkmal stört oder weglässt) wird aus einer Testmenge geschätzt, und man gibt gern ein **Intervall** dazu.
Die Frage dieser Demo: **Wen deckt dieses Intervall?** Ein Intervall hat ein **Ziel**: die Wichtigkeit **dieses einen Modells**, die Wichtigkeit eines Modells aus 1000 Zeilen **im Mittel über Neu-Trainings**
oder die **wahre Wichtigkeit je Verfahren** (für die bekannte Funktion μ). Drei Verfahren stehen nebeneinander: **Permutation**, **bedingte Permutation** und **LOCO** (Merkmal weglassen, neu trainieren).

Die Daten sind ausgedacht: ein erfundener Generator für Container-Verweilzeiten, dessen wahre Mitte bekannt ist. Es gibt keine reale Datenquelle. In den gelesenen Quellen gab es keine Messung, ob Intervalle
um die Wichtigkeit von Merkmalen in Logistik, Lager oder Fertigung zu besseren Entscheidungen führen; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. Gezählt werden Zufalls-Seeds
(Neu-Trainings mit eigenem Testsatz), nicht Menschen oder Standorte. **Gemessen, kein Verfahren empfohlen:** die Demo zeigt, was die einfachen Intervalle tun, nicht welches man nehmen soll.

Weiter unten: die drei Kernfragen, vier Experimente (ein Neu-Training mit live gerechnetem Bootstrap; wen deckt das Intervall; mehr Testzeilen; zwei Spalten, die dasselbe messen),
**🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
""".strip()

SO_FUNKTIONIERT = """
1. **Wichtigkeit schätzen.** Je Testzeile $i$ wird die Zunahme des quadratischen Fehlers $d_i$ gerechnet: bei der **Permutation** wird die Spalte des Merkmals unter allen Testzeilen umgeordnet (drei Umordnungen,
   gemittelt), bei der **bedingten Permutation** nur innerhalb von Schichten (Art; für die Art selbst: Vorlauf größer als 0 gegen gleich 0), bei **LOCO** wird das Merkmal weggelassen und das Modell **neu trainiert**.
   Die Wichtigkeit ist das Mittel $\\bar d$ der $d_i$.
2. **Intervall bilden.** Aus den Testzeilen: **B** = Bootstrap-Perzentile (5. und 95. Perzentil von 300 Neuziehungen der Zeilen), **N** = Mittel ± 1,6449 Standardfehler. Beide kennen **nur die Auswahl der Testzeilen**,
   nicht das Training und nicht die Wahl der Umordnung.
3. **Abdeckung zählen.** Je Zelle (Klasse, Verfahren, Merkmal) wurden 100 Zufalls-Seeds trainiert, jeder mit eigener Historie und eigenem Testsatz. Die **Abdeckung** ist der Anteil der Zufalls-Seeds, deren Intervall das Ziel enthält
   (nominal 90 %).
""".strip()

FORMEL_LESEHILFE = """
**Lesehilfe.** Gegeben: ein trainiertes Modell $f$, Testzeilen $(x_i, y_i)$ mit $i = 1, \\dots, m$ und ein Merkmal $j$. Gesucht: ein Intervall für die Wichtigkeit von $j$ und die Frage, welches Ziel es deckt.

| Symbol | Bedeutung |
|---|---|
| $m$ | Zahl der Testzeilen (300, 1000, 3000) |
| $d_i$ | Zunahme des quadratischen Fehlers in Testzeile $i$ durch das Stören oder Weglassen des Merkmals |
| $\\bar d$ | Mittel der $d_i$: geschätzte Wichtigkeit |
| $s$ | Standardabweichung der $d_i$ (mit $m-1$) |
| $z$ | 1,6449 (zweiseitig 90 %) |
| $\\tau$ | Streuung der Modellwichtigkeit über Neu-Trainings |
| $\\mu$ | wahre Mitte der Zielgröße (bekannte Funktion des Generators) |
""".strip()

FORMEL = r"""
**Verlust und Wichtigkeit.** Mit dem quadratischen Fehler $\ell(f; x, y) = (f(x) - y)^2$:

$$d_i^{\mathrm{perm}} = \frac{1}{R}\sum_{r=1}^{R} \Bigl(\ell(f; \tilde x_i^{(r)}, y_i) - \ell(f; x_i, y_i)\Bigr), \qquad d_i^{\mathrm{LOCO}} = \ell\bigl(f_{-j}; x_i, y_i\bigr) - \ell(f; x_i, y_i)$$

mit $R = 3$ Umordnungen der Spalte $j$ ($\tilde x$: bei der bedingten Permutation nur innerhalb der Schichten) und dem ohne $j$ neu trainierten Modell $f_{-j}$. Die Wichtigkeit ist $\bar d = \frac{1}{m}\sum_i d_i$.

**Intervalle (90 %).**

$$\text{N: } \bar d \pm z\,\frac{s}{\sqrt{m}}, \qquad \text{B: } 5.\text{ und } 95.\text{ Perzentil von } \bar d^{*(1)}, \dots, \bar d^{*(300)} \text{ (Neuziehungen der Zeilen mit Zurücklegen)}$$

**Abdeckung der Mitte über Neu-Trainings (geschlossen).** Streut die Wichtigkeit des Modells über Neu-Trainings mit der Standardabweichung $\tau$ und schätzt das Intervall sie mit dem Standardfehler $\mathrm{se}$ (halbe Breite $z\,\mathrm{se}$), gilt mit $r = \tau / \mathrm{se}$

$$\mathrm{Abdeckung} = 2\,\Phi\!\left(\frac{z}{\sqrt{1 + r^2}}\right) - 1.$$

Das Verhältnis $V = \tau / (\text{halbe Breite})$ der Tabellen ist nicht dieses $r$: $r = z\,V$.

Das ist ein **vollständiges Modell der Rechnung dieser Demo** (Verlust, drei Verfahren, beide Intervalle, Abdeckung); die Formel von $\mu$ (Art, Kunde, Zoll, Wochenende, Vorlauf), die Verteilung der Merkmale und die
Modelleinstellungen stehen im Code (`wik_generator.py`, `tools/wik_make_grid.py`). Die **Wahrheit je Verfahren** ist dasselbe Verfahren auf die Funktion $\mu$ angewandt (Permutation und bedingte Permutation
aus 300 000 Zeilen, LOCO exakt aus der bedingten Erwartung: Verlustanstieg des besten Vorhersagers ohne das Merkmal).

**Quellen (Stufe).** Hooker, Mentch, Zhou (arXiv 1905.03151, Abstract gelesen): unbeschränktes Permutieren erzwingt Extrapolation. Strobl u. a. 2008 (BMC Bioinformatics 9:307, Abstract gelesen): bedingte Permutation.
Lei u. a. 2018 (JASA, Titel), Williamson u. a. 2023 (JASA 118:1645, Abstract), Watson und Wright (Abstract), Verdinelli und Wasserman (arXiv 2303.05981, Abstract): **nicht gelesen**, ihre Aussagen über die Gültigkeit von Intervallen
werden hier nicht übernommen, sondern nachgemessen. Keine Neuheitsaussage: alle drei Verfahren und die Intervalle sind Lehrbuch und Forschung; Vorbilder: Hooker/Mentch (Version 1 mit bekannter Funktion), XAI-Bench, OpenXAI.
""".strip()


def grenzen_markdown(grid: dict) -> str:
    """Abschnitt „Wo die Annahmen enden“; die Zahlen kommen aus dem Raster (die README enthält denselben Text, ein Test vergleicht)."""
    n = C.N_SEEDS
    noise = {}
    for method in C.METHODS:
        shares = []
        for kl in grid["meta"]["klassen"]:
            lo, _ = interval(grid, kl, 1, method, 5, "B")
            shares.append(float(np.mean(lo > 0)))
        noise[method] = (min(shares), max(shares))
    loco_neg = all(float(est_array(grid, kl)[:, 1, col("loco", 5), 0].mean()) < 0 for kl in grid["meta"]["klassen"])
    rows = [
        ("**Ein Vehikel, eine Trainingsgröße**", f"Erfundener Verweilzeit-Generator, sechs Merkmale, quadratischer Fehler, {C.N_TRAIN} Trainingszeilen; Boosting, Random Forest und kleines Netz mit festen Einstellungen. Andere Trainingsgrößen, "
                                                "andere Verluste (Klassifikation) und andere Einstellungen sind nicht gemessen."),
        ("**Intervall nur über die Testzeilen**", "Beide Intervalle (B und N) berücksichtigen nur die Auswahl der Testzeilen, nicht das Training und nicht die Wahl der Umordnung. Intervalle, die das Training einbeziehen "
                                                  "(Bootstrap mit Neu-Training, Stichprobenteilung, Williamson u. a., Watson/Wright), sind nicht gemessen und nicht gelesen."),
        ("**Keine eine Wahrheit der Wichtigkeit**", "Die Wahrheit je Verfahren ist selbst ein Funktional: dieselbe Funktion μ hat für die Art gewöhnliche Permutation "
                                                    f"{fmt(grid['truth']['perm'][0])}, bedingte Permutation {fmt(grid['truth']['cperm'][0], 3)} und LOCO {fmt(grid['truth']['loco'][0], 3)}. Bei LOCO ist die Wahrheit der Verlustanstieg des besten Vorhersagers, "
                                                    "nicht die Wichtigkeit dieses Modells ohne das Merkmal."),
        ("**Rauschmerkmal (Gewicht)**", f"Anteil der {n} Zufalls-Seeds, in denen das Intervall B (1000 Testzeilen) über 0 liegt (einseitig nominal 5 %): Permutation {pct(noise['perm'][0])} bis {pct(noise['perm'][1])}, "
                                        f"bedingte Permutation {pct(noise['cperm'][0])} bis {pct(noise['cperm'][1])}, LOCO {pct(noise['loco'][0])} bis {pct(noise['loco'][1])} (je nach Klasse); "
                                        + ("das Mittel des LOCO-Werts ist bei allen drei Klassen negativ: das Weglassen eines Rauschmerkmals verbessert den Testfehler." if loco_neg else "das Mittel des LOCO-Werts ist nicht bei allen Klassen negativ.")),
        ("**Kein empfohlenes Verfahren**", "Die Reihenfolge der Verfahren bei der Abdeckung der Wahrheit hängt an Klasse, Merkmal und Testzeilen; eine Empfehlung „nimm LOCO“ gibt die Messung nicht her."),
        ("**Kein Nutzen belegt**", "Ob Intervalle um die Wichtigkeit in Betrieb, Logistik oder Recht zu besseren Entscheidungen führen, ist nicht belegt und nicht Gegenstand."),
    ]
    return "| Annahme oder Grenze | Was gilt (gemessen in diesem Vehikel) |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)


GRENZEN_EINLEITUNG = "Was die Zahlen nicht sagen (Zahlen aus dem eingefrorenen Raster, Voraussetzungen im Satz):"
