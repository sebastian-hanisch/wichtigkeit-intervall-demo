# Wichtigkeit der Merkmale mit Intervall: wen deckt das Intervall? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-wichtigkeit-intervall-demo.streamlit.app/)** -->

Demo der Reihe **„Beitrag und Beleg“** (Feld „Modelle verstehen“) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Reihe zeigt, dass die Streuung der Wichtigkeit eines Merkmals von Neu-Training zu Neu-Training schwankt;
dieses Stück fragt nach der naheliegenden Antwort: **ein Intervall aus der Testmenge** soll sagen, wie sicher die Zahl ist. Ein Intervall hat aber ein **Ziel**, und was es deckt, hängt am Ziel und am Verfahren.

Die Daten sind ausgedacht: ein erfundener Generator für Container-Verweilzeiten, dessen wahre Mitte bekannt ist. Es gibt keine reale Datenquelle. In den gelesenen Quellen gab es keine Messung, ob Intervalle um die Wichtigkeit von Merkmalen
in Logistik, Lager oder Fertigung zu besseren Entscheidungen führen; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet. Gezählt werden Zufalls-Seeds (Neu-Trainings mit eigenem Testsatz), nicht Menschen oder Standorte.

## Kernfrage

Wie breit ist das Intervall um die **Wichtigkeit der Merkmale**, und **wen deckt es**: die Wichtigkeit **dieses einen Modells** (Ziel „Modell“, gerechnet auf einer festen Bevölkerung von 30 000 Zeilen), die Wichtigkeit eines Modells aus 1000 Zeilen **im Mittel über Neu-Trainings**
(Ziel „Mitte“) oder die **wahre Wichtigkeit je Verfahren** für die bekannte Funktion μ (Ziel „Wahrheit“)? Drei Verfahren stehen nebeneinander: **Permutation**, **bedingte Permutation** (Umordnen innerhalb von Schichten) und **LOCO**
(leave-one-covariate-out: ein **Merkmal** weglassen und **neu trainieren**; nicht das Leave-one-out der Trainingspunkte). Es gibt **keine eine Wahrheit der Wichtigkeit**: dieselbe Funktion μ hat für die Art bei gewöhnlicher Permutation 1,713, bei bedingter Permutation 0,073 und bei LOCO 0,037. **Wichtigkeit** heißt hier immer die Wichtigkeit der Merkmale (global), nicht der **Beitrag** einer Zeile. **Gemessen, kein Verfahren empfohlen.**

## Befunde und Korrekturen gegenüber Plan und Vormessung

Die Zahlen des Plans (Vormessung mit dem Zufallsstrom von numpy, 27 Zellen aus drei Klassen, drei Datenmengen und drei Verfahren) sind Erwartungen. Hier ist **nur die Datenmenge n = 1000** neu gemessen (Plan Z5 c: „bei Platzmangel nur n = 1000“), mit dem Zufallsstrom
SplitMix64 der Demo, 100 Zufalls-Seeds je Klasse und dem Ziel „Modell“ aus 30 000 statt 300 000 Zeilen; die Datenmengen 300 und 3000 des Plans sind **nicht** gerechnet. Die Testzeilen 300, 1000 und 3000 sind Anfänge desselben Testsatzes. **Gleich geblieben:** die Richtung (das Intervall deckt
die Wichtigkeit des Modells fast nominal, die Mitte über Neu-Trainings schlechter, die Wahrheit je Verfahren am schlechtesten), gewöhnliche Permutation deckt die wahre Wichtigkeit von Art und Vorlauf in 0 von 100 Zufalls-Seeds (9 von 9 Zellen), das Rauschmerkmal, die Zwillinge.
**Verschoben:**

- **Abdeckung des Modells:** 86 bis 92 % (Mittel über fünf Merkmale mit Wirkung, 54 Zellen aus drei Klassen, drei Verfahren, drei Testzeilen-Stufen und beiden Intervallarten) statt 83 bis 90 % bei K08; bei 3000 Testzeilen sind es 86 bis 92 %, die Lücke bei vielen Testzeilen ist hier
  kleiner als im Plan (83 bis 86 %). Der Standardfehler eines Anteils um 90 % liegt bei 100 Zufalls-Seeds bei 3,0 Prozentpunkten, ein Mittel über fünf Merkmale ist weniger streuend, aber die Merkmale sind nicht unabhängig.
- **Rangfolge der Verfahren bei der Abdeckung der Wahrheit** (Permutation < bedingte Permutation < LOCO): in **7 von 9** Zellen (Klasse x Testzeilen), Plan 8 von 9. Gesichert ist nur, dass die Permutation in allen 9 Zellen deutlich seltener deckt als die beiden anderen Verfahren; zwischen bedingter Permutation und LOCO ist die Reihenfolge nur bei Boosting und Random Forest erkennbar, beim Netz sind beide im Rahmen der Streuung gleich. Ausnahmen: das Netz mit 300 Testzeilen (bedingt 84 %, LOCO 83 %) und mit 3000 (je 60 %, gleich); beim Netz mit 1000 liegen beide bei 78 % (nur im Nachkommabereich geordnet, so unter den 7 gezählt).
- **Verhältnis V und Formel:** die Formel trifft die 135 Zellen (3 Klassen x 3 Verfahren x 5 Merkmale x 3 Testzeilen-Stufen) im Mittel auf 3,8 Prozentpunkte (größter Abstand 14,2; in 96 % der Zellen höchstens 10, in 70 % höchstens 5); Plan 3,7, 13 und 97 % / 73 %.
- **Nicht gerechnet:** n = 300 und n = 3000 Trainingszeilen (die 27 Zellen des Plans), die Partner-Klasse der bedingten Permutation im Zwillings-Experiment (hier nur Schichten nach der Art), mehr Umordnungen (Befund G des Plans), Experiment 5 des Plans (Rauschmerkmal, nur als Zeile im Abschnitt „Nicht gemessen“).
- **Gegenprobe mit zweitem Zufalls-Seed-Bereich:** siehe Abschnitt „Gegenprobe“ unten.

## Modell

Je Testzeile $i$ die Zunahme des quadratischen Fehlers gegen $y$ (Formeln im 📐-Expander, ein vollständiges Modell der Rechnung dieser Demo). **Permutation:** die Spalte des Merkmals unter allen Testzeilen umordnen, drei Umordnungen gemittelt. **Bedingte Permutation:** innerhalb der Schichten Art
(für die Art selbst: Vorlauf größer als 0 gegen gleich 0). **LOCO:** Merkmal weglassen, neu trainieren (Boosting und Random Forest wirklich ohne die Spalte, das Netz mit einer Konstanten an ihrer Stelle). Die Wichtigkeit ist das Mittel der Zeilenwerte.
**Intervalle (90 %, je Testsatz):** **B** = 5. und 95. Perzentil von 300 Neuziehungen der Testzeilen (SplitMix64), **N** = Mittel ± 1,6449 Standardfehler. Beide berücksichtigen nur die Auswahl der Testzeilen. **Abdeckung** = Anteil der Zufalls-Seeds, deren Intervall das Ziel enthält.
**Wahrheit je Verfahren:** dasselbe Verfahren auf μ angewandt (Permutation und bedingte Permutation aus 300 000 Zeilen, LOCO exakt aus der bedingten Erwartung: Verlustanstieg des besten Vorhersagers ohne das Merkmal; Gewicht 0).
Die drei **Ziele**: „Modell“ = dasselbe Verfahren desselben Modells auf der festen Bevölkerung von 30 000 Zeilen, „Mitte“ = Mittel davon über die 100 Zufalls-Seeds der Zelle, „Wahrheit“.

## Methodik

- **Drei Modellklassen** (immer mindestens zwei): Boosting (200 Bäume, Tiefe 3, Lernrate 0,1), Random Forest (60 Bäume, min. Blatt 3, max_features 0,5), kleines Netz (eine versteckte Schicht mit 16 Einheiten, L2 3, L-BFGS; auf R² gegen μ abgestimmt). 1000 Trainingszeilen, 1000 bis 3000 Testzeilen.
- **Raster eingefroren** (`data/wik_grid.json`, Quelle, Stand und Umgebung im Kopf der Datei): je Klasse und Zufalls-Seed (100) und Testzeilen-Stufe (300, 1000, 3000): Schätzwert und beide Intervalle für 3 Verfahren x 6 Merkmale, dazu der Wert auf der Bevölkerung und das Zwillings-Experiment
  (eine siebte Spalte Vorlauf + N(0; 0,05²), die μ nicht liest; 1000 Testzeilen). `python tools/wik_make_grid.py --check` rechnet die Wahrheit neu und eine Zelle als Band nach.
- **Live** läuft der Zeilen-Bootstrap auf den eingefrorenen Verlustdifferenzen eines Beispielmodells je Klasse (`data/wik_example.json`, Zufalls-Seed 1, 3000 Testzeilen x 18 Spalten): 300 Neuziehungen; er gibt dieselben Intervalle wie das Raster (Test), Rechenzeit bei 3000 Testzeilen etwa 2 Sekunden (gemessen 2,2 s, mit `st.spinner` und Zwischenspeicher), bei 300 Testzeilen 0,2 s.
  Keine Modellrechnung zur Laufzeit; scikit-learn steht nur in `requirements-dev.txt`.
- **Zufall:** SplitMix64 (`wik_rng.py`, aus der Vorlage kopiert) für Daten, Umordnungen und Bootstrap; Modelle und BLAS können in der letzten Stelle abweichen, darum prüfen die Tests die Neu-Rechnung als Band.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für diesen erfundenen Generator, drei Modellklassen mit festen Einstellungen, 1000 Trainingszeilen, quadratischen Fehler, Zeilen-Bootstrap und Normalnäherung, und sind **Mittel über die fünf Merkmale mit Wirkung** (Art, Kunde, Zoll, Wochenende, Vorlauf) über
**100 Zufalls-Seeds**; bei einzelnen Merkmalen weichen sie ab. Intervall B, wenn nicht anders genannt.

**Wahrheit je Verfahren (Art / Kunde / Zoll / Wochenende / Vorlauf / Gewicht).** Permutation 1,713 / 0,132 / 0,050 / 0,016 / 1,343 / 0,000; bedingte Permutation 0,073 / 0,132 / 0,064 / 0,016 / 0,280 / 0,000; LOCO 0,037 / 0,066 / 0,032 / 0,008 / 0,142 / 0,000.
Die Wichtigkeit der Art in den Modellen (Mittel von „Modell“ über die Zufalls-Seeds, Permutation): Boosting 0,13, Random Forest 0,14, Netz 0,83.

**Wen deckt das Intervall? 1000 Testzeilen, Anteil der Zufalls-Seeds.**

| Klasse | Verfahren | Wichtigkeit des Modells | Mitte über Neu-Trainings | Wahrheit je Verfahren |
|---|---|---|---|---|
| Boosting | Permutation | 89 % | 67 % | 35 % |
| Boosting | Bedingte Permutation | 88 % | 81 % | 68 % |
| Boosting | LOCO | 90 % | 86 % | 80 % |
| Random Forest | Permutation | 88 % | 71 % | 26 % |
| Random Forest | Bedingte Permutation | 87 % | 77 % | 59 % |
| Random Forest | LOCO | 91 % | 89 % | 86 % |
| Kleines Netz | Permutation | 89 % | 65 % | 41 % |
| Kleines Netz | Bedingte Permutation | 88 % | 80 % | 78 % |
| Kleines Netz | LOCO | 90 % | 88 % | 78 % |

Das Intervall deckt die Wichtigkeit des Modells bei 300 Testzeilen zu 86 bis 89 %, bei 1000 zu 87 bis 91 % und bei 3000 zu 86 bis 92 % (je neun Zellen, Intervall B; Normalnäherung 87 bis 92 %), nominal sind 90. Gewöhnliche Permutation deckt die wahre Wichtigkeit von **Art und Vorlauf in 0 von 100 Zufalls-Seeds**
in allen neun Zellen (Klasse x Testzeilen); **0 von 100 heißt nicht, dass die Wahrscheinlichkeit 0 ist**. Bei der Abdeckung der Wahrheit liegt die Permutation in allen 9 Zellen deutlich unter den beiden anderen Verfahren; die volle Rangfolge gilt in 7 von 9 Zellen (siehe oben), zwischen bedingter Permutation und LOCO nur bei Boosting und Random Forest erkennbar. Eine einfache Empfehlung („nimm LOCO“) gibt die Messung nicht her.
Rangmaß neben dem Betragsmaß: die Rangkorrelation der sechs Schätzwerte mit dem Rang der Wahrheit je Verfahren liegt bei 1000 Testzeilen im Mittel über die Zufalls-Seeds zwischen 0,88 und 0,99 (Permutation, bedingt, LOCO; Boosting 0,88 / 0,98 / 0,96, Random Forest 0,92 / 0,94 / 0,97, Netz 0,98 / 0,99 / 0,95); die Rangfolge der
sechs Merkmale wird in dieser Messung im Mittel gut getroffen (0,88 bis 0,99). Das ist kein Vergleich mit der Abdeckung der Beträge (je nach Verfahren 26 bis 86 % in der Tabelle oben), und bei sehr verschiedenen Wahrheiten ist eine hohe Rangkorrelation leicht zu erreichen.

**Mehr Testzeilen, aber nicht mehr Verlass (Intervall B).** Halbe Breite und V = SD der Modellwichtigkeit über Neu-Trainings geteilt durch die halbe Breite, bei 300 / 1000 / 3000 Testzeilen:

| Klasse, Verfahren | halbe Breite | V | Abdeckung der Mitte | Abdeckung des Modells |
|---|---|---|---|---|
| Boosting, Permutation | 0,0362 / 0,0197 / 0,0114 | 0,40 / 0,73 / 1,25 | 81 % / 67 % / 54 % | 88 % / 89 % / 86 % |
| Boosting, Bedingte Permutation | 0,0329 / 0,0181 / 0,0103 | 0,25 / 0,46 / 0,80 | 86 % / 81 % / 62 % | 88 % / 88 % / 87 % |
| Boosting, LOCO | 0,0309 / 0,0168 / 0,0097 | 0,15 / 0,28 / 0,48 | 89 % / 86 % / 81 % | 88 % / 90 % / 88 % |
| Random Forest, Permutation | 0,0361 / 0,0199 / 0,0115 | 0,38 / 0,68 / 1,18 | 81 % / 71 % / 53 % | 87 % / 88 % / 88 % |
| Random Forest, Bedingte Permutation | 0,0322 / 0,0179 / 0,0103 | 0,26 / 0,46 / 0,80 | 83 % / 77 % / 65 % | 87 % / 87 % / 87 % |
| Random Forest, LOCO | 0,0302 / 0,0168 / 0,0097 | 0,15 / 0,28 / 0,48 | 86 % / 89 % / 78 % | 86 % / 91 % / 90 % |
| Kleines Netz, Permutation | 0,0608 / 0,0335 / 0,0193 | 0,51 / 0,92 / 1,60 | 75 % / 65 % / 49 % | 87 % / 89 % / 87 % |
| Kleines Netz, Bedingte Permutation | 0,0336 / 0,0185 / 0,0106 | 0,25 / 0,45 / 0,79 | 85 % / 80 % / 64 % | 87 % / 88 % / 87 % |
| Kleines Netz, LOCO | 0,0293 / 0,0161 / 0,0093 | 0,15 / 0,27 / 0,47 | 89 % / 88 % / 80 % | 89 % / 90 % / 92 % |

Mehr Testzeilen verengen das Intervall (bei Boosting mit Permutation das 3,2-Fache von 300 auf 3000 Zeilen), ohne die Streuung durch das Training zu verkleinern; die Abdeckung der Mitte über Neu-Trainings **sinkt** (Boosting, Permutation: 81, 67 und 54 %), die des Modells bleibt (86 bis 89 %).
Die geschlossene Formel Abdeckung = 2 Φ(z / √(1 + r²)) − 1 mit r = z · V trifft die gemessene Abdeckung der Mitte in 135 Zellen im Mittel auf 3,8 Prozentpunkte (größter Abstand 14,2).

**Zwei Spalten, die dasselbe messen (Vorlauf und sein Zwilling; 1000 Testzeilen, Intervall B).**

| Klasse | Permutation: beide über 0 | Bedingte Permutation: beide über 0 | LOCO einzeln: beide nicht über 0 | LOCO einzeln: beide über 0 | LOCO ohne beide: über 0 |
|---|---|---|---|---|---|
| Boosting | 100 % | 94 % | 82 % | 0 % | 100 % |
| Random Forest | 100 % | 100 % | 90 % | 0 % | 100 % |
| Kleines Netz | 100 % | 100 % | 82 % | 6 % | 100 % |

Die Permutation nennt beide Spalten wichtig; LOCO einzeln nennt in 82 bis 90 % der Zufalls-Seeds beide unwichtig, nur die Gruppe wird in 100 % als wichtig erkannt: der Wert von LOCO ohne beide liegt bei 0,15 (Boosting, Random Forest und Netz). Der Zwilling ist bei Permutation in 54 % (Boosting), 56 % (Random Forest) und 37 % (Netz) der Zufalls-Seeds größer als der Vorlauf.
Geschlossenes Gegenstück (Gauß-Paar mit Korrelation 9/10, f = (A + B)/2; Handrechnung und Simulation in `tests/test_importance.py`): Permutation je Zwilling 1/2, bedingte Permutation 19/200, LOCO einzeln 19/400, LOCO ohne beide 19/20.

### Zahl → Test

| Zahl der README | Test |
|---|---|
| Wahrheit je Verfahren | `tests/test_claims.py::test_wahrheit` (Band) und `tests/test_grid.py` |
| Tabelle „Wen deckt das Intervall?“ | `tests/test_claims.py::test_tabelle_abdeckung` (Raster) |
| Spannen der Abdeckung des Modells, Rangfolge 7 von 9 | `tests/test_claims.py::test_spannen_und_rangfolge` |
| Tabelle „Mehr Testzeilen“, Formel auf 3,8 Prozentpunkte | `tests/test_claims.py::test_tabelle_testzeilen` |
| Tabelle „Zwei Spalten“ | `tests/test_claims.py::test_tabelle_zwilling` |
| Art und Vorlauf 0 von 100 | `tests/test_grid.py::test_gewoehnliche_permutation_deckt_die_wahre_wichtigkeit_von_art_und_vorlauf_nie_in_den_gezaehlten_zellen` |
| Handbeispiele (24 Umordnungen, 852/1296, 768/1296, 7/8, 352/625, 19/200, 19/400, 19/20, Formelwerte) | `tests/test_importance.py`, `tests/test_intervals.py` (exakt) |
| Rauschmerkmal im Abschnitt „Nicht gemessen“ | `tests/test_claims.py::test_grenzen_abschnitt_steht_in_der_readme` |

Messungen dieser README ohne Test: die Gegenprobe (Abschnitt unten), die Rangkorrelationen und die Rechenzeit der Raster-Erzeugung (802 s mit vier Prozessen, nur Orientierung).

## Ehrliche Grenzen

- **Nur in diesem Vehikel:** ein erfundener Generator, MSE als Verlust, drei Modellklassen mit festen Einstellungen, **eine** Trainingsgröße (1000 Zeilen), Testzeilen-Bootstrap und Normalnäherung.
- **Das Intervall kennt nur die Testzeilen.** Es kennt nicht das Training und nicht die Wahl der Umordnung. Welches Intervall man stattdessen nehmen soll, sagt dieses Stück **nicht**; Williamson u. a., Watson/Wright und Lei u. a. sind nicht gelesen.
- **Die Wahrheit je Verfahren ist selbst ein Funktional.** Bei LOCO ist sie der Verlustanstieg des besten Vorhersagers, nicht die Wichtigkeit dieses Modells ohne das Merkmal; das Weglassen eines Rauschmerkmals **verbessert** den Testfehler (siehe unten), die Wahrheit 0 wird deshalb nur teilweise gedeckt.
- **Mittel über fünf Merkmale:** einzelne Merkmale weichen um mehrere Prozentpunkte ab; die Spannen sind Zufallsstreuung von 100 Zufalls-Seeds (Standardfehler eines Anteils um 90 % etwa 3 Prozentpunkte).
- **Das Ziel „Modell“ kommt aus 30 000 Zeilen**; die Bevölkerung ist für alle Zufalls-Seeds dieselbe, ihr Stichprobenfehler ist ein gemeinsamer Versatz.
- Ursachen der Lücken (Abhängigkeit durch Umordnen in einer endlichen Testmenge, endliche Bevölkerung, Wahl der Umordnung) sind nicht getrennt gemessen; Deutungen sind Hypothesen.

## 🚧 Bewusst nicht enthalten / nicht gemessen

<!-- GRENZEN -->
| Annahme oder Grenze | Was gilt (gemessen in diesem Vehikel) |
|---|---|
| **Ein Vehikel, eine Trainingsgröße** | Erfundener Verweilzeit-Generator, sechs Merkmale, quadratischer Fehler, 1000 Trainingszeilen; Boosting, Random Forest und kleines Netz mit festen Einstellungen. Andere Trainingsgrößen, andere Verluste (Klassifikation) und andere Einstellungen sind nicht gemessen. |
| **Intervall nur über die Testzeilen** | Beide Intervalle (B und N) berücksichtigen nur die Auswahl der Testzeilen, nicht das Training und nicht die Wahl der Umordnung. Intervalle, die das Training einbeziehen (Bootstrap mit Neu-Training, Stichprobenteilung, Williamson u. a., Watson/Wright), sind nicht gemessen und nicht gelesen. |
| **Keine eine Wahrheit der Wichtigkeit** | Die Wahrheit je Verfahren ist selbst ein Funktional: dieselbe Funktion μ hat für die Art gewöhnliche Permutation 1,71, bedingte Permutation 0,073 und LOCO 0,037. Bei LOCO ist die Wahrheit der Verlustanstieg des besten Vorhersagers, nicht die Wichtigkeit dieses Modells ohne das Merkmal. |
| **Rauschmerkmal (Gewicht)** | Anteil der 100 Zufalls-Seeds, in denen das Intervall B (1000 Testzeilen) über 0 liegt (einseitig nominal 5 %): Permutation 4 % bis 8 %, bedingte Permutation 3 % bis 8 %, LOCO 0 % bis 1 % (je nach Klasse); das Mittel des LOCO-Werts ist bei allen drei Klassen negativ: das Weglassen eines Rauschmerkmals verbessert den Testfehler. |
| **Kein empfohlenes Verfahren** | Die Reihenfolge der Verfahren bei der Abdeckung der Wahrheit hängt an Klasse, Merkmal und Testzeilen; eine Empfehlung „nimm LOCO“ gibt die Messung nicht her. |
| **Kein Nutzen belegt** | Ob Intervalle um die Wichtigkeit in Betrieb, Logistik oder Recht zu besseren Entscheidungen führen, ist nicht belegt und nicht Gegenstand. |
<!-- /GRENZEN -->

- Intervalle, die das Training einbeziehen (Bootstrap mit Neu-Training, Stichprobenteilung, Williamson u. a., CPI, Knockoffs): nicht gemessen und nicht gelesen.
- Klassifikationsverlust, mehr als sechs Merkmale, andere Zwillingsstärke als 0,05 und mehr als ein Zwillingspaar, Testgrößen unter 300, t-Verteilung statt Normalnäherung: nicht gemessen.
- Trainingsgrößen 300 und 3000, die Partner-Klasse der bedingten Permutation, mehr Umordnungen: nicht gerechnet.
- Plattform: Raster auf Windows, Python 3.14, scikit-learn 1.9; Linux-Lauf (Docker, Python 3.12) am 2026-11-20 grün.
- Wirkung in Betrieb: nicht belegt. Menschenstudien, Praxis und Regulierung: Einordnung, keine Rechtsaussage.
- Begriffe „Wichtigkeit (Intervall)“ und „Abdeckung“ in der Seite „Grundlagen“ der Website: eigene Freigabe, nicht Teil dieser Demo.

## Gegenprobe mit zweitem Zufalls-Seed-Bereich

Dieselbe Rechnung mit einem zweiten Zufalls-Seed-Bereich (alle Zufallsströme um 1 verschoben: andere Trainings-, Test-, Bevölkerungs- und Bootstrap-Zeilen; Rechenzeit 790 s, **Messung dieser README, nicht im Test**, die Datei liegt nicht im Repo). Die Wahrheit je Verfahren
ändert sich um höchstens 0,003 (Permutation, Art: 1,715 gegen 1,713). Über die 81 Zellen aus Klasse, Verfahren, drei Testzeilen-Stufen und den drei Zielen (Intervall B) unterscheiden sich die beiden Bereiche im Mittel um 1,3 Prozentpunkte, im größten Fall um 4 (Netz, Permutation, 3000 Testzeilen, Mitte über
Neu-Trainings; Boosting, bedingte Permutation, 1000 Testzeilen, Wahrheit). Die Abdeckung des Modells liegt im zweiten Bereich bei 85 bis 91 % (beide Intervallarten, 54 Zellen), die Rangfolge der Verfahren bei der Abdeckung der Wahrheit gilt in 8 von 9 Zellen (Hauptlauf 7 von 9: die Ausnahmen beim Netz sind
Gleichstände im Rahmen des Standardfehlers). Gewöhnliche Permutation deckt die wahre Wichtigkeit von Art und Vorlauf wieder in 0 von 100 Zufalls-Seeds in allen neun Zellen. Zwillinge: Permutation beide über 0 in 100 %, LOCO einzeln beide nicht über 0 in 87 / 89 / 77 % (Boosting, Random Forest, Netz), LOCO ohne beide über 0 in 100 %.

## Quellen (Stufe)

- Hooker, Mentch, Zhou, arXiv 1905.03151 (**A**, Abstract; Titel gewechselt: „Please Stop Permuting Features“, später „Unrestricted Permutation forces Extrapolation“): unbeschränktes Permutieren erzwingt Extrapolation.
- Strobl u. a. 2008, BMC Bioinformatics 9:307 (**S**): bedingte Permutation.
- Lei u. a. 2018, JASA, „Distribution-Free Predictive Inference For Regression“ (**A**, Titel; die Zuordnung von LOCO ist nicht bestätigt); Verdinelli/Wasserman, arXiv 2303.05981 (**A**); Williamson u. a. 2023, JASA 118:1645 (**A**); Watson/Wright (**A**); Ewald u. a. 2024 (**A**);
  Slack u. a. 2020 (**A**); Fisher/Rudin/Dominici 2019 (**A**); Covert/Lee 2021 (**A**). **Nicht gelesen:** alle PDFs; die Behauptungen der Quellen über Gültigkeit der Intervalle werden nicht übernommen, sondern an einem Vehikel mit bekannter Funktion nachgemessen.
- **Keine Neuheitsaussage:** alle drei Verfahren und die Intervalle sind Lehrbuch und Forschung; die Demo führt sie nachrechenbar mit Zähleinheit, exakten Handbeispielen und bekannter Funktion vor. Vorbilder: Hooker/Mentch (Version 1), XAI-Bench, OpenXAI.

## Verifikation

Die Suite hat **91 Tests** (`pytest --collect-only`; Laufzeit unter starker Last durch parallel arbeitende Sitzungen gemessen: etwa 190 Sekunden mit `-n 3`, Python 3.14, Windows; einzelne Tests brauchen unter Last bis 2 Minuten, davon `--check` mit dem Neu-Trainieren einer Boosting-Zelle):

| Art | Tests | Datei |
|---|---|---|
| Orakel und Handbeispiele | 12 + 9 | `tests/test_importance.py` (24 Umordnungen von Hand 5/2 und 11 / 2 / 11, `sklearn.inspection.permutation_importance` im Band, jede Schicht behält ihre Werte, LOCO gegen die Normalgleichung, Gauß-Zwilling 1/2, 19/200, 19/400, 19/20 exakt und simuliert, LOCO-Wahrheit gegen geschlossene Form) und `tests/test_intervals.py` (Abdeckung 852/1296 und 768/1296 durch Aufzählung auf zwei Wegen, Bernoulli 7/8 und 352/625, Perzentil gegen numpy und scipy, Abdeckung des Intervall-Codes auf χ²-Verlusten im Band, Formelwerte 0,9000 / 0,8588 / 0,7552 / 0,5380 / 0,3970 gegen Simulation) |
| Generator und Zufallsstrom | 12 | `tests/test_generator.py` |
| Raster | 9 | `tests/test_grid.py`: Form, Herkunft, Wahrheit je Verfahren im Band, Intervalle geordnet, Beispielmodell live gleich eingefroren, 0 von 100 in 9 Zellen, Rauschmerkmal, Neu-Rechnen als Band (`--check`) |
| Claims | 7 | `tests/test_claims.py`: jede Zahl der README aus dem Raster |
| Verdikte | 8 + 7 | `tests/test_verdicts.py` (handgesetztes Raster, Randfälle), `tests/test_preset_verdicts.py` (je Preset die erwartete Zeile, Klassen, Verfahren, Rauschmerkmal) |
| Presets, Layout, App, Footer, Laufzeit-Importe | 6 + 15 + 3 + 2 + 1 | AppTest über jeden Regler und Button |

Rechenzeit der App je Klick: alle Kennzahlen und Verdikte eines Klicks 0,1 s (Raster als numpy-Felder zwischengespeichert), der Zeilen-Bootstrap live 0,2 s bei 300 und 2,2 s bei 3000 Testzeilen (Spinner, Zwischenspeicher); gemessen, keine Zeit-Assertion im Test. Eine Browser-Sitzung (Streamlit auf Port 10077) hat die Seite geladen,
sechs Diagramme gezeichnet, den Live-Bootstrap gezeigt und keine KaTeX-Fehler gemeldet; nicht jedes Preset wurde im Browser durchgeklickt (das prüft der AppTest).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `wik_constants.py` | Generator-Konstanten, Raster-Größen, Presets |
| `wik_generator.py` | Verweilzeit-Generator (kopiert aus verweilzeit-demo) mit Zwillingsspalte |
| `wik_rng.py` | SplitMix64 (aus der Vorlage kopiert) |
| `wik_importance.py` | Permutation, bedingte Permutation, LOCO je Testzeile |
| `wik_intervals.py` | Bootstrap-Intervall, Normalnäherung, Abdeckung |
| `wik_formula.py` | Abdeckung der Mitte über Neu-Trainings in geschlossener Form |
| `wik_value.py` | bedingte Erwartung von μ (LOCO-Wahrheit) |
| `wik_evaluation.py` | Kennzahlen aus dem Raster, Verdikte als reine Funktionen |
| `wik_texts.py` | Texte der App, Abschnitt „Wo die Annahmen enden“ |
| `wik_presets.py`, `wik_layout.py`, `wik_visualization.py` | Presets und Permalink, fester Kopf, Plotly-Abbildungen |
| `data/wik_grid.json`, `data/wik_example.json` | eingefrorenes Raster, Verlustdifferenzen des Beispielmodells |
| `tools/wik_make_grid.py` | rechnet das Raster, `--check` prüft die Datei |
| `tests/` | Orakel, Handbeispiele, Eigenschaften, Randfälle, Claims, Verdikte, Layout, AppTest, Footer, Laufzeit-Importe |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
