"""Wichtigkeit der Merkmale mit Intervall: wen deckt das Intervall? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Reihe „Beitrag und Beleg“ (Modelle verstehen): Permutation, bedingte Permutation und LOCO mit Intervall (Bootstrap und Normalnäherung) auf einem erfundenen Container-Vehikel mit bekannter wahrer Funktion.
Die Modelle (3 Klassen x 100 Zufalls-Seeds) stehen eingefroren in data/wik_grid.json (tools/wik_make_grid.py); live laufen der Zeilen-Bootstrap auf den Verlustdifferenzen eines Beispielmodells,
die Abdeckungsformel und die Verdikte.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import wik_constants as C
import wik_evaluation as E
import wik_intervals as IV
import wik_texts as T
from wik_formula import coverage_from_ratio
from wik_layout import dunkles_design, kopf_css
from wik_presets import (
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from wik_visualization import (
    build_bootstrap_hist,
    build_coverage_bars,
    build_formula,
    build_interval,
    build_testrows,
    build_twin,
)

st.set_page_config(page_title="Wichtigkeit der Merkmale mit Intervall: wen deckt das Intervall? – Sebastian Hanisch", layout="wide")


@st.cache_resource(show_spinner=False)
def _grid():
    return E.load_grid()


@st.cache_data(show_spinner=False)
def _example():
    return E.load_example()


@st.cache_data(show_spinner=False)
def _live_bootstrap(kl, method, feature, mi):
    """Zeilen-Bootstrap live auf den eingefrorenen Verlustdifferenzen des Beispielmodells (Zufalls-Seed 1 der Zelle): 300 Neuziehungen der Testzeilen."""
    d = np.asarray(_example()["diff"][kl], dtype=float)[:, E.col(method, feature)]
    m = C.TEST_SIZES[mi]
    idx = IV.bootstrap_indices(m, C.N_BOOT, C.SEED_BOOT + mi)
    t0 = time.perf_counter()
    means = IV.bootstrap_means(d[:m], idx)
    return means, IV.interval_summary(d[:m], idx), time.perf_counter() - t0


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in wik_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(f"📏 {T.TITEL}")
st.markdown(T.EINLEITUNG)

with st.expander("So entsteht das Intervall", expanded=True):
    st.markdown(T.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
grid = _grid()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Modell und Verfahren**")
    klasse = st.selectbox("Modellklasse", C.CLASSES, key="klasse_select", format_func=C.CLASS_LABELS.get, help="Drei Klassen mit festen Einstellungen, je 1000 Trainingszeilen.")
    verfahren = st.radio("Verfahren", C.METHODS, key="verfahren_radio", format_func=lambda m: C.METHOD_LABELS[m],
                         help="Permutation: Spalte unter allen Testzeilen umordnen. Bedingt: nur innerhalb von Schichten. LOCO: Merkmal weglassen und neu trainieren.")
    merkmal = st.selectbox("Merkmal", list(range(C.D)), key="merkmal_select", format_func=lambda j: C.FEATURE_LABELS[j] + (" (reines Rauschen)" if j == 5 else ""))
    st.markdown("**Intervall**")
    art = st.radio("Intervallart", C.INTERVAL_KINDS, key="art_radio", format_func=lambda k: C.INTERVAL_LABELS[k], horizontal=True)
    testzeilen = st.select_slider("Testzeilen", options=C.TEST_SIZES, key="testzeilen_select", help="Anfänge desselben, je Zufalls-Seed neu gezogenen Testsatzes mit 3000 Zeilen.")
    seed = st.slider("Zufalls-Seed der Zelle (Neu-Training)", 1, C.SEED_MAX, key="seed_slider", help="Welches der 100 Neu-Trainings Experiment 1 zeigt; die Abdeckungen laufen über alle 100.")
    st.button("🎲 Anderen Zufalls-Seed zeigen", width="stretch", on_click=randomize_seed)

sync_query_params({"klasse_select": klasse, "verfahren_radio": verfahren, "merkmal_select": int(merkmal), "art_radio": art, "testzeilen_select": int(testzeilen), "seed_slider": int(seed)})
mi = C.TEST_SIZES.index(int(testzeilen))
seed = int(seed)
merkmal = int(merkmal)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: ein Neu-Training, ein Intervall")
lo, hi = (a[seed - 1] for a in E.interval(grid, klasse, mi, verfahren, merkmal, art))
est = float(E.estimate(grid, klasse, mi, verfahren, merkmal)[seed - 1])
tg = E.targets(grid, klasse, verfahren, merkmal)
ranks, rho = E.rank_view(grid, klasse, mi, verfahren, seed)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Schätzwert (Betragsmaß)", E.fmt(est, 3), delta=f"Intervall {E.fmt(float(lo), 3)} bis {E.fmt(float(hi), 3)}", delta_color="off")
c2.metric("Halbe Breite", E.fmt(float(hi - lo) / 2, 3))
c3.metric("Rang unter den sechs Merkmalen", f"{int(ranks[merkmal])} von {C.D}", delta="Rangmaß", delta_color="off")
c4.metric("Rangkorrelation mit der Wahrheit je Verfahren", E.fmt(rho), delta="dieses Neu-Training", delta_color="off")
st.plotly_chart(build_interval(float(lo), float(hi), est, float(tg["pop"][seed - 1]), tg["tmean"], tg["truth"], C.FEATURE_LABELS[merkmal]), width="stretch", key="main_interval")
kind, text = E.verdict_intervall(grid, klasse, mi, verfahren, merkmal, art, seed)
getattr(st, kind)(text)
st.caption("Punkt mit Balken: Schätzwert und Intervall dieses Neu-Trainings. Senkrechte Linien: die drei Ziele (gepunktet: Wichtigkeit dieses Modells auf einer Bevölkerung von 30 000 Zeilen; gestrichelt: Mittel davon über die 100 Zufalls-Seeds; "
           "durchgezogen: Wahrheit je Verfahren). Ein einzelnes Neu-Training ist kein Befund: die Abdeckungen unten laufen über 100 Zufalls-Seeds.")

st.markdown("---")

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie breit ist das Intervall, und wen deckt es?**")
kind, text = E.verdict_abdeckung(grid, klasse, art, mi)
getattr(st, kind)(text)
st.markdown("**2. Was ändert sich mit mehr Testzeilen?**")
kind, text = E.verdict_testzeilen(grid, klasse, verfahren, art)
getattr(st, kind)(text)
st.markdown("**3. Was passiert bei zwei Spalten, die dasselbe messen?**")
kind, text = E.verdict_zwilling(grid, klasse)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Neu-Training, ein Intervall (Bootstrap live)")
st.caption("Der Zeilen-Bootstrap läuft hier live auf den eingefrorenen Verlustdifferenzen eines Beispielmodells (Zufalls-Seed 1 der gewählten Klasse, 3000 Testzeilen x 18 Spalten). Klasse, Verfahren, Merkmal und Testzeilen stehen in der Seitenleiste.")
with st.spinner("Bootstrap: 300 Neuziehungen der Testzeilen (bei 3000 Testzeilen etwa zwei Sekunden)"):
    means, summ, secs = _live_bootstrap(klasse, verfahren, merkmal, mi)
st.plotly_chart(build_bootstrap_hist(means, summ[1], summ[2], summ[0]), width="stretch", key="boot_hist")
frozen = E.est_array(grid, klasse)[0, mi, E.col(verfahren, merkmal)]
st.markdown(f"Live: Schätzwert {E.fmt(summ[0], 4)}, Bootstrap-Intervall {E.fmt(summ[1], 4)} bis {E.fmt(summ[2], 4)}, Normalnäherung {E.fmt(summ[3], 4)} bis {E.fmt(summ[4], 4)} "
            f"(gerechnet in {secs:.2f} s). Eingefroren im Raster, Zufalls-Seed 1: {E.fmt(float(frozen[0]), 4)}, {E.fmt(float(frozen[1]), 4)} bis {E.fmt(float(frozen[2]), 4)}.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wen deckt das Intervall? (die Falle)")
table = {m: tuple(E.mean_coverage(grid, klasse, m, art, mi, t) for t in C.TARGETS) for m in C.METHODS}
st.plotly_chart(build_coverage_bars(table, C.NOMINAL, E.standard_error(C.NOMINAL, C.N_SEEDS)), width="stretch", key="coverage_bars")
kind, text = E.verdict_abdeckung(grid, klasse, art, mi)
getattr(st, kind)(text)
st.markdown("**Ehrlicher Negativbefund:** Die Rangfolge der Verfahren bei der Abdeckung der Wahrheit hängt an Klasse und Testzeilen; eine einfache Empfehlung („nimm LOCO“) gibt die Messung nicht her.")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Mehr Testzeilen, aber nicht mehr Verlass")
hw = [float(np.mean([E.mean_half_width(grid, klasse, verfahren, f, art, k) for f in C.EFFECT_FEATURES])) for k in range(3)]
cov_m = [E.mean_coverage(grid, klasse, verfahren, art, k, "tmean") for k in range(3)]
cov_p = [E.mean_coverage(grid, klasse, verfahren, art, k, "pop") for k in range(3)]
st.plotly_chart(build_testrows(list(C.TEST_SIZES), hw, cov_m, cov_p), width="stretch", key="testrows")
kind, text = E.verdict_testzeilen(grid, klasse, verfahren, art)
getattr(st, kind)(text)
st.plotly_chart(build_formula(E.cells_135(grid, art)), width="stretch", key="formula")
st.caption("Punkte: die 135 Zellen (3 Klassen x 3 Verfahren x 5 Merkmale mit Wirkung x 3 Testzeilen-Stufen) mit dem gemessenen V und der gemessenen Abdeckung der Mitte über Neu-Trainings; Linie: die Formel aus dem Expander 📐. "
           f"Beispiel: V = 1,25 gibt nach der Formel {E.pct(coverage_from_ratio(1.25))}.")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Zwei Spalten, die dasselbe messen")
tw = E.twin_stats(grid, klasse)
st.plotly_chart(build_twin([(C.TWIN_LABELS["perm"], tw["perm"]["beide_ueber_0"], tw["perm"]["beide_nicht_ueber_0"]), (C.TWIN_LABELS["cperm"], tw["cperm"]["beide_ueber_0"], tw["cperm"]["beide_nicht_ueber_0"]),
                 ("LOCO einzeln", tw["loco"]["beide_ueber_0"], tw["loco"]["beide_nicht_ueber_0"])]), width="stretch", key="twin")
kind, text = E.verdict_zwilling(grid, klasse)
getattr(st, kind)(text)
st.markdown("Geschlossenes Gegenstück (Gauß-Paar mit Korrelation 9/10 und f = (A + B)/2, Modell = wahre Funktion; in den Tests nachgerechnet): Permutation je Zwilling 1/2, bedingte Permutation 19/200, LOCO einzeln 19/400, LOCO beide weggelassen 19/20.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN_EINLEITUNG)
st.markdown(T.grenzen_markdown(grid))

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.FORMEL_LESEHILFE)
    st.markdown(T.FORMEL)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
