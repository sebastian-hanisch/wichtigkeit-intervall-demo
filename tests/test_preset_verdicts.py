"""Je Preset die erwartete Verdikt-Zeile in der App (DEMO-PLAYBOOK §11a Punkt 7): jedes Preset zeigt seine Klasse, sein Verfahren und seine Zahlen, ohne Ausnahme."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import wik_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")

ERWARTET = {
    "Boosting, 1000 Zeilen, Permutation": ("Boosting, 1000 Testzeilen, Permutation, Art, Zufalls-Seed 1", "Wahrheit je Verfahren (1,713): deckt nicht", "Permutation: Modell 89 %, Mitte 67 %, Wahrheit 35 %"),
    "Random Forest, bedingte Permutation": ("Random Forest, 1000 Testzeilen, Bedingte Permutation, Art, Zufalls-Seed 1", "Schätzwert 0,063", "Bedingte Permutation: Modell 87 %, Mitte 77 %, Wahrheit 59 %"),
    "Netz, LOCO": ("Kleines Netz, 1000 Testzeilen, LOCO, Art, Zufalls-Seed 1", "Wahrheit je Verfahren (0,037): deckt", "LOCO: Modell 90 %, Mitte 88 %, Wahrheit 78 %"),
    "3000 Testzeilen (enges Intervall)": ("Boosting, 3000 Testzeilen, Permutation, Vorlauf, Zufalls-Seed 1", "Wahrheit je Verfahren (1,343): deckt nicht", "Permutation: Modell 86 %, Mitte 54 %, Wahrheit 19 %"),
    "Zwillingsmerkmal": ("Intervall Normalnäherung", "Boosting, 100 Zufalls-Seeds, 1000 Testzeilen, Intervall B: Permutation nennt Vorlauf und Zwilling beide wichtig (beide Intervalle über 0 in 100 %)", "beide nicht über 0 in 82 %"),
}


def _texte(at):
    return " ".join([e.value for e in at.info] + [e.value for e in at.warning] + [e.value for e in at.success])


def test_erwartungen_decken_alle_presets():
    assert set(ERWARTET) == set(C.PRESETS)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_zeigt_die_erwartete_verdikt_zeile(name):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    text = _texte(at)
    for frag in ERWARTET[name]:
        assert frag in text, (name, frag)


def test_klassen_verfahren_und_merkmal_laufen_ohne_fehler():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for kl, m in ("gbm", "perm"), ("rf", "cperm"), ("mlp", "loco"), ("mlp", "perm"):
        at.sidebar.selectbox(key="klasse_select").set_value(kl).run()
        at.sidebar.radio(key="verfahren_radio").set_value(m).run()
        assert not at.exception, (kl, m, [str(e) for e in at.exception])
    at.sidebar.select_slider(key="testzeilen_select").set_value(300).run()
    at.sidebar.radio(key="art_radio").set_value("N").run()
    at.sidebar.selectbox(key="merkmal_select").set_value(5).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "Gewicht" in _texte(at)                      # das Rauschmerkmal wird gezeigt
