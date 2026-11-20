"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import wik_constants as C
from wik_formula import coverage_from_ratio


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_interval(lo, hi, est, pop, tmean, truth, label):
    """Ein Intervall mit Schätzwert und den drei Zielen als senkrechte Linien; die Achse reicht bis zum größten gezeigten Wert."""
    xmax = max(hi, truth, pop, tmean, est) * 1.12 + 1e-9
    xmin = min(0.0, lo, truth, pop, tmean) * 1.12 - 1e-9
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[est], y=[label], mode="markers", name="Schätzwert", marker=dict(size=12, color=C.COLOR_MODEL),
                             error_x=dict(type="data", symmetric=False, array=[hi - est], arrayminus=[est - lo], color=C.COLOR_MODEL, thickness=3, width=10),
                             hovertemplate=f"Schätzwert %{{x:.3f}} (Intervall {lo:.3f} bis {hi:.3f})<extra></extra>"))
    for value, name, color, dash in ((pop, "Wichtigkeit des Modells", C.COLOR_GRAY, "dot"), (tmean, "Mitte über Neu-Trainings", C.COLOR_BAD, "dash"), (truth, "Wahrheit je Verfahren", C.COLOR_TRUTH, "solid")):
        fig.add_trace(go.Scatter(x=[value, value], y=[-0.5, 0.5], mode="lines", name=name, line=dict(color=color, width=3, dash=dash), hovertemplate=f"{name}: {value:.3f}<extra></extra>"))
    fig.update_xaxes(range=[xmin, xmax], title_text="Zunahme des mittleren quadratischen Fehlers")
    fig.update_yaxes(range=[-0.6, 0.6], showticklabels=False)
    return _base(fig, height=240)


def build_bootstrap_hist(means, lo, hi, est):
    fig = go.Figure(go.Histogram(x=[float(v) for v in means], nbinsx=30, marker_color=C.COLOR_MODEL, hovertemplate="%{x:.3f}: %{y}<extra></extra>", name="Neuziehungen"))
    for v, name, color in ((lo, "5. Perzentil", C.COLOR_BAD), (hi, "95. Perzentil", C.COLOR_BAD), (est, "Mittel der Testzeilen", C.COLOR_TRUTH)):
        fig.add_vline(x=float(v), line_color=color, line_dash="dash", annotation_text=name, annotation_position="top")
    fig.update_xaxes(title_text="Mittel der Verlustdifferenz in einer Neuziehung der Testzeilen")
    fig.update_yaxes(title_text="Neuziehungen")
    return _base(fig, height=280)


def build_coverage_bars(table, nominal, se):
    """table[verfahren] = (Modell, Mitte, Wahrheit) in Anteilen; gruppierte Balken, Linie bei nominal."""
    fig = go.Figure()
    colors = (C.COLOR_MODEL, C.COLOR_BAD, C.COLOR_TRUTH)
    for k, tg in enumerate(C.TARGETS):
        fig.add_trace(go.Bar(x=[C.METHOD_LABELS[m].split(" (")[0] for m in table], y=[100 * table[m][k] for m in table], name=C.TARGET_LABELS[tg], marker_color=colors[k],
                             hovertemplate="%{x}: %{y:.0f} %<extra>" + C.TARGET_LABELS[tg] + "</extra>"))
    fig.add_hline(y=100 * nominal, line_dash="dot", line_color="gray", annotation_text=f"nominal 90 % (Standardfehler etwa {100 * se:.1f} Prozentpunkte)".replace(".", ","))
    fig.update_yaxes(title_text="Abdeckung (% der Zufalls-Seeds)", range=[0, 105])
    fig.update_layout(barmode="group")
    return _base(fig, height=340)


def build_testrows(sizes, half_widths, cov_mean, cov_pop):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=[100 * v for v in cov_pop], mode="lines+markers", name="Wichtigkeit des Modells", line=dict(color=C.COLOR_MODEL, width=3)))
    fig.add_trace(go.Scatter(x=sizes, y=[100 * v for v in cov_mean], mode="lines+markers", name="Mitte über Neu-Trainings", line=dict(color=C.COLOR_BAD, width=3)))
    fig.add_hline(y=90, line_dash="dot", line_color="gray")
    fig.update_xaxes(title_text="Testzeilen", type="log", tickvals=list(sizes), ticktext=[str(s) for s in sizes])
    fig.update_yaxes(title_text="Abdeckung (% der Zufalls-Seeds)", range=[0, 105])
    return _base(fig, height=320)


def build_formula(cells):
    """Die 135 Zellen als Punkte (V, gemessene Abdeckung der Mitte) und die Formel 2 Phi(z / sqrt(1 + r^2)) - 1 mit r = z V."""
    vmax = max(max(c[0] for c in cells) * 1.05, 0.5)
    xs = np.linspace(0, vmax, 80)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[c[0] for c in cells], y=[100 * c[1] for c in cells], mode="markers", name="gemessen (135 Zellen)", marker=dict(size=6, color=C.COLOR_MODEL, opacity=0.6),
                             hovertemplate="V = %{x:.2f}: %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[100 * coverage_from_ratio(float(v)) for v in xs], mode="lines", name="Formel", line=dict(color=C.COLOR_BAD, width=3)))
    fig.update_xaxes(title_text="V = SD der Modellwichtigkeit / halbe Breite")
    fig.update_yaxes(title_text="Abdeckung der Mitte über Neu-Trainings (%)", range=[0, 105])
    return _base(fig, height=340)


def build_twin(rows):
    """rows = [(Verfahren, beide über 0, beide nicht über 0)] in Anteilen."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[100 * r[1] for r in rows], name="beide Intervalle über 0", marker_color=C.COLOR_MODEL))
    fig.add_trace(go.Bar(x=[r[0] for r in rows], y=[100 * r[2] for r in rows], name="beide nicht über 0", marker_color=C.COLOR_BAD))
    fig.update_yaxes(title_text="% der Zufalls-Seeds", range=[0, 105])
    fig.update_layout(barmode="group")
    return _base(fig, height=320)
