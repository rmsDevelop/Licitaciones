#Las figuras matplotlib del panel EDA del dashboard: la API las computa y
#las sirve como SVG (GET /eda/fig/{dim} de api.py). Un solo builder generico
#— barras con el numero de licitaciones por categoria y un eje gemelo con
#dos lineas (descuento medio % y numero de ofertas medio) — porque todas las
#dimensiones del EDA (ano, cpv, tipo de contrato) cuentan la misma historia.
#
#Los colores son la paleta categorica del dashboard (slots azul/amarillo de
#--series-1/--series-2 mas aqua y naranja), validada por CVD en claro y
#oscuro contra las superficies del panel; TEMAS lleva los tokens CSS del
#dashboard, no un flip automatico. El color sigue a la entidad: descuento es
#siempre aqua y ofertas siempre naranja, cambie lo que cambie el filtro; las
#barras toman el color del conjunto (amarillo en menores, azul en el resto).

from __future__ import annotations

import io
import re
import threading

import matplotlib

matplotlib.use("Agg")  # sin display: la API solo rasteriza/vectoriza a SVG
import matplotlib.pyplot as plt  # noqa: E402  (tras el backend Agg)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# svg.fonttype=none deja el texto como texto (lo tipografia el navegador con
# la sans del sistema, como el resto del dashboard) y evita los path-glyphs.
matplotlib.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.unicode_minus": False,
})

TEMAS = {
    "claro": {
        "texto": "#0b0b0b", "texto_2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7", "superficie": "#fcfcfb",
        "barras_lic": "#2a78d6",   # slot 1 azul (series-1 del dashboard)
        "barras_men": "#eda100",   # slot 4 amarillo (series-2)
        "linea_desc": "#1baf7a",   # slot 3 aqua
        "linea_ofertas": "#eb6834",  # slot 2 naranja
    },
    "oscuro": {
        "texto": "#ffffff", "texto_2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835", "superficie": "#1a1a19",
        "barras_lic": "#3987e5",
        "barras_men": "#c98500",
        "linea_desc": "#199e70",
        "linea_ofertas": "#d95926",
    },
}


def _k(v: float, _: int) -> str:
    """1,2 M / 34 K / 780 — ticks del eje de conteos."""
    if abs(v) >= 1e6:
        return f"{v / 1e6:.1f} M".replace(".", ",")
    if abs(v) >= 1e3:
        return f"{v / 1e3:.0f} K"
    return f"{v:g}"


# matplotlib Agg no es thread-safe: la API puede renderizar la figura clara
# y la oscura a la vez (el <picture> del dashboard pide ambas), asi que el
# render entero va bajo lock — son ~80 ms, serializa sin notarse.
_LOCK = threading.Lock()


def fig_eda(labels: list[str], n: list, desc: list, ofertas: list,
            tema: str = "claro", conjunto: str = "ambos",
            formato: str = "svg") -> str | bytes:
    """Figura de una dimension del EDA: barras (n) + lineas (desc, ofertas).

    Los tres ejes llegan ya agregados (api._eda): aqui solo forma y color.
    Eje izquierdo = conteos; eje derecho = el que comparten las dos lineas
    (unidades distintas que la leyenda desambigua: % y ofertas medias).
    formato=png (bytes) es para los tests visuales; la API sirve svg.
    """
    t = TEMAS[tema]
    color_barras = t["barras_men"] if conjunto == "menores" else t["barras_lic"]
    with _LOCK:
        return _dibuja(labels, n, desc, ofertas, t, color_barras, formato)


def _dibuja(labels, n, desc, ofertas, t, color_barras, formato):
    """El render propiamente dicho (bajo lock)."""
    # ancho de tarjeta: a ~900 px de render las fuentes salen a tamano
    # nativo (9-10 pt ~ 12-13 px), ni miniaturas ni cartelera
    fig, ax = plt.subplots(figsize=(9.4, 3.4))
    fig.patch.set_alpha(0)  # transparente: el fondo lo pone la tarjeta HTML

    x = range(len(labels))
    ax.bar(x, n, width=0.68, color=color_barras, zorder=2)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(_k))

    # eje gemelo con las dos medias; el anillo de superficie separa los
    # marcadores de las barras que cruzan
    ax2 = ax.twinx()
    ax2.plot(x, desc, color=t["linea_desc"], linewidth=2, marker="o",
             markersize=5.5, markeredgecolor=t["superficie"],
             markeredgewidth=1.5, zorder=4, clip_on=False,
             solid_capstyle="round")
    ax2.plot(x, ofertas, color=t["linea_ofertas"], linewidth=2, marker="o",
             markersize=5.5, markeredgecolor=t["superficie"],
             markeredgewidth=1.5, zorder=4, clip_on=False,
             solid_capstyle="round")

    # cromo recessivo: solo baseline, tinta muted
    for a, lado in ((ax, "left"), (ax2, "right")):
        for spine, visible in (("bottom", True), (lado, True),
                               ("top", False),
                               ("right" if lado == "left" else "left", False)):
            a.spines[spine].set_visible(visible)
            a.spines[spine].set_color(t["baseline"])
        a.tick_params(colors=t["muted"], labelsize=9, length=3)
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_color(t["muted"])
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    # etiquetas del eje x: giradas si son largas, raleadas si son muchas
    # (rotation_mode=anchor alinea la etiqueta girada bajo su tick)
    etiquetas = [str(s) for s in labels]
    paso = max(1, -(-len(etiquetas) // 18))  # ceil: <= 18 etiquetas visibles
    pos = list(x)[::paso]
    largas = max(map(len, etiquetas), default=0) > 6
    ax.set_xticks(pos, [etiquetas[i] for i in pos],
                  rotation=25 if largas else 0,
                  ha="right" if largas else "center",
                  rotation_mode="anchor")

    # leyenda siempre (>= 2 series): parche + las dos lineas, tinta de texto
    ax.legend(
        handles=[Patch(facecolor=color_barras, label="nº licitaciones"),
                 Line2D([], [], color=t["linea_desc"], linewidth=2,
                        marker="o", markersize=5, label="descuento medio (%)"),
                 Line2D([], [], color=t["linea_ofertas"], linewidth=2,
                        marker="o", markersize=5, label="nº ofertas medio")],
        loc="upper left", frameon=False, fontsize=9,
        labelcolor=t["texto_2"], handlelength=1.6, borderaxespad=0.2)

    buf = io.BytesIO()
    # bbox tight: el lienzo crece hasta abarcar las etiquetas giradas del
    # eje x (los nombres de tipo de contrato son largos y se cortaban)
    fig.savefig(buf, format=formato, transparent=True, pad_inches=0.12,
                bbox_inches="tight", dpi=150 if formato == "png" else None)
    plt.close(fig)
    out = buf.getvalue()
    if formato != "svg":
        return out
    # responsive: sin width/height fijos en la raiz (solo viewBox) — el
    # <img> del dashboard le da width:100% y el navegador escala
    # proporcional al hueco disponible
    return re.sub(r'(<svg\b[^>]*?) width="[^"]*" height="[^"]*"',
                  r"\1", out.decode(), count=1)
