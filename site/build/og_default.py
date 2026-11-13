"""Write site/assets/og-default.png: the 1200x630 social preview for pages without their own image
(tools, hubs, glossary, about). Rerun only if the wording or palette changes; the PNG is committed.

    python site/build/og_default.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sparvi.config import ROOT  # noqa: E402

TEAL, SURFACE, INK2 = "#0f5257", "#fcfcfb", "#52514e"
OUT = ROOT / "site" / "assets" / "og-default.png"

fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=SURFACE)
fig.add_artist(plt.Rectangle((0, 0), 0.025, 1, transform=fig.transFigure, color=TEAL))
fig.text(0.08, 0.60, "Sparvi", fontsize=88, fontweight="bold", color=TEAL, family="DejaVu Sans")
fig.text(0.08, 0.42, "Free, independent sports analytics", fontsize=34, color=INK2, family="DejaVu Sans")
fig.text(0.08, 0.30, "with the data out in the open", fontsize=34, color=INK2, family="DejaVu Sans")
fig.text(0.08, 0.10, "sparvi.io", fontsize=24, color=TEAL, family="DejaVu Sans")
fig.savefig(OUT, dpi=100, facecolor=SURFACE, metadata={"Software": None})
plt.close(fig)
print(f"wrote {OUT.relative_to(ROOT)}")
