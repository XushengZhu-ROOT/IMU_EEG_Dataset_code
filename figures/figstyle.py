"""Style of the paper figures: Helvetica 8 pt (the macOS system font; elsewhere the matplotlib default sans-serif), black
text, 11-pt bold title, axes placed in cm, PNG at 300 dpi and PDF with TrueType fonts."""
import contextlib
import logging
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

CM = 1 / 2.54
BLUE = (42 / 255, 120 / 255, 214 / 255)
ORANGE = (235 / 255, 104 / 255, 52 / 255)
GREEN = (26 / 255, 107 / 255, 60 / 255)
HELVETICA = '/System/Library/Fonts/Helvetica.ttc'
RC = {
    'font.size': 8, 'text.color': 'k',
    'axes.titlesize': 8, 'axes.titleweight': 'normal', 'axes.titlecolor': 'k', 'axes.titlepad': 3,
    'axes.labelsize': 8, 'axes.labelcolor': 'k', 'axes.labelpad': 2, 'axes.linewidth': 0.6, 'axes.edgecolor': 'k',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.facecolor': 'white',
    'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.minor.width': 0.6, 'ytick.minor.width': 0.6,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'xtick.color': 'k', 'ytick.color': 'k',
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'legend.fontsize': 8, 'legend.frameon': False,
    'axes.unicode_minus': True, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'}


@contextlib.contextmanager
def paper_style(**rc):
    """Figures drawn and saved inside the block get the paper style (rc: changed rcParams). matplotlib reads only the
    first face (Regular) of Helvetica.ttc, so the Bold face is written to a temporary .ttf and registered. DejaVu Sans
    supplies the glyphs that Helvetica lacks (the bar U+2223 of |r|)."""
    logging.getLogger('fontTools').setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory() as tmp:
        family = 'sans-serif'
        if os.path.exists(HELVETICA):
            from fontTools.ttLib import TTCollection
            for face in TTCollection(HELVETICA, recalcTimestamp=False).fonts:
                if face['name'].getDebugName(4) == 'Helvetica Bold':
                    path = os.path.join(tmp, 'Helvetica-Bold.ttf')
                    face.save(path)
                    font_manager.fontManager.addfont(path)
                    family = 'Helvetica'
        with plt.rc_context({**RC, 'font.family': [family, 'DejaVu Sans'], **rc}):
            yield


def save(fig, png, pdf):
    fig.savefig(png, dpi=300)
    fig.savefig(pdf, metadata={'CreationDate': None, 'Creator': None, 'Producer': None})
    plt.close(fig)


def x_unit(ax, text):
    """x label = the unit only, 7 pt, close to the axis."""
    ax.set_xlabel(text, fontsize=7, labelpad=1, color='k')


class Page:
    """Figure of w_cm x h_cm; axes placed in cm from the top-left corner."""

    def __init__(self, w_cm, h_cm):
        self.w, self.h = w_cm, h_cm
        self.fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))

    def axes(self, x, y_top, w, h):
        return self.fig.add_axes([x / self.w, 1 - (y_top + h) / self.h, w / self.w, h / self.h])

    def title(self, text):
        """11 pt bold, centred at the top."""
        self.fig.text(0.5, 1 - 0.12 / self.h, text, ha='center', va='top', fontsize=11, fontweight='bold', color='k')

    def legend(self, handles, labels, y_top=0.72, x=0.15):
        """One row at the top left, under the title."""
        self.fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(x / self.w, 1 - y_top / self.h),
                        ncol=len(handles), frameon=False, fontsize=8, handlelength=2.2, handletextpad=0.6,
                        columnspacing=1.8, borderaxespad=0, borderpad=0, labelcolor='k')
