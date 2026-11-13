"""eyeart: draws ADAM's eye as a grid of pixels. Pure Python, no dependencies.

ADAM (the ship's computer in Metroid Fusion) is a grey metal housing around a glowing iris with a
dark pupil and a white glint. This file only does the DRAWING: give it a size, how wide the pupil
is, how shut the eyelid is and which colours to use, and it hands back rows of (r, g, b) pixels
(or None for "nothing here"). adam.py turns those pixels into terminal characters and animates them.
"""
import colorsys
import math

NAVY = (7, 11, 36)          # the panel behind the eye (also what "outside the eye" looks like)
OUTLINE = (24, 24, 58)
GAP = (26, 20, 64)
PUPIL = (22, 20, 56)
WHITE = (244, 246, 251)
LID = (139, 146, 144)
LID_EDGE = (42, 44, 54)
# metal tones, brightest to darkest (light comes from the top left)
METAL = [(232, 244, 241), (185, 207, 203), (139, 146, 144), (91, 95, 98), (61, 64, 68)]
SEAM = (43, 45, 58)


def rgb(hex_colour: str) -> tuple:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# Iris colours: rim (edge), body (mid), shadow (deep), and the pale crescent (light).
PALETTES = {
    "pink":  {"edge": rgb("#ff2bf0"), "mid": rgb("#c01cc0"), "deep": rgb("#5a0f66"), "light": rgb("#f7b6f7")},
    "cyan":  {"edge": rgb("#2be8ff"), "mid": rgb("#1aa6cc"), "deep": rgb("#0a4768"), "light": rgb("#bdf7ff")},
    "amber": {"edge": rgb("#ffb62b"), "mid": rgb("#cc8514"), "deep": rgb("#6e4308"), "light": rgb("#ffe3a0")},
    "green": {"edge": rgb("#3dff8a"), "mid": rgb("#1fb55c"), "deep": rgb("#0b5a2c"), "light": rgb("#c4ffd9")},
    "red":   {"edge": rgb("#ff3b3b"), "mid": rgb("#c21f1f"), "deep": rgb("#610d0d"), "light": rgb("#ffb3b3")},
    "ice":   {"edge": rgb("#a8b4ff"), "mid": rgb("#6c78e0"), "deep": rgb("#262a78"), "light": rgb("#e6e9ff")},
}


def mix(a: tuple, b: tuple, k: float) -> tuple:
    k = max(0.0, min(1.0, k))
    return tuple(round(a[i] + (b[i] - a[i]) * k) for i in range(3))


def mix_palette(a: dict, b: dict, k: float) -> dict:
    """A palette part-way between two others (used to fade smoothly from one colour to the next)."""
    return {name: mix(a[name], b[name], k) for name in a}


def hue_palette(hue: float) -> dict:
    """A palette for any hue 0..1 (the rainbow easter egg spins this)."""
    def c(s, v):
        return tuple(round(x * 255) for x in colorsys.hsv_to_rgb(hue % 1.0, s, v))
    return {"edge": c(0.85, 1.0), "mid": c(0.9, 0.75), "deep": c(1.0, 0.35), "light": c(0.3, 1.0)}


# --- ADAM's dialogue box, as in the game -------------------------------------------------
# A dark green box with a bright green rim and a pale inner line. ADAM's eye sits on its top-left
# corner in a grey metal mount that wraps down around the box's left side and bottom corner.
BOX_FILL = (29, 86, 52)
BOX_OUTLINE = (9, 30, 17)
BOX_RIM_DIM = (40, 150, 70)
BOX_RIM_BRIGHT = (125, 255, 160)
BOX_INNER = (190, 255, 200)


EYE_STRETCH = 1.3        # the eye is drawn this much wider than it is tall (it suits the box)
_eye_trim: dict = {}


def _stretch(rows: list, sx: float) -> list:
    width = round(len(rows[0]) * sx)
    return [[r[min(len(r) - 1, int((x + 0.5) / sx))] for x in range(width)] for r in rows]


def _trim(n: int, sx: float) -> tuple:
    """(empty columns on the left, width) of a stretched n-pixel eye. Its outline never changes, so
    this is worked out once per size."""
    if (n, sx) not in _eye_trim:
        rows = _stretch(eye_pixels(n, 0.27, 0, PALETTES["pink"]), sx)
        lead = min(next((i for i, p in enumerate(r) if p is not None), len(r)) for r in rows)
        _eye_trim[(n, sx)] = (lead, len(rows[0]) - lead)
    return _eye_trim[(n, sx)]


def eye_width(n: int, sx: float = EYE_STRETCH) -> int:
    return _trim(n, sx)[1]


def stretch_eye(rows: list, n: int, sx: float = EYE_STRETCH) -> list:
    """Widen the eye and drop the empty columns on its left, so it starts right at the edge."""
    lead = _trim(n, sx)[0]
    return [r[lead:] for r in _stretch(rows, sx)]


def box_geometry(w: int, h: int, n: int, ew: int | None = None) -> dict:
    """Where everything goes, in character cells, for a w x h widget with an n-pixel-tall eye that is
    ew cells wide. (One cell is one pixel wide and two pixels tall.)"""
    ew = ew or eye_width(n)
    box_top = (n // 4) * 2                                   # in pixels; even, so it starts on a cell edge
    text_left = ew + 2
    return {
        "box_left": ew // 2,                                 # the box starts under the eye's centre
        "box_top_px": box_top,
        "eye_w": ew,
        "eye": (0, 0, ew - 1, n // 2 - 1),                   # x0, y0, x1, y1 (cells, inclusive)
        "text": (text_left, box_top // 2 + 2, w - 5, h - 3),   # where ADAM's words go
        "arrow_down": (text_left + (w - 4 - text_left) // 2, h - 2),
        "arrow_up": (text_left + (w - 4 - text_left) // 2, box_top // 2 + 1),
    }


def dialogue_pixels(w: int, h: int, n: int, glow: float, eye: list | None) -> list:
    """Draw the whole thing (box, metal mount, eye) on a w x (2h) pixel grid. glow 0..1 brightens the rim."""
    hp = h * 2
    ew = eye_width(n)
    g = box_geometry(w, h, n, ew)
    bx, bt = g["box_left"], g["box_top_px"]
    rim = mix(BOX_RIM_DIM, BOX_RIM_BRIGHT, glow)
    layers = (BOX_OUTLINE, rim, BOX_INNER, BOX_FILL)
    buf = [[None] * w for _ in range(hp)]
    for y in range(bt, hp):                                  # the box: outline, rim, inner line, fill
        dy = min(y - bt, hp - 1 - y)
        lv = 0 if dy == 0 else 1 if dy <= 2 else 2 if dy == 3 else 3
        for x in range(bx, w):
            dx = min(x - bx, w - 1 - x)
            if dx == 0 and dy == 0:
                continue                                     # clipped corners
            lh = 0 if dx == 0 else 1 if dx == 1 else 2 if dx == 2 else 3
            buf[y][x] = layers[min(lv, lh)]

    def metal_block(x0, y0, x1, y1, light_left=True):
        for y in range(max(0, y0), min(hp, y1 + 1)):
            for x in range(max(0, x0), min(w, x1 + 1)):
                if x in (x0, x1) or y in (y0, y1):
                    buf[y][x] = OUTLINE
                else:
                    tone = 1 if (light_left and x - x0 <= 1) or y - y0 <= 1 else 3 if (x1 - x <= 1 or y1 - y <= 1) else 2
                    buf[y][x] = METAL[tone]

    cy = n // 2
    metal_block(0, cy, bx + 3, hp - 1)                       # the mount under the eye, over the box's left edge
    metal_block(bx + 3, hp - 5, bx + 8, hp - 1)              # a lip that wraps the box's bottom-left corner
    metal_block(ew - 2, bt - 1, ew + 3, bt + 3)              # a bracket where the eye meets the box's top edge
    if eye:                                                  # the eye goes on top of everything, widened
        for y, row in enumerate(stretch_eye(eye, n)):
            for x, px in enumerate(row):
                if px is not None and y < hp and x < w:
                    buf[y][x] = px
    return buf


def eye_pixels(n: int, pupil: float, lid: float, pal: dict, scan: float | None = None) -> list:
    """Draw the eye on an n x n grid.

    pupil: pupil radius as a fraction of the eye (about 0.12 tight focus ... 0.34 wide open)
    lid:   0 = open, 1 = shut
    scan:  a row number to light up like a scanner beam (None = off)
    """
    c = (n - 1) / 2
    R = n / 2
    iris_edge = 0.60
    seams = [math.radians(a) for a in (-100, 20, 140)]

    def in_body(x: int, y: int) -> bool:
        """The housing's silhouette. Not a circle: a rounded-square body with a lug on each top corner
        (the right one a touch bigger), like the game's art. The iris inside stays round."""
        if not (0 <= x < n and 0 <= y < n):
            return False
        ux, uy = (x - c) / R, (y - c) / R
        if abs(ux) ** 3 + abs(uy) ** 3 <= 0.86 ** 3 and uy <= 0.9:
            return True
        for side, size in ((-1, 0.27), (1, 0.31)):
            lx, ly = ux - side * 0.62, uy + 0.62
            if abs(lx) <= size and abs(ly) <= size and not (abs(lx) > size - 0.1 and abs(ly) > size - 0.1):
                return True
        return False

    inside = [[in_body(x, y) for x in range(n)] for y in range(n)]
    rows = []
    for y in range(n):
        row = []
        for x in range(n):
            dx, dy = x - c, y - c
            d = math.hypot(dx, dy)
            u = d / R
            light = -(dx + dy) / (d * 1.4142 + 1e-9)       # +1 toward the top left, -1 toward the bottom right
            if not inside[y][x]:
                row.append(None)
                continue
            if not all(in_body(x + ox, y + oy) for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                row.append(OUTLINE)                         # dark outline all the way round the silhouette
                continue
            if u > 0.68:                                    # the metal housing
                if light > 0.55:
                    tone = 0 if (u > 0.80 and light > 0.78) else 1
                elif light > 0.12:
                    tone = 1 if u > 0.80 else 2
                elif light > -0.35:
                    tone = 2
                elif light > -0.7:
                    tone = 3
                else:
                    tone = 4
                px = METAL[tone]
                a = math.atan2(dy, dx)                      # panel seams, like the real thing
                if any(abs((a - s + math.pi) % (2 * math.pi) - math.pi) * d < 0.55 for s in seams):
                    px = SEAM
                row.append(px)
                continue
            if u > iris_edge + 0.04:                        # the dark gap between housing and iris
                row.append(GAP)
                continue
            # --- inside: iris, pupil, glint, scanner, eyelid ---
            if u < pupil:
                px = PUPIL
            else:
                t = (u - pupil) / max(1e-6, iris_edge - pupil)
                t = max(0.0, min(1.0, t))
                q = round(t * 4) / 4                        # quantised: that is what makes it look like pixel art
                px = mix(pal["deep"], pal["mid"], q / 0.5) if q < 0.5 else mix(pal["mid"], pal["edge"], (q - 0.5) / 0.5)
                if light < -0.2 and dy > 0 and 0.45 < t < 0.82:
                    px = pal["light"]                       # the pale crescent at the bottom
            gx, gy = -0.33 * R, -0.33 * R                   # the glint
            gd = math.hypot(dx - gx, dy - gy)
            gr = max(0.9, 0.13 * R)
            if gd <= gr:
                px = WHITE
            elif gd <= gr + 0.9:
                px = pal["light"]
            if scan is not None and abs(y - scan) < 0.7:
                px = mix(px, WHITE, 0.55)
            if lid > 0.04:                                  # eyelid closing from top and bottom
                open_half = R * 0.66 * (1 - lid)
                if abs(dy) > open_half + 1:
                    px = LID
                elif abs(dy) > open_half:
                    px = LID_EDGE
            row.append(px)
        rows.append(row)
    return rows
