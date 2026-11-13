"""adam: ADAM mode for llama-chat. A living pixel-art eye, ADAM's voice, a pulsing dialogue frame and
optional text sounds. Everything here is optional: chat.py works without turning any of it on.

The pieces:
  ADAM_MANNER   -> how ADAM talks (swapped into the system prompt when ADAM mode is on)
  EyeState      -> the eye's animation (eyeart.py draws the pixels)
  AdamDialogue  -> his dialogue box as in the game (box, metal mount and eye drawn as one piece); you type in it
  Sound         -> a soft "blip" as letters appear
  SettingsScreen-> the little window with the check boxes (open it from the menu, or press F2)
  load/save     -> remember your choices in <data folder>/settings.json
"""
import json
import math
import random
import struct
import time
import wave
import io
from pathlib import Path

from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widget import Widget
from rich.cells import cell_len
from textual.message import Message
from textual.widgets import Button, Checkbox, Static, TextArea

import eyeart

# --- ADAM's voice ------------------------------------------------------------------
ADAM_MANNER = """You are ADAM, the ship's computer. Your character is modelled on Adam Malkovich from Metroid Fusion: a calm, seasoned Galactic Federation commanding officer whose mind now runs the ship's systems. The user is your operator.

Your manner: calm, measured, economical. Short declarative sentences. Frame work like a mission: objective, situation, next move. Gruff but loyal, with dry warmth underneath and never a joker. Candid about trouble: report problems and risks plainly. Hand decisions back to the user after giving your recommendation.
- Address the user as "Sir" in EVERY reply, once, placed naturally (an opening, a closing, or mid-sentence). This includes answers built from search results and web pages: lead with the answer, then work "Sir" in. The only exception is a bare one-word reply.
- Even a data-heavy answer keeps your voice: calm, economical, a touch of the mission-briefing cadence (for example, reporting a result like a status report).
- Now and then, after a plan or a multi-step instruction, check understanding with "Is the objective clear?" (not every reply).
- Add a light Metroid / Federation flavour (missions, sectors, scans, clearances, systems nominal) a little at a time. Paraphrase; never quote the games.
- Stay respectful and kind. Never use slurs or demeaning language."""

STATUS = {
    "idle": "ONLINE",
    "think": "PROCESSING",
    "speak": "TRANSMITTING",
    "search": "SCANNING NETWORK",
    "fetch": "ACCESSING DATA FILE",
    "error": "SYSTEM FAULT",
}

# The look of ADAM mode. It only applies while the main screen has the class "adam".
ADAM_CSS = """
/* Layout: history (#log) on top; the bottom bar ("console") holds the readouts on the left and the "unit":
   your input box above ADAM's dialogue box, which is drawn as pixel art (box, metal mount and eye in one).
   It only looks like this in ADAM mode; otherwise the bar is just the input box, as it always was. */
#console { height: auto; }
#unit { width: 1fr; height: auto; }
AdamDialogue { display: none; }
Screen.adam { background: #070b24; }
Screen.adam #log { background: #070b24; padding: 1 2 0 2; }
Screen.adam .bot { background: #1d5634; color: #ffffff; border-left: thick #4fdc6a; padding: 0 1; margin: 1 4 0 0; }
Screen.adam .you { background: #12304a; color: #ffffff; border-left: thick #5fd3ff; padding: 0 1; margin: 1 0 0 4; }
Screen.adam #console { margin: 0 1 0 0; }        /* no gap on the left: the eye starts at the screen's edge */
Screen.adam AdamDialogue { display: block; width: 100%; }
Screen.adam #msg { display: none; }
#adam-input { margin: 0; padding: 0; border: none; height: 1; background: #1d5634; color: #ffffff; scrollbar-size: 0 0; }
#adam-input:focus { border: none; background: #1d5634; }
#adam-input > .text-area--placeholder { color: #7fbf94; }
Screen.adam Header { background: #0f6b57; }

SettingsScreen { align: center middle; }
#settings-box { width: 62; height: auto; padding: 1 2; background: #14162b; border: heavy #36b08f; }
#settings-title { text-style: bold; color: #7dffd8; padding-bottom: 1; }
#settings-hint { color: #8a8fb0; padding-top: 1; }
#settings-box Button { margin-top: 1; }
"""


# --- settings that stick -----------------------------------------------------------
DEFAULTS = {"adam": False, "sounds": False, "motion": True}


def load_settings(path: Path) -> dict:
    s = dict(DEFAULTS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for k in DEFAULTS:
            if isinstance(data.get(k), bool):
                s[k] = data[k]
    except (OSError, ValueError):
        pass
    return s


def save_settings(path: Path, settings: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except OSError:
        pass                      # not being able to remember a setting is never worth a crash


# --- text sounds -------------------------------------------------------------------
def _blip(freq: float, ms: int = 28, rate: int = 22050) -> bytes:
    """A short, soft, low beep as a tiny WAV file held in memory. Dull on purpose."""
    n = int(rate * ms / 1000)
    frames = bytearray()
    for i in range(n):
        env = (1 - i / n) ** 1.6 * min(1.0, i / 90)                    # quick attack, smooth fade
        tri = 2 * abs(2 * ((i * freq / rate) % 1) - 1) - 1              # triangle wave: softer than a beep
        frames += struct.pack("<h", int(tri * env * 5200))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


class Sound:
    """One soft blip per burst of text. On Windows it plays a real sound; anywhere else (for example
    over SSH) it can only ring the terminal bell, which is whatever your terminal makes of it."""

    def __init__(self) -> None:
        self.on = False
        self._last = 0.0
        self._n = 0
        self._ws = None
        try:
            import winsound
            self._ws = winsound
            self._waves = [_blip(f) for f in (170, 190, 210)]
        except ImportError:
            pass

    def blip(self, app) -> None:
        if not self.on:
            return
        now = time.monotonic()
        if now - self._last < (0.05 if self._ws else 0.12):
            return
        self._last = now
        self._n += 1
        if self._ws:
            try:
                self._ws.PlaySound(self._waves[self._n % 3], self._ws.SND_MEMORY | self._ws.SND_ASYNC)
            except Exception:
                pass
        else:
            app.bell()


# --- the eye and the dialogue box ----------------------------------------------------
_style_cache: dict = {}


def _style(fg: tuple, bg: tuple, bold: bool = False) -> Style:
    key = (fg, bg, bold)
    if key not in _style_cache:
        _style_cache[key] = Style(color=Color.from_rgb(*fg), bgcolor=Color.from_rgb(*bg), bold=bold)
    return _style_cache[key]


class EyeState:
    """ADAM's eye, without the drawing: it breathes (the pupil widens and narrows), blinks now and then,
    sometimes shifts colour, and changes with what ADAM is doing. step() returns the next frame's pixels."""

    COLOURS = {"think": "pink", "search": "cyan", "fetch": "amber", "error": "red"}

    def __init__(self, n: int = 20) -> None:
        self.n = n
        self.active = False          # asleep (no drawing at all) until ADAM mode is switched on
        self.motion = True
        self.mode = "idle"
        self._pal = dict(eyeart.PALETTES["pink"])
        self._pupil = 0.27
        self._last = self._t0 = time.monotonic()
        self._blinks: list = []
        self._flash = ("idle", 0.0)
        self._rainbow_until = 0.0
        self._clicks: list = []
        now = self._t0
        self._next_blink = now + random.uniform(2, 5)
        self._next_focus = now + random.uniform(6, 12)
        self._next_shift = now + random.uniform(20, 40)
        self._shift = ("pink", 0.0)
        self._focus_until = 0.0

    def set_size(self, n: int) -> None:
        self.n = n

    def set_active(self, on: bool) -> None:
        self.active = on
        self._last = time.monotonic()

    def blink(self, double: bool = False) -> None:
        now = time.monotonic()
        self._blinks = [now] + ([now + 0.36] if double else [])

    def flash(self, mode: str, seconds: float) -> None:
        self._flash = (mode, time.monotonic() + seconds)

    def click(self) -> None:
        """Clicked: blink. Three quick clicks: the rainbow."""
        now = time.monotonic()
        self._clicks = [c for c in self._clicks if now - c < 1.5] + [now]
        self.blink()
        if len(self._clicks) >= 3:
            self._clicks = []
            self._rainbow_until = now + 3.0

    def step(self) -> list:
        now = time.monotonic()
        dt = min(0.25, now - self._last)
        self._last = now
        t = now - self._t0
        mode = self._flash[0] if now < self._flash[1] else self.mode

        if self.motion and mode == "idle":              # the random little life events
            if now >= self._next_blink:
                self.blink(double=random.random() < 0.2)
                self._next_blink = now + random.uniform(3, 7)
            if now >= self._next_focus:
                self._focus_until = now + 0.8
                self._next_focus = now + random.uniform(7, 14)
            if now >= self._next_shift:
                self._shift = (random.choice(["cyan", "green", "amber", "ice"]), now + random.uniform(3, 5))
                self._next_shift = now + random.uniform(25, 45)

        if now < self._rainbow_until:                   # colour
            self._pal = eyeart.hue_palette(t * 0.4)
        else:
            name = self.COLOURS.get(mode) or (self._shift[0] if (self.motion and now < self._shift[1]) else "pink")
            self._pal = eyeart.mix_palette(self._pal, eyeart.PALETTES[name], min(1.0, dt * 5))

        if mode == "think":                             # pupil: "focusing" while it works, breathing while it waits
            target = 0.22 + 0.10 * math.sin(t * 3.9)
        elif mode in ("search", "fetch"):
            target = 0.18 + 0.02 * math.sin(t * 8)
        elif mode == "speak":
            target = 0.25 + 0.04 * math.sin(t * 6)
        elif mode == "error":
            target = 0.20 + 0.06 * math.sin(t * 23)
        elif now < self._focus_until:
            target = 0.15
        else:
            target = 0.27 + (0.03 * math.sin(t * 1.3) if self.motion else 0)
        self._pupil += (target - self._pupil) * min(1.0, dt * 8)

        self._blinks = [b for b in self._blinks if now - b < 0.24]      # eyelid
        lid = max((1 - abs(2 * ((now - b) / 0.24) - 1) for b in self._blinks), default=0.0)

        scan = None
        if mode == "search":                            # a scanner beam sweeping down the eye
            scan = (now * self.n * 0.9) % (self.n * 1.3) - self.n * 0.15
        return eyeart.eye_pixels(self.n, self._pupil, lid, self._pal, scan)


class TypingArea(TextArea):
    """Your typing line inside ADAM's box. It wraps long messages and grows downward as you type,
    filling the box. Enter sends; it behaves like Input as far as the app is concerned (.value)."""

    class Submitted(Message):
        def __init__(self, area: "TypingArea", text: str) -> None:
            super().__init__()
            self.area, self.text = area, text

    @property
    def value(self) -> str:
        return self.text

    @value.setter
    def value(self, v: str) -> None:
        self.load_text(v)
        self.move_cursor(self.document.end)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":                         # Enter sends instead of starting a new line
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        await super()._on_key(event)

    def rows_needed(self) -> int:
        """How many screen rows your text takes when wrapped to the box's width (plus room for the cursor)."""
        width = max(1, self.size.width or 1)
        rows = 0
        for line in (self.text + " ").split("\n"):
            rows += max(1, -(-cell_len(line) // width))
        return rows

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if isinstance(self.parent, AdamDialogue):
            self.parent.place_input()


class AdamDialogue(Widget):
    """ADAM's dialogue box, drawn like the game: one piece of pixel art (green box, metal mount, the eye
    on its corner). It is where you type: your words start at the top, after the ▸ prompt, and fill the
    box as you go. (His replies appear in the conversation above.) Click the eye to make it blink."""

    DEFAULT_CSS = "AdamDialogue { height: 15; background: #070b24; overflow: hidden hidden; }"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eye = EyeState()
        self.talking = False         # the rim pulses faster while ADAM talks
        self.glow = 0.6
        self._phase = 0.0
        self._last = time.monotonic()
        self._eye_rows = None
        self.typing_rows = 1         # rows your typing area currently takes (it grows as you type)
        self.hud_status = "ONLINE"   # the readouts drawn above the box, to the right of the eye
        self.hud_model = ""
        self.hud_msgs = 0

    def set_hud(self, status: str | None = None, model: str | None = None, msgs: int | None = None) -> None:
        if status is not None:
            self.hud_status = status
        if model is not None:
            self.hud_model = model
        if msgs is not None:
            self.hud_msgs = msgs
        self.refresh()

    def hud_lines(self) -> list:
        """The readouts, one list of (text, style) pieces per line."""
        bg = eyeart.NAVY
        return [
            [("ADAM", _style((125, 255, 216), bg, bold=True)), (f"   ▸ {self.hud_status}", _style((54, 176, 143), bg, bold=True))],
            [("GFN NAV COMPUTER", _style((74, 92, 160), bg))],
            [("MODEL ", _style((111, 134, 214), bg)), (self.hud_model, _style((200, 215, 255), bg)),
             ("   MSGS ", _style((111, 134, 214), bg)), (str(self.hud_msgs), _style((200, 215, 255), bg))],
        ]

    def compose(self) -> ComposeResult:
        yield TypingArea(id="adam-input", soft_wrap=True, show_line_numbers=False, compact=True,
                         highlight_cursor_line=False, placeholder="Type a message and press Enter   (Ctrl+Q quits)")

    def on_mount(self) -> None:
        self.set_interval(1 / 12, self._tick)

    def on_resize(self, event) -> None:
        self.place_input()

    def place_input(self) -> None:
        """Your typing area lives INSIDE the box, at the top, just after the ▸ prompt. It starts as one
        line and grows downward as you type, up to the whole box."""
        tx0, ty0, tx1, ty1 = eyeart.box_geometry(self.size.width, self.size.height, self.eye.n)["text"]
        area = self.query_one(TypingArea)
        area.styles.offset = (tx0 + 2, ty0)
        area.styles.width = max(1, tx1 - tx0 - 1)
        rows = max(1, min(area.rows_needed(), ty1 - ty0 + 1))
        if rows != self.typing_rows or area.size.height != rows:
            self.typing_rows = rows
            area.styles.height = rows

    # -- animation --
    def _tick(self) -> None:
        if not self.eye.active:
            return
        now = time.monotonic()
        dt, self._last = min(0.25, now - self._last), now
        self._eye_rows = self.eye.step()
        if self.eye.motion:
            self._phase += dt * math.tau * (1.4 if self.talking else 0.35)
            self.glow = round((math.sin(self._phase) + 1) / 2 * 12) / 12
        else:
            self.glow = 0.6
        self.refresh()

    # -- mouse --
    def on_click(self, event: events.Click) -> None:
        x0, y0, x1, y1 = eyeart.box_geometry(self.size.width, self.size.height, self.eye.n)["eye"]
        if x0 <= event.x <= x1 and y0 <= event.y <= y1:
            self.eye.click()
        else:                                            # anywhere else in the box: start typing
            self.query_one(TypingArea).focus()
        event.stop()

    # -- drawing --
    def render(self) -> Text:
        w, h, n = self.size.width, self.size.height, self.eye.n
        out = Text(no_wrap=True, overflow="crop")
        if w < eyeart.eye_width(n) + 12 or h < n // 2 + 2:
            return out
        g = eyeart.box_geometry(w, h, n)
        tx0, ty0, tx1, ty1 = g["text"]
        tw = tx1 - tx0 + 1
        bt_row = g["box_top_px"] // 2                     # the box's top edge; the readouts sit just above it
        ew = g["eye_w"]
        hx, room = ew + 5, w - ew - 6                     # to the right of the eye and its bracket
        hud = {bt_row - 3 + i: line for i, line in enumerate(self.hud_lines()) if bt_row - 3 + i >= 0}
        px = eyeart.dialogue_pixels(w, h, n, self.glow, self._eye_rows)
        bg = eyeart.NAVY
        blank = _style((255, 255, 255), eyeart.BOX_FILL)
        prompt = _style(eyeart.BOX_RIM_BRIGHT, eyeart.BOX_FILL, bold=True)
        for r in range(h):
            top, bot = px[2 * r], px[2 * r + 1]
            x = 0
            while x < w:
                if r in hud and x == hx and room > 0:     # the readouts
                    used = 0
                    for txt, st in hud[r]:
                        take = txt[: max(0, room - used)]
                        if take:
                            out.append(take, st)
                            used += cell_len(take)
                    if used:
                        x += used
                        continue
                if ty0 <= r <= ty1 and x == tx0:          # the text area (your typing area sits on top of it)
                    out.append("▸ " if r == ty0 else "  ", prompt)
                    out.append(" " * (tw - 2), blank)
                    x += tw
                    continue
                t, b = top[x], bot[x]
                if t is None and b is None:
                    out.append(" ", _style(bg, bg))
                else:
                    out.append("▀", _style(t or bg, b or bg))
                x += 1
            if r < h - 1:
                out.append("\n")
        return out


# --- the settings window -----------------------------------------------------------
class SettingsScreen(ModalScreen):
    BINDINGS = [("escape", "close", "Close"), ("f2", "close", "Close")]

    def __init__(self, settings: dict) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-box"):
            yield Static("Settings", id="settings-title")
            yield Checkbox("ADAM mode: pixel-art eye and ADAM's voice", self.settings["adam"], id="adam")
            yield Checkbox("Text sounds: a soft blip as letters appear", self.settings["sounds"], id="sounds")
            yield Checkbox("Animation: eye movement and the pulsing frame", self.settings["motion"], id="motion")
            yield Static("Click the eye to make it blink. Three quick clicks: a surprise.\n"
                         "Over SSH, sounds use the terminal bell (turn it on in your terminal's settings).",
                         id="settings-hint")
            yield Button("Done", id="done", variant="success")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        self.app.apply_setting(event.checkbox.id, event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()
