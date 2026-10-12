#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import threading
import time
import urllib.request
from collections import namedtuple

import cv2
import numpy as np
import pygame

try:
    import mediapipe as mp
except ImportError:
    sys.exit("mediapipe is not installed.  Run:  pip install -r requirements.txt")

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

SCALE = 2
STAGE_W, STAGE_H = 480, 360
GAME_W, GAME_H = STAGE_W * SCALE, STAGE_H * SCALE
PANEL_W = 260
WIN_W, WIN_H = GAME_W + PANEL_W, GAME_H
FPS = 60

LANE_X = [-175, -90, 0, 90, 175]
LANE_NAMES = ["DOWN-LEFT", "UP-LEFT", "MIDDLE", "UP-RIGHT", "DOWN-RIGHT"]
LANE_HELP = ["Point finger down-left", "Point finger up-left", "Make a fist",
             "Point finger up-right", "Point finger down-right"]
LANE_ROT = [90, 0, 0, -90, 180]
LANE_COLORS = [(255, 90, 200), (255, 150, 60), (255, 220, 70),
               (90, 220, 255), (150, 120, 255)]

RECEPTOR_Y = 150
SPAWN_Y = -191
SPEED = 200.0
HIT_FROM_Y = 100
MISS_Y = 200
HIT_POINTS = 100
MAX_MISSES = 6
SPAWN_EVERY = 3.0

RAMP = True
RAMP_MIN_INTERVAL = 1.0
RAMP_STEP = 0.02

HOLD_TO_START = 1.0

EXTEND_RATIO = 1.08

DIAG_TOLERANCE = 38
MIN_POINT_LEN = 0.55
CONFIRM_FRAMES = 3
LOSE_HAND_AFTER = 0.25

PIP_W, PIP_H = PANEL_W, int(PANEL_W * 3 / 4)

OPEN = "OPEN"

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
                    (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
                    (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify(pts):
    wrist = pts[0]

    def extended(tip, pip):
        return _dist(pts[tip], wrist) > _dist(pts[pip], wrist) * EXTEND_RATIO

    index, middle, ring, pinky = (extended(8, 6), extended(12, 10),
                                  extended(16, 14), extended(20, 18))
    others = middle + ring + pinky

    if index and middle and ring and pinky:
        return OPEN
    if not index and others == 0:
        return 2
    if index and others == 0:
        vx, vy = pts[8][0] - pts[5][0], pts[8][1] - pts[5][1]
        palm = _dist(wrist, pts[9]) or 1.0
        if math.hypot(vx, vy) < MIN_POINT_LEN * palm:
            return None
        ang = math.degrees(math.atan2(-vy, vx))
        for lane, centre in ((1, 135), (3, 45), (0, -135), (4, -45)):
            if abs((ang - centre + 180) % 360 - 180) <= DIAG_TOLERANCE:
                return lane
    return None


class Stabilizer:
    def __init__(self):
        self.stable = None
        self._cand = None
        self._count = 0
        self._last_seen = 0.0

    def update(self, raw, now):
        if raw is not None:
            self._last_seen = now
            self._count = self._count + 1 if raw == self._cand else 1
            self._cand = raw
            if self._count >= CONFIRM_FRAMES:
                self.stable = raw
        else:
            self._cand, self._count = None, 0
            if now - self._last_seen > LOSE_HAND_AFTER:
                self.stable = None
        return self.stable

class HandDetector:
    def __init__(self):
        self._legacy = None
        self._task = None
        self._t0 = time.monotonic()
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._legacy = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1, model_complexity=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            self._ensure_model()
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO, num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self._task = vision.HandLandmarker.create_from_options(opts)

    @staticmethod
    def _ensure_model():
        if os.path.exists(MODEL_PATH):
            return
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print("Downloading hand model (one time, ~8 MB) ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
            os.replace(MODEL_PATH + ".part", MODEL_PATH)
        except Exception as exc:
            raise RuntimeError(
                f"Could not download the hand model ({exc}).\n"
                f"Download it manually from:\n  {MODEL_URL}\n"
                f"and save it as:\n  {MODEL_PATH}") from exc

    def detect(self, rgb):
        if self._legacy is not None:
            res = self._legacy.process(rgb)
            if res.multi_hand_landmarks:
                return [(p.x, p.y) for p in res.multi_hand_landmarks[0].landmark]
            return None
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000)
        res = self._task.detect_for_video(img, ts)
        if res.hand_landmarks:
            return [(p.x, p.y) for p in res.hand_landmarks[0]]
        return None


Snap = namedtuple("Snap", "gesture raw hand preview error")


class HandInput:
    def __init__(self, camera=0):
        self.camera = camera
        self._lock = threading.Lock()
        self._snap = Snap(None, None, False, None, None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return self._snap

    def _set(self, **kw):
        with self._lock:
            self._snap = self._snap._replace(**kw)

    def _open_camera(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(self.camera, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
        return None

    def _run(self):
        try:
            detector = HandDetector()
        except Exception as exc:
            self._set(error=str(exc))
            return
        cap = self._open_camera()
        if cap is None:
            self._set(error=f"Camera {self.camera} could not be opened.\n"
                            "Close other apps using it, or try --camera 1")
            return
        stab = Stabilizer()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            norm = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pts = [(x * w, y * h) for x, y in norm] if norm else None
            raw = classify(pts) if pts else None
            stable = stab.update(raw, time.monotonic())
            self._draw_overlay(frame, pts, raw)
            small = cv2.cvtColor(cv2.resize(frame, (PIP_W, PIP_H)), cv2.COLOR_BGR2RGB)
            self._set(gesture=stable, raw=raw, hand=pts is not None,
                      preview=small.tobytes(), error=None)
        cap.release()

    @staticmethod
    def _draw_overlay(frame, pts, raw):
        if not pts:
            return
        ip = [(int(x), int(y)) for x, y in pts]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, ip[a], ip[b], (255, 255, 255), 2)
        for p in ip:
            cv2.circle(frame, p, 4, (255, 0, 160), -1)
        if isinstance(raw, int) and raw != 2:
            c = LANE_COLORS[raw]
            cv2.arrowedLine(frame, ip[5], ip[8], (c[2], c[1], c[0]), 5, tipLength=0.35)


def make_blip(freqs, dur=0.12, vol=0.35):
    rate = 44100
    n = int(rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    seg = n // len(freqs)
    wave = np.concatenate([np.sin(2 * np.pi * f * t[:seg]) for f in freqs])
    wave = np.pad(wave, (0, n - len(wave)))
    wave *= np.exp(-t * 18) * vol
    mono = (wave * 32767).astype(np.int16)
    return pygame.mixer.Sound(buffer=np.column_stack([mono, mono]).tobytes())

class Game:
    def __init__(self, assets, sound=None):
        self.a = assets
        self.sound = sound or (lambda name: None)
        self.state = "title"
        self.reset()

    def reset(self):
        self.score = self.misses = self.combo = self.best_combo = self.hits = 0
        self.arrows = []
        self.spawn_in = SPAWN_EVERY
        self.popups = []
        self.flash = [0.0] * 5
        self.hold = 0.0
        self.over_lock = 0.0

    def start(self):
        self.reset()
        self.state = "play"
        self.sound("start")

    def update(self, dt, pose):
        for i in range(5):
            self.flash[i] = max(0.0, self.flash[i] - dt)
        for p in self.popups:
            p[3] += dt
        self.popups = [p for p in self.popups if p[3] < 0.7]

        if self.state != "play":
            self.over_lock = max(0.0, self.over_lock - dt)
            if pose == OPEN and self.over_lock <= 0:
                self.hold += dt
                if self.hold >= HOLD_TO_START:
                    self.start()
            else:
                self.hold = 0.0
            return

        self.spawn_in -= dt
        if self.spawn_in <= 0:
            self.arrows.append([random.randint(0, 4), float(SPAWN_Y)])
            interval = SPAWN_EVERY
            if RAMP:
                interval = max(RAMP_MIN_INTERVAL, SPAWN_EVERY - RAMP_STEP * self.hits)
            self.spawn_in += interval

        alive = []
        for arrow in self.arrows:
            lane, _ = arrow
            arrow[1] += SPEED * dt
            y = arrow[1]
            if HIT_FROM_Y <= y <= MISS_Y and pose == lane:
                self._hit(lane, abs(y - RECEPTOR_Y))
            elif y > MISS_Y:
                self._miss(lane)
            else:
                alive.append(arrow)
        self.arrows = alive

        if self.misses >= MAX_MISSES:
            self.state = "over"
            self.arrows = []
            self.over_lock = 1.5
            self.hold = 0.0
            self.sound("over")

    def _hit(self, lane, off):
        self.score += HIT_POINTS
        self.hits += 1
        self.combo += 1
        self.best_combo = max(self.best_combo, self.combo)
        self.flash[lane] = 0.25
        text = "PERFECT" if off <= 12 else "GREAT" if off <= 30 else "GOOD"
        self.popups.append([text, LANE_COLORS[lane], lane, 0.0])
        self.sound("hit")

    def _miss(self, lane):
        self.misses += 1
        self.combo = 0
        self.popups.append(["MISS", (255, 80, 80), lane, 0.0])
        self.sound("miss")

    @staticmethod
    def _sx(x):
        return int((x + STAGE_W / 2) * SCALE)

    @staticmethod
    def _sy(y):
        return int((STAGE_H / 2 - y) * SCALE)

    def draw(self, screen, snap, pose):
        a = self.a
        screen.blit(a["backdrop"], (0, 0))

        for lane in range(5):
            img = a["recv"][lane]
            active = pose == lane
            img.set_alpha(255 if (active or self.flash[lane] > 0) else 128)
            rect = img.get_rect(center=(self._sx(LANE_X[lane]), self._sy(RECEPTOR_Y)))
            if active:
                pygame.draw.circle(screen, LANE_COLORS[lane], rect.center, 62, 4)
            screen.blit(img, rect)

        for lane, y in self.arrows:
            img = a["arrow"][lane]
            screen.blit(img, img.get_rect(center=(self._sx(LANE_X[lane]), self._sy(y))))

        for text, col, lane, age in self.popups:
            surf = a["font_m"].render(text, True, col)
            surf.set_alpha(int(255 * (1 - age / 0.7)))
            screen.blit(surf, surf.get_rect(center=(self._sx(LANE_X[lane]),
                                                    self._sy(RECEPTOR_Y) + 80 + int(age * 40))))

        if self.state == "title":
            self._draw_title(screen, snap)
        elif self.state == "over":
            self._draw_over(screen)
        self._draw_panel(screen, snap, pose)

    def _dim(self, screen, alpha):
        veil = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        veil.fill((10, 0, 25, alpha))
        screen.blit(veil, (0, 0))

    def _centered(self, screen, font, text, y, col=(255, 255, 255)):
        s = self.a[font].render(text, True, col)
        screen.blit(s, s.get_rect(center=(GAME_W // 2, y)))

    def _hold_ring(self, screen, cx, cy):
        if self.hold > 0:
            rect = pygame.Rect(0, 0, 90, 90)
            rect.center = (cx, cy)
            pygame.draw.circle(screen, (60, 60, 90), rect.center, 45, 8)
            pygame.draw.arc(screen, (120, 255, 160), rect, math.pi / 2,
                            math.pi / 2 + 2 * math.pi * min(1, self.hold / HOLD_TO_START), 8)

    def _draw_title(self, screen, snap):
        self._dim(screen, 175)
        self._centered(screen, "font_xl", "POINT", 200, (255, 120, 220))
        self._centered(screen, "font_xl", "IT UP", 270, (255, 255, 255))
        if snap.error:
            for i, line in enumerate(snap.error.split("\n")):
                self._centered(screen, "font_s", line, 380 + i * 28, (255, 120, 120))
            return
        self._centered(screen, "font_m", "Show an OPEN PALM to the camera", 400)
        self._centered(screen, "font_m", "and hold it to start", 435)
        self._centered(screen, "font_s", "(no keyboard needed - see the sign guide on the right)", 490, (200, 200, 220))
        self._hold_ring(screen, GAME_W // 2, 580)

    def _draw_over(self, screen):
        self._dim(screen, 200)
        go = self.a["gameover"]
        screen.blit(go, go.get_rect(center=(GAME_W // 2, 300)))
        self._centered(screen, "font_l", f"SCORE  {self.score}", 590, (255, 255, 120))
        self._centered(screen, "font_s", f"best combo {self.best_combo}", 628, (220, 220, 240))
        if self.over_lock <= 0:
            self._centered(screen, "font_s", "Hold OPEN PALM to play again", 668)
        self._hold_ring(screen, GAME_W // 2 + 330, 640)

    def _draw_panel(self, screen, snap, pose):
        a = self.a
        x0 = GAME_W
        pygame.draw.rect(screen, (22, 10, 38), (x0, 0, PANEL_W, WIN_H))
        pygame.draw.line(screen, (255, 90, 200), (x0, 0), (x0, WIN_H), 3)

        if snap.preview:
            screen.blit(pygame.image.frombuffer(snap.preview, (PIP_W, PIP_H), "RGB"), (x0, 0))
        else:
            pygame.draw.rect(screen, (40, 30, 60), (x0, 0, PIP_W, PIP_H))
            msg = "NO CAMERA" if snap.error else "starting camera..."
            s = a["font_s"].render(msg, True, (200, 200, 220))
            screen.blit(s, s.get_rect(center=(x0 + PANEL_W // 2, PIP_H // 2)))

        if pose == OPEN:
            name, col = "OPEN PALM", (120, 255, 160)
        elif isinstance(pose, int):
            name, col = LANE_NAMES[pose], LANE_COLORS[pose]
        else:
            name, col = ("no sign" if snap.hand else "no hand"), (150, 150, 170)
        s = a["font_m"].render(name, True, col)
        screen.blit(s, (x0 + 14, PIP_H + 10))

        y = PIP_H + 52
        for lane in (1, 3, 2, 0, 4):
            ic = a["icon"][lane]
            screen.blit(ic, ic.get_rect(center=(x0 + 28, y + 14)))
            t = a["font_xs"].render(LANE_HELP[lane], True,
                                    (255, 255, 255) if pose == lane else (190, 180, 210))
            screen.blit(t, (x0 + 54, y + 6))
            y += 34
        t = a["font_xs"].render("Open palm (hold) = start", True, (120, 255, 160))
        screen.blit(t, (x0 + 14, y + 4))

        screen.blit(a["font_s"].render("SCORE", True, (255, 90, 200)), (x0 + 14, WIN_H - 130))
        screen.blit(a["font_l"].render(str(self.score), True, (255, 255, 255)), (x0 + 14, WIN_H - 108))
        screen.blit(a["font_s"].render(f"combo {self.combo}", True, (255, 220, 70)), (x0 + 14, WIN_H - 62))
        for i in range(MAX_MISSES):
            lit = i >= self.misses
            pygame.draw.circle(screen, (255, 90, 200) if lit else (70, 50, 90),
                               (x0 + 24 + i * 36, WIN_H - 24), 11)


def load_assets():
    def img(name, scale):
        im = pygame.image.load(os.path.join(ASSETS, name)).convert_alpha()
        w, h = im.get_size()
        return pygame.transform.smoothscale(im, (int(w * scale), int(h * scale)))

    arrow, foot = img("arrow.png", 0.25), img("footprint.png", 0.25)
    arrows = [pygame.transform.rotate(arrow, LANE_ROT[i]) if i != 2 else foot for i in range(5)]
    go = pygame.image.load(os.path.join(ASSETS, "gameover.png")).convert_alpha()
    go = pygame.transform.smoothscale(go, (GAME_W, int(GAME_W * go.get_height() / go.get_width())))
    back = pygame.image.load(os.path.join(ASSETS, "backdrop.png")).convert()
    return {
        "backdrop": pygame.transform.smoothscale(back, (GAME_W, GAME_H)),
        "arrow": arrows,
        "recv": [s.copy() for s in arrows],
        "icon": [pygame.transform.smoothscale(s, (int(s.get_width() * 30 / s.get_height()), 30))
                 for s in arrows],
        "gameover": go,
        "font_xl": pygame.font.Font(None, 96),
        "font_l": pygame.font.Font(None, 60),
        "font_m": pygame.font.Font(None, 38),
        "font_s": pygame.font.Font(None, 30),
        "font_xs": pygame.font.Font(None, 24),
    }


def make_sound_player(use_music):
    sounds, music = {}, None
    try:
        sounds = {"hit": make_blip([880, 1320], 0.12), "miss": make_blip([200, 140], 0.2),
                  "start": make_blip([523, 659, 784], 0.3), "over": make_blip([392, 330, 262], 0.5)}
        if use_music:
            music = pygame.mixer.Sound(os.path.join(ASSETS, "music.wav"))
            music.set_volume(0.5)
    except pygame.error:
        return lambda name: None

    def play(name):
        if name == "start" and music:
            music.stop()
            music.play(-1)
        elif name == "over" and music:
            music.stop()
        if name in sounds:
            sounds[name].play()
    return play


def main():
    ap = argparse.ArgumentParser(description="Point It Up")
    ap.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--no-music", action="store_true", help="turn the background music off")
    args = ap.parse_args()

    try:
        pygame.mixer.pre_init(44100, -16, 2, 512)
    except pygame.error:
        pass
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Point It Up")
    game = Game(load_assets(), make_sound_player(not args.no_music))

    hands = HandInput(args.camera)
    hands.start()
    clock = pygame.time.Clock()
    try:
        running = True
        while running:
            dt = min(clock.tick(FPS) / 1000.0, 0.05)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
            snap = hands.snapshot()
            game.update(dt, snap.gesture)
            game.draw(screen, snap, snap.gesture)
            pygame.display.flip()
    finally:
        hands.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
