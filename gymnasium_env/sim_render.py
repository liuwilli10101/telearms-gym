"""Camera-realistic rendering with visual domain randomization, for replayed demos.

The real cameras are RealSense units with 16:9 colour streams: a D435IF on top
(colour 69 x 42 deg) and a D405 on each wrist (87 x 58 deg). The real robot letterboxes
them to 224x224 with antialiasing (full frame, centred black bars, no crop) before the
policy sees them. Here each camera renders at REAL_WIDTH x REAL_HEIGHT with its real pose
and field of view (apply_real_cameras), and the frame goes through the same letterbox
(Letterbox), so training sees what deployment will.

Visual randomization (per episode, seeded): light pose, colour and strength;
procedural table, floor and sky textures in low-saturation colours (task colours
stay distinguishable); small camera pose and field-of-view jitter. Sensor noise
belongs in training-time augmentation: video compression would remove most of it.
None of it changes physics, so recorded actions stay valid.
"""
from __future__ import annotations

import colorsys

import numpy as np

REAL_WIDTH, REAL_HEIGHT = 640, 360   # assumed 16:9 camera mode
OUT = 224                            # policy input
# A smaller shadow map than the scene's 4096 renders ~2.5x faster; lighting is randomized anyway.
SHADOW_SIZE = 1024
KEEP_TEXTURE = 0.15                   # chance a surface keeps its original texture

# The real cameras. The top D435IF was measured on 12 Oct 2026 for sim2real (Helen's simtoreal
# branch, sim2real/real_cameras.py: depth plane fit, runs/real_static/20261012_final) at 0.72 m
# above the table top, 15.6 deg off vertical, centred between the arms, lens above the arm bases
# (x=0.21). At 0.72 m a cube leaves the view in ~12 % of demos, so it moves to a fixed mount at
# 0.95 m (none leaves the view), same lens x and tilt: TOP_POSITION looking at TOP_TARGET, sim
# world coordinates. Re-measure once mounted. Vertical fields of view of the 1280x720 colour
# streams from the RealSense intrinsics. The wrist cameras keep the sim's mounts, which are the
# real arms'.
TOP_POSITION = (0.21, 0.0, 1.90)
TOP_TARGET = (0.21 + 0.95 * 0.20 / 0.72, 0.0, 0.95)
VFOV = {"front": 43.11, "left": 58.0, "right": 57.99}
# Principal point minus image centre (px at 1280x720). Sim renders with a centred principal
# point, so each camera is turned to look along its real image-centre ray (top: ~1.1 deg).
PRINCIPAL_OFFSET_PX = {"front": (-1.5, 18.1), "left": (-5.8, -10.6), "right": (7.1, 0.5)}
NATIVE_HEIGHT = 720


def apply_real_cameras(model, cameras: dict[str, str]) -> None:
    """Put the sim cameras where the real ones are: the top camera at its measured pose, every
    camera at its real vertical field of view and turned by its principal-point offset. Call
    after the workspace_v2 camera profile (keys front, left, right)."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cameras["front"])
    body = model.cam_bodyid[cid]
    if model.body_weldid[body] != 0:
        raise ValueError(f"expected the top camera on a fixed body, got body {body}")
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    position, target = np.array(TOP_POSITION), np.array(TOP_TARGET)
    back = (position - target) / np.linalg.norm(position - target)
    right = np.cross([0.0, 0.0, 1.0], back)
    right /= np.linalg.norm(right)
    rotation = np.column_stack((right, np.cross(back, right), back))  # camera x right, y up, looks along -z
    frame = data.xmat[body].reshape(3, 3)
    model.cam_pos[cid] = frame.T @ (position - data.xpos[body])
    mujoco.mju_mat2Quat(model.cam_quat[cid], (frame.T @ rotation).reshape(9))
    for key, name in cameras.items():
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        model.cam_fovy[cam] = VFOV[key]
        # The real image centre lies (-dx/f, +dy/f) off the optical axis: pitch by atan(dy/f), yaw by atan(dx/f).
        f = NATIVE_HEIGHT / 2 / np.tan(np.deg2rad(VFOV[key]) / 2)
        dx, dy = PRINCIPAL_OFFSET_PX[key]
        pitch, yaw, turn, quat = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(pitch, np.array([1.0, 0.0, 0.0]), np.arctan(dy / f))
        mujoco.mju_axisAngle2Quat(yaw, np.array([0.0, 1.0, 0.0]), np.arctan(dx / f))
        mujoco.mju_mulQuat(turn, yaw, pitch)
        mujoco.mju_mulQuat(quat, model.cam_quat[cam].copy(), turn)
        model.cam_quat[cam] = quat


# ---------------------------------------------------------------------------
# Letterbox like the real robot's (sim2real/real_cameras.letterbox on Helen's simtoreal branch):
# the full frame scaled to 224 wide with antialiasing, centred black padding. The real side area-
# averages its 1280x720 frames; bilinear with antialias on the 640x360 render comes close to it
# (uint8, channels-last kernel). The policy gets 224x224 images, so its own resize never runs.
# ---------------------------------------------------------------------------
class Letterbox:
    def __init__(self, width: int = REAL_WIDTH, height: int = REAL_HEIGHT, out: int = OUT):
        ratio = max(width / out, height / out)
        self.rw, self.rh = int(width / ratio), int(height / ratio)
        self.top = (out - self.rh) // 2
        self.left = (out - self.rw) // 2
        self.out = out

    def batch(self, images: np.ndarray, flip: bool = False) -> np.ndarray:
        """(N, H, W, 3) uint8 at the camera resolution -> (N, out, out, 3) uint8.
        flip: the images are upside down (an OpenGL readback); resizing commutes with the flip."""
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)  # channels-last, for the uint8 kernel
        resized = F.interpolate(t, size=(self.rh, self.rw), mode="bilinear", align_corners=False, antialias=True)
        if flip:
            resized = resized.flip(2)
        frames = np.zeros((len(images), self.out, self.out, 3), dtype=np.uint8)
        frames[:, self.top:self.top + self.rh, self.left:self.left + self.rw] = resized.permute(0, 2, 3, 1).numpy()
        return frames

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.batch(np.ascontiguousarray(image)[None])[0]


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------
def _muted_color(rng: np.random.Generator, saturation: float = 0.25) -> np.ndarray:
    h, s, v = rng.random(), rng.uniform(0.0, saturation), rng.uniform(0.3, 0.9)
    return np.array(colorsys.hsv_to_rgb(h, s, v), dtype=np.float32) * 255


def random_texture(rng: np.random.Generator, height: int, width: int, saturation: float = 0.25) -> np.ndarray:
    """A flat, checker, gradient, noise or grain pattern in two muted colours, (H, W, 3) uint8.
    Drawn at half resolution for large textures (finer detail is lost in rendering anyway)."""
    f = 2 if min(height, width) >= 512 else 1
    h, w = -(-height // f), -(-width // f)
    a, b = _muted_color(rng, saturation), _muted_color(rng, saturation)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) * f
    kind = rng.choice(["flat", "checker", "gradient", "noise", "grain"])
    if kind == "flat":
        t = np.zeros((h, w), np.float32)
    elif kind == "checker":
        period = rng.uniform(16, 128)
        t = ((np.floor(xx / period) + np.floor(yy / period)) % 2).astype(np.float32)
    elif kind == "gradient":
        angle = rng.uniform(0, np.pi)
        u = xx * np.cos(angle) + yy * np.sin(angle)
        t = (u - u.min()) / max(float(np.ptp(u)), 1.0)
    elif kind == "noise":
        cells = int(rng.integers(4, 32))
        coarse = rng.random((cells + 1, cells + 1)).astype(np.float32)
        gy, gx = yy / height * cells, xx / width * cells
        y0, x0 = gy.astype(int), gx.astype(int)
        fy, fx = gy - y0, gx - x0
        y1, x1 = np.minimum(y0 + 1, cells), np.minimum(x0 + 1, cells)
        t = (coarse[y0, x0] * (1 - fx) + coarse[y0, x1] * fx) * (1 - fy) + (coarse[y1, x0] * (1 - fx) + coarse[y1, x1] * fx) * fy
    else:  # wood-like grain
        freq, wobble = rng.uniform(0.02, 0.15), rng.uniform(5, 40)
        t = 0.5 + 0.5 * np.sin(yy * freq + wobble * np.sin(xx * 0.01 + rng.uniform(0, 6)))
    image = a * (1 - t[..., None]) + b * t[..., None]
    image += 3 * rng.standard_normal(image.shape, dtype=np.float32)  # surface variation
    image = np.clip(image, 0, 255).astype(np.uint8)
    if f > 1:
        image = image.repeat(f, axis=0).repeat(f, axis=1)[:height, :width]
    return image


def _hsv_color(rng: np.random.Generator, hue, sat, val) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb(rng.uniform(*hue), rng.uniform(*sat), rng.uniform(*val)), np.float32) * 255


def _streaks(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    """Smooth noise in [0, 1] that varies along y and barely along x: grain, fibres, folds."""
    ny, nx = cells, max(2, cells // 16)
    coarse = rng.random((ny + 1, nx + 1)).astype(np.float32)
    gy = np.linspace(0, ny, h, dtype=np.float32)[:, None]
    gx = np.linspace(0, nx, w, dtype=np.float32)[None, :]
    y0, x0 = np.minimum(gy.astype(int), ny - 1), np.minimum(gx.astype(int), nx - 1)
    fy, fx = gy - y0, gx - x0
    fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)  # smoothstep: no visible cell edges
    return ((coarse[y0, x0] * (1 - fx) + coarse[y0, x0 + 1] * fx) * (1 - fy)
            + (coarse[y0 + 1, x0] * (1 - fx) + coarse[y0 + 1, x0 + 1] * fx) * fy)


# Table tops the real-scene dressing samples from: (weight, name, look, hue, saturation, value,
# contrast). Policies are tested on a wooden table (the real one is light oak, RGB ~218/165/105
# in photos of it), so wood is drawn 80 % of the time; other lab surfaces take the other 20 %.
# Each draws its colour within its ranges, and the profile's lighting, tint and sensor effects
# act on top. TABLE_STYLES_VERSION is recorded with each render.
TABLE_STYLES = [
    (0.50, "oak", "wood", (0.065, 0.11), (0.35, 0.6), (0.7, 0.95), (0.06, 0.2)),
    (0.13, "maple", "wood", (0.09, 0.13), (0.15, 0.35), (0.8, 0.97), (0.04, 0.12)),
    (0.085, "walnut", "wood", (0.04, 0.08), (0.4, 0.7), (0.25, 0.5), (0.08, 0.25)),
    (0.085, "mdf", "wood", (0.08, 0.11), (0.2, 0.4), (0.55, 0.8), (0.02, 0.06)),
    (0.08, "white_laminate", "plain", (0.0, 1.0), (0.0, 0.06), (0.75, 0.95), (0.0, 0.03)),
    (0.08, "dark_laminate", "plain", (0.0, 1.0), (0.0, 0.08), (0.06, 0.3), (0.0, 0.05)),
    (0.04, "steel", "brushed", (0.55, 0.65), (0.0, 0.08), (0.45, 0.75), (0.04, 0.12)),
]
TABLE_STYLES_VERSION = 3


def table_texture(rng: np.random.Generator, height: int, width: int) -> tuple[np.ndarray, str]:
    """A table top drawn from TABLE_STYLES; grain and brushing run along x. Returns the
    (H, W, 3) uint8 texture (drawn at half resolution) and the style's name."""
    weights = np.array([s[0] for s in TABLE_STYLES])
    _, name, look, hue, sat, val, contrast = TABLE_STYLES[rng.choice(len(TABLE_STYLES), p=weights / weights.sum())]
    f = 2 if min(height, width) >= 512 else 1
    h, w = -(-height // f), -(-width // f)
    base = _hsv_color(rng, hue, sat, val)
    if look == "wood":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        warp = rng.uniform(4, 20) * np.sin(xx / w * rng.uniform(2, 8) + rng.uniform(0, 6)) + 30 * _streaks(rng, h, w, 32)
        rings = 0.5 + 0.5 * np.sin((yy + warp) * rng.uniform(0.08, 0.3))
        t = 0.55 * rings + 0.45 * _streaks(rng, h, w, int(rng.integers(64, 256)))
    elif look == "brushed":
        t = _streaks(rng, h, w, int(rng.integers(256, 512)))
    else:  # plain laminate: only a faint, large-scale unevenness
        t = _streaks(rng, h, w, 4) * 0.5 + _streaks(rng, w, h, 4).T * 0.5
    shade = 1 + rng.uniform(*contrast) * (t - 0.5) * 2
    image = base * shade[..., None] + rng.uniform(2, 5) * rng.standard_normal((h, w, 3), dtype=np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if f > 1:
        image = image.repeat(f, axis=0).repeat(f, axis=1)[:height, :width]
    return image, name


def cloth_texture(rng: np.random.Generator, height: int, width: int, val=(0.02, 0.14)) -> np.ndarray:
    """Black fabric like the real backdrop curtain: soft vertical folds and a little weave noise;
    val: its brightness range. (H, W, 3) uint8, drawn at half resolution."""
    f = 2 if min(height, width) >= 512 else 1
    h, w = -(-height // f), -(-width // f)
    base = _hsv_color(rng, (0.55, 0.65), (0.0, 0.12), val)
    folds = _streaks(rng, w, h, int(rng.integers(8, 40))).T  # varies across x: vertical folds
    shade = 1 + rng.uniform(0.15, 0.45) * (folds - 0.5) * 2
    image = base * shade[..., None] + 3 * rng.standard_normal((h, w, 3), dtype=np.float32)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if f > 1:
        image = image.repeat(f, axis=0).repeat(f, axis=1)[:height, :width]
    return image


# ---------------------------------------------------------------------------
# Per-episode visual randomization of a MuJoCo model
# ---------------------------------------------------------------------------
def _small_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    import mujoco
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    quat = np.zeros(4)
    mujoco.mju_axisAngle2Quat(quat, axis, np.deg2rad(rng.uniform(0, max_deg)))
    return quat


# How hard each DR render randomizes. The top camera's fixed mount is set by hand, so every
# profile tilts it up to +-5 deg up or down and +-3 deg sideways (top_pitch, top_yaw) on top of its
# small pose jitter; at 0.95 m no cube leaves the view at the start even at the extremes. "default" is what every DR render used before these
# existed (and still the first render of an episode); further renders of the same episode
# can take a wider profile, so one demo is seen under very different conditions.
DR_PROFILES = {
    "default": dict(light_pos=0.2, light_rot=12.0, gain=(0.6, 1.2), tint=(0.92, 1.08), specular=(0.5, 1.5),
                    ambient=(0.0, 0.1), headlight=(0.7, 1.3), keep_texture=KEEP_TEXTURE, texture_saturation=0.25,
                    top_reach=0.02, top_rot=2.0, top_fovy=2.0, top_pitch=5.0, top_yaw=3.0,
                    wrist_reach=0.005, wrist_rot=2.0, wrist_fovy=2.0,
                    reaim=False, object_color=None, image=None, real_scene=0.0),
    # Viewpoint: the top camera moves up to +-15 cm per axis (~26 cm across) and is re-aimed at the
    # point it looked at, so the view angle really changes while the workspace stays in frame.
    "viewpoint": dict(light_pos=0.2, light_rot=12.0, gain=(0.6, 1.2), tint=(0.92, 1.08), specular=(0.5, 1.5),
                      ambient=(0.0, 0.1), headlight=(0.7, 1.3), keep_texture=KEEP_TEXTURE, texture_saturation=0.25,
                      top_reach=0.15, top_rot=8.0, top_fovy=8.0, top_pitch=5.0, top_yaw=3.0,
                      wrist_reach=0.02, wrist_rot=8.0, wrist_fovy=5.0,
                      reaim=True, object_color=None, image="mild", real_scene=1.0),
    # Appearance: harsh, tinted, moving light; every surface retextured; cube and patch colours
    # shifted (still recognizably their colour); a real sensor's exposure / white balance / gamma.
    "appearance": dict(light_pos=0.5, light_rot=35.0, gain=(0.3, 1.7), tint=(0.75, 1.25), specular=(0.2, 2.0),
                       ambient=(0.0, 0.25), headlight=(0.4, 1.6), keep_texture=0.0, texture_saturation=0.35,
                       top_reach=0.04, top_rot=4.0, top_fovy=3.0, top_pitch=5.0, top_yaw=3.0,
                       wrist_reach=0.008, wrist_rot=4.0, wrist_fovy=3.0,
                       reaim=False, object_color=dict(hue=0.025, sat=(0.8, 1.2), val=(0.75, 1.25)), image="strong",
                       real_scene=1.0),
}
# Realistic: the default's small camera and light ranges in the real-scene dressing (synthetic
# variants' one render).
DR_PROFILES["realistic"] = dict(DR_PROFILES["default"], real_scene=1.0)
# real_scene: the chance an episode is dressed like the real setup (photos of it): a table top
# from TABLE_STYLES (mostly the real light oak), a black backdrop curtain, a dark floor, a black
# robot stand; otherwise the random textures above. Drawn only by profiles that use it, so
# "default" draws as before.
TOP_TARGET_DISTANCE = float(np.linalg.norm(np.subtract(TOP_TARGET, TOP_POSITION)))


def _reaim(model, cid: int, offset: np.ndarray, distance: float) -> None:
    """Move a camera by `offset` (its parent frame) and turn it to look at the point it looked
    at from `distance` before, by the smallest rotation (keeps its roll)."""
    import mujoco
    quat, pos = model.cam_quat[cid].copy(), model.cam_pos[cid].copy()
    forward = np.zeros(3)
    mujoco.mju_rotVecQuat(forward, np.array([0.0, 0.0, -1.0]), quat)  # cameras look along -z
    target = pos + forward * distance
    new_forward = target - (pos + offset)
    new_forward /= np.linalg.norm(new_forward)
    axis = np.cross(forward, new_forward)
    s, c = float(np.linalg.norm(axis)), float(np.dot(forward, new_forward))
    if s > 1e-9:
        turn, turned = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, axis / s, float(np.arctan2(s, c)))
        mujoco.mju_mulQuat(turned, turn, quat)
        model.cam_quat[cid] = turned
    model.cam_pos[cid] = pos + offset


class ImageAug:
    """A real sensor's response, per camera and constant over an episode (exposure gain, white
    balance, gamma, contrast, saturation, sometimes a slight blur). Applied to the letterboxed
    frame's image area only: the black bars stay black, as the policy's own letterbox makes
    them. Per-channel response is one lookup table per camera. No per-frame noise: it made the
    videos 3-5x larger, and training-time augmentation adds it more cheaply."""

    LEVELS = {
        "mild": dict(gain=(0.85, 1.15), wb=(0.95, 1.05), gamma=(0.9, 1.1), contrast=(0.9, 1.1),
                     saturation=(0.9, 1.1), blur=0.0),
        "strong": dict(gain=(0.65, 1.35), wb=(0.85, 1.15), gamma=(0.75, 1.35), contrast=(0.75, 1.25),
                       saturation=(0.7, 1.3), blur=0.3),
    }

    def __init__(self, level: str, rng: np.random.Generator, keys, letterbox: "Letterbox"):
        p = self.LEVELS[level]
        self.box = letterbox
        self.params, self._lut = {}, {}
        for key in keys:
            q = {"gain": float(rng.uniform(*p["gain"])), "wb": rng.uniform(*p["wb"], 3).round(4).tolist(),
                 "gamma": float(rng.uniform(*p["gamma"])), "contrast": float(rng.uniform(*p["contrast"])),
                 "saturation": float(rng.uniform(*p["saturation"])), "blur": bool(rng.random() < p["blur"])}
            self.params[key] = q
            v = np.arange(256, dtype=np.float32)[:, None] / 255.0
            v = np.clip(v * q["gain"] * np.asarray(q["wb"], np.float32)[None], 0.0, 1.0) ** q["gamma"]
            self._lut[key] = np.clip((v - 0.5) * q["contrast"] + 0.5, 0.0, 1.0).astype(np.float32)  # (256, 3)

    def __call__(self, key: str, frame: np.ndarray) -> np.ndarray:
        q, b, lut = self.params[key], self.box, self._lut[key]
        area = (slice(b.top, b.top + b.rh), slice(b.left, b.left + b.rw))
        x = frame[area]
        x = np.stack([lut[x[..., c], c] for c in range(3)], axis=-1)
        gray = x.mean(axis=2, keepdims=True)
        x = gray + (x - gray) * q["saturation"]
        if q["blur"]:
            x = (4 * x + np.roll(x, 1, 0) + np.roll(x, -1, 0) + np.roll(x, 1, 1) + np.roll(x, -1, 1)) / 8
        out = frame.copy()
        out[area] = (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return out


class VisualRandomizer:
    """Randomize lights, background textures and cameras for one episode, then put them back."""

    def __init__(self, model):
        import mujoco
        self.model = model
        # Sky, floor and table: the RGB textures that are not cube maps (cube maps are letter faces).
        self.textures = [t for t in range(model.ntex)
                         if model.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_CUBE and model.tex_nchannel[t] == 3]
        # What real_scene dresses: the table top's texture and material, and the robots' stand.
        self.table_tex = [t for t in self.textures if model.texture(t).name.split("/")[-1] == "table"]
        texid = model.mat_texid.reshape(model.nmat, -1)
        self.table_mats = [i for i in range(model.nmat) if set(self.table_tex) & set(texid[i].tolist())]
        self.stand = [g for g in range(model.ngeom) if model.geom(g).name.endswith("base_stand")]
        model.vis.quality.shadowsize = SHADOW_SIZE
        self._saved = None

    def _texture_slice(self, tex: int) -> slice:
        m = self.model
        start = int(m.tex_adr[tex])
        return slice(start, start + int(m.tex_height[tex] * m.tex_width[tex] * 3))

    def randomize(self, rng: np.random.Generator, cameras: dict[str, str], wrist_keys=("left", "right"),
                  profile: str = "default", colored=None) -> dict:
        """Call after the task's reset and camera profile; upload textures before rendering.
        profile: a DR_PROFILES key. colored: geom-id lists of the task's coloured cubes and
        patches, one list per object (their colour is jittered by profiles that do so)."""
        import mujoco
        m, p = self.model, DR_PROFILES[profile]
        cids = {key: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) for key, name in cameras.items()}
        colored = [list(ids) for ids in (colored or []) if len(ids)] if p["object_color"] else []
        self._saved = {
            "light": [getattr(m, f).copy() for f in ("light_pos", "light_dir", "light_diffuse", "light_specular", "light_ambient")],
            "headlight": (m.vis.headlight.diffuse.copy(), m.vis.headlight.ambient.copy()),
            "cam": {cid: (m.cam_pos[cid].copy(), m.cam_quat[cid].copy(), float(m.cam_fovy[cid])) for cid in cids.values()},
            "tex": {t: m.tex_data[self._texture_slice(t)].copy() for t in self.textures},
            "rgba": {g: m.geom_rgba[g].copy() for g in [g for ids in colored for g in ids] + self.stand},
            "mat": {i: m.mat_rgba[i].copy() for i in self.table_mats},
        }
        params: dict = {"profile": profile, "lights": [], "cameras": {}}
        a = p["light_pos"]
        for i in range(m.nlight):
            m.light_pos[i] += rng.uniform([-a, -a, -a], [a, a, a])
            direction = np.zeros(3)
            mujoco.mju_rotVecQuat(direction, m.light_dir[i].copy(), _small_rotation(rng, p["light_rot"]))
            m.light_dir[i] = direction / np.linalg.norm(direction)
            gain, tint = rng.uniform(*p["gain"]), rng.uniform(*p["tint"], 3)
            m.light_diffuse[i] = np.clip(m.light_diffuse[i] * gain * tint, 0, 1)
            m.light_specular[i] = np.clip(m.light_specular[i] * rng.uniform(*p["specular"]), 0, 1)
            m.light_ambient[i] = np.clip(m.light_ambient[i] + rng.uniform(*p["ambient"]), 0, 1)
            params["lights"].append({"pos": m.light_pos[i].round(3).tolist(), "dir": m.light_dir[i].round(3).tolist(),
                                     "diffuse": m.light_diffuse[i].round(3).tolist()})
        m.vis.headlight.diffuse[:] = np.clip(m.vis.headlight.diffuse * rng.uniform(*p["headlight"]), 0, 1)
        m.vis.headlight.ambient[:] = np.clip(m.vis.headlight.ambient * rng.uniform(*p["headlight"]), 0, 1)
        params["headlight"] = {"diffuse": m.vis.headlight.diffuse.round(3).tolist(),
                               "ambient": m.vis.headlight.ambient.round(3).tolist()}

        params["textures"] = {}
        real = p["real_scene"] > 0 and rng.random() < p["real_scene"]
        for tex in self.textures if real else []:
            h, w = int(m.tex_height[tex]), int(m.tex_width[tex])
            if tex in self.table_tex:
                image, kind = table_texture(rng, h, w)
            elif m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D:
                image, kind = cloth_texture(rng, h, w, (0.04, 0.3)), "floor"
            else:
                image, kind = cloth_texture(rng, h, w, (0.02, 0.14)), "curtain"
            m.tex_data[self._texture_slice(tex)] = image.reshape(-1)
            params["textures"][m.texture(tex).name] = kind
        if real:
            params["table_styles"] = TABLE_STYLES_VERSION
            for i in self.table_mats:  # the texture carries the colour
                m.mat_rgba[i][:3] = rng.uniform(0.9, 1.0)
            for g in self.stand:
                m.geom_rgba[g][:3] = rng.uniform(0.04, 0.15)
        for tex in [] if real else self.textures:
            if rng.random() < p["keep_texture"]:
                params["textures"][m.texture(tex).name] = "original"
                continue
            h, w = int(m.tex_height[tex]), int(m.tex_width[tex])
            m.tex_data[self._texture_slice(tex)] = random_texture(rng, h, w, p["texture_saturation"]).reshape(-1)
            params["textures"][m.texture(tex).name] = "random"

        for key, cid in cids.items():
            wrist = key in wrist_keys
            reach = p["wrist_reach"] if wrist else p["top_reach"]
            if p["reaim"] and not wrist:
                _reaim(m, cid, rng.uniform(-reach, reach, 3), TOP_TARGET_DISTANCE)
            else:
                m.cam_pos[cid] += rng.uniform(-reach, reach, 3)
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, m.cam_quat[cid].copy(), _small_rotation(rng, p["wrist_rot"] if wrist else p["top_rot"]))
            m.cam_quat[cid] = quat
            if not wrist:  # tilt about the camera's x (right) axis, turn about its y (up) axis
                tilt, pan = rng.uniform(-p["top_pitch"], p["top_pitch"]), rng.uniform(-p["top_yaw"], p["top_yaw"])
                pitch, yaw, turn = np.zeros(4), np.zeros(4), np.zeros(4)
                mujoco.mju_axisAngle2Quat(pitch, np.array([1.0, 0.0, 0.0]), np.deg2rad(tilt))
                mujoco.mju_axisAngle2Quat(yaw, np.array([0.0, 1.0, 0.0]), np.deg2rad(pan))
                mujoco.mju_mulQuat(turn, yaw, pitch)
                mujoco.mju_mulQuat(quat, m.cam_quat[cid].copy(), turn)
                m.cam_quat[cid] = quat
            fovy = p["wrist_fovy"] if wrist else p["top_fovy"]
            m.cam_fovy[cid] += rng.uniform(-fovy, fovy)
            params["cameras"][key] = {"pos": m.cam_pos[cid].round(4).tolist(), "quat": m.cam_quat[cid].round(4).tolist(),
                                      "fovy": round(float(m.cam_fovy[cid]), 2)}
            if not wrist:
                params["cameras"][key]["tilt_pan"] = [round(float(tilt), 2), round(float(pan), 2)]

        if colored:  # each object's colour moves in HSV, its geoms together; alpha untouched
            jitter = p["object_color"]
            params["object_colors"] = []
            for ids in colored:
                r, g, b, alpha = (float(x) for x in m.geom_rgba[ids[0]])
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                h = (h + rng.uniform(-jitter["hue"], jitter["hue"])) % 1.0
                s = min(1.0, s * rng.uniform(*jitter["sat"]))
                v = min(1.0, v * rng.uniform(*jitter["val"]))
                rgb = colorsys.hsv_to_rgb(h, s, v)
                for gid in ids:
                    m.geom_rgba[gid][:3] = rgb
                params["object_colors"].append([round(x, 3) for x in rgb])
        return params

    def restore(self) -> None:
        if self._saved is None:
            return
        m, saved = self.model, self._saved
        for field, value in zip(("light_pos", "light_dir", "light_diffuse", "light_specular", "light_ambient"), saved["light"]):
            getattr(m, field)[:] = value
        m.vis.headlight.diffuse[:], m.vis.headlight.ambient[:] = saved["headlight"]
        for cid, (pos, quat, fovy) in saved["cam"].items():
            m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = pos, quat, fovy
        for tex, value in saved["tex"].items():
            m.tex_data[self._texture_slice(tex)] = value
        for gid, rgba in saved.get("rgba", {}).items():
            m.geom_rgba[gid] = rgba
        for mat, rgba in saved.get("mat", {}).items():
            m.mat_rgba[mat] = rgba
        self._saved = None


# The real cell under the arms (Helen's simtoreal branch, sim2real/real_cameras.py): the arms
# stand on a dark aluminium extrusion on top of the table, with table top all around them; in sim
# they stand on a grey box behind the table's front edge. Sim world frame: table top z=0.95, arm
# bases at x=0.20, z=0.96; the sim table top spans x 0.30..0.90.
TABLE_CENTER_X, TABLE_COPY_CENTER_X = 0.60, 0.0
EXTRUSION_X = (0.08, 0.21)            # front edge right below the top camera's lens
EXTRUSION_HALF_Y = 0.55
EXTRUSION_RGBA = (0.08, 0.13, 0.18, 1.0)  # dark blue-grey anodised, as photographed
TABLE_TOP_Z, ARM_BASE_Z = 0.95, 0.96


class RealCell:
    """Scene hook (after update_scene) that makes the render look like the real cell: hides the
    stand under the arms, adds a second copy of the table-top mesh under them (same mesh, UVs and
    material, so it takes any DR texture) turned 180 deg so its raised lip ends up behind the
    arms, and the extrusion bar. Only in the render scene, never in the model: nothing collides
    with them and recorded trajectories stay valid."""

    _COPIED = ("type", "dataid", "matid", "category", "texcoord", "size", "rgba", "emission",
               "specular", "shininess", "reflectance", "transparent", "segid")

    def __init__(self, model):
        import mujoco
        self._mj = mujoco
        names = [model.geom(i).name for i in range(model.ngeom)]
        self.stand = {i for i, n in enumerate(names) if n == "base_stand" or n.endswith("/base_stand")}
        top = [i for i, n in enumerate(names) if (n == "mesh" or n.endswith("table/mesh"))
               and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        if len(top) != 1:
            raise ValueError(f"expected one table-top mesh geom, found {len(top)}")
        self.table_top = top[0]
        self.flip = np.diag([-1.0, -1.0, 1.0])  # 180 deg about z

    def _new_geom(self, scene):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("render scene full; cannot add the real cell")
        scene.ngeom += 1
        return scene.geoms[scene.ngeom - 1]

    def __call__(self, scene) -> None:
        mujoco = self._mj
        table = None
        for i in range(scene.ngeom):
            g = scene.geoms[i]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                if g.objid in self.stand:
                    g.rgba[3] = 0.0
                elif g.objid == self.table_top:
                    table = g
        if table is None:
            raise RuntimeError("table top not in the render scene")
        copy = self._new_geom(scene)
        mujoco.mjv_initGeom(copy, mujoco.mjtGeom.mjGEOM_BOX, np.ones(3), np.zeros(3), np.eye(3).reshape(9),
                            np.ones(4, np.float32))
        for field in self._COPIED:
            value = getattr(table, field)
            if isinstance(value, np.ndarray):
                getattr(copy, field)[...] = value
            else:
                setattr(copy, field, value)
        copy.pos[:] = self.flip @ (np.asarray(table.pos) - [TABLE_CENTER_X, 0.0, 0.0]) + [TABLE_COPY_CENTER_X, 0.0, 0.0]
        copy.mat[:] = self.flip @ np.asarray(table.mat).reshape(3, 3)
        copy.objtype, copy.objid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1
        bar = self._new_geom(scene)
        size = np.array([(EXTRUSION_X[1] - EXTRUSION_X[0]) / 2, EXTRUSION_HALF_Y, (ARM_BASE_Z - TABLE_TOP_Z) / 2])
        pos = np.array([sum(EXTRUSION_X) / 2, 0.0, (ARM_BASE_Z + TABLE_TOP_Z) / 2])
        mujoco.mjv_initGeom(bar, mujoco.mjtGeom.mjGEOM_BOX, size, pos, np.eye(3).reshape(9),
                            np.asarray(EXTRUSION_RGBA, np.float32))
        bar.category = mujoco.mjtCatBit.mjCAT_STATIC


class WideRenderer:
    """All cameras side by side at the real resolution, one GPU readback, letterboxed to 224.
    The scene is dressed as the real cell (RealCell)."""

    def __init__(self, model, cameras: dict[str, str], width: int = REAL_WIDTH, height: int = REAL_HEIGHT):
        import mujoco
        self._mj = mujoco
        self.real_cell = RealCell(model)
        self.keys, self.cams = [], []
        for key, name in cameras.items():
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            self.keys.append(key)
            self.cams.append(camera)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width * len(self.cams))
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width * len(self.cams))
        self.tiles = [mujoco.MjrRect(i * width, 0, width, height) for i in range(len(self.cams))]
        self.viewport = mujoco.MjrRect(0, 0, width * len(self.cams), height)
        self.pixels = np.empty((height, width * len(self.cams), 3), dtype=np.uint8)
        self.width = width
        self.letterbox = Letterbox(width, height)
        self._model = model

    def render(self, data) -> dict[str, np.ndarray]:
        r = self.renderer
        if r._gl_context:
            r._gl_context.make_current()
        for camera, tile in zip(self.cams, self.tiles):
            r.update_scene(data, camera=camera)
            self.real_cell(r.scene)
            self._mj.mjr_render(tile, r.scene, r._mjr_context)
        self._mj.mjr_readPixels(self.pixels, None, self.viewport, r._mjr_context)
        tiles = self.pixels.reshape(self.pixels.shape[0], len(self.keys), self.width, 3).transpose(1, 0, 2, 3)
        frames = self.letterbox.batch(tiles, flip=True)
        return {key: frames[i] for i, key in enumerate(self.keys)}

    def upload_textures(self) -> None:
        """Send the model's current textures to the GPU (after changing tex_data)."""
        r = self.renderer
        if r._gl_context:
            r._gl_context.make_current()
        for tex in range(self._model.ntex):
            self._mj.mjr_uploadTexture(self._model, r._mjr_context, tex)

    def close(self) -> None:
        self.renderer.close()
