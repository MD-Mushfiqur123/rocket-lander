# ═══════════════════════════════════════════════════════════════════════════
#  🚀 ওয়ান-সেল ফুল-মিশন ল্যাব — সব কিছু এই এক সেলে:
#    [১] ডিপেন্ডেন্সি ইনস্টল          [২] রিপো সেটআপ (clone)
#    [৩] বাস্তব-স্পেক অডিট (real data) [৪] ফিজিক্স সেলফ-চেক
#    [৫] PPO-GPU ট্রেনিং (কিউরিকুলাম + রিজিউম)
#    [৬] ইভাল টেবিল                  [৭] GIF মিশন-রিপ্লে + সেভ
#  ⚠️ কপি-পেস্ট হ্যাজার্ড: GitHub "view"-এর কোড কপি করলে নিউলাইন হারিয়ে
#    SyntaxError হয়। র-লিঙ্ক (raw) থেকে কপি করো বা ফাইল ডাউনলোড করো।
#  র-লিঙ্ক: https://raw.githubusercontent.com/MD-Mushfiqur123/rocket-lander/vibe/one-cell-lab-03248f/one_cell_full_mission_lab.py
# ═══════════════════════════════════════════════════════════════════════════
import os, sys, subprocess, math, io, glob, json, base64, time
import numpy as np

# ── [০] কনফিগ নব — এখানে হাত দিয়ে বদলাও (Colab-এ এই চারটাই মূল খেলা) ──────
PRESET      = os.environ.get("LAB_PRESET", "earth_falcon9")  # earth_falcon9 | mars_edl
PHASE       = os.environ.get("LAB_PHASE", "landing")          # landing|reentry|boostback|ascent
TRAIN_STEPS = int(os.environ.get("LAB_TRAIN_STEPS", 3_000_000))
N_ENVS      = int(os.environ.get("LAB_N_ENVS", 4))            # 2|4|8 রাখো
BATCH_SIZE  = int(os.environ.get("LAB_BATCH_SIZE", 16384))    # GPU-তে কাজের ব্যাচ
N_STEPS     = int(os.environ.get("LAB_N_STEPS", 32768))       # রোলআউট বাফার (= N_STEPS×N_ENVS স্যাম্পল/ইটারেশন)
NET_SIZE    = int(os.environ.get("LAB_NET_SIZE", 512))        # MLP [512, 512] — GPU-ওয়ার্দি
EVAL_EPS    = int(os.environ.get("LAB_EVAL_EPS", 5))
N_REPLAY    = int(os.environ.get("LAB_REPLAY_EPS", 3))
RESUME      = os.environ.get("LAB_RESUME", "1") == "1"        # পুরনো মডেল থাকলে ওখান থেকেই চলবে
EXTREME     = os.environ.get("LAB_EXTREME", "0") == "1"       # কিউরিকুলাম শেষে 1→2: এক্সট্রিম বিগ-ডেটাসেট মোদ
# ────────────────────────────────────────────────────────────────────────────

MODEL_ZIP = f"{PRESET}_{PHASE}_ppo"
DONE_FILE = MODEL_ZIP + "_steps.json"
REPO_URL  = "https://github.com/MD-Mushfiqur123/rocket-lander.git"


def sh(cmd, quiet=True):
    return subprocess.run(cmd, capture_output=quiet, text=True).returncode


# ── [১] ডিপেন্ডেন্সি: যা নেই শুধু সেটাই ইনস্টল + পুরনো gym বাদ ─────────────
def stage1_deps():
    import importlib.util
    need = {"numpy": "numpy", "matplotlib": "matplotlib", "PIL": "pillow",
            "gymnasium": "gymnasium", "stable_baselines3": "stable-baselines3"}
    missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"📦 ইনস্টল হচ্ছে: {', '.join(missing)} …")
        sh([sys.executable, "-m", "pip", "install", "-q", *missing])
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "gym"])  # SB3-এর পুরনো gym সতর্কবার্তা বন্ধ
    print("✅ ডিপেন্ডেন্সি রেডি")


# ── [২] রিপো সেটআপ: env.py না থাকলে clone, sys.path বুটস্ট্র্যাপ ────────────
def stage2_repo():
    if not os.path.exists(os.path.join(os.getcwd(), "full_mission", "env.py")):
        base = "/content" if os.path.isdir("/content") else os.getcwd()
        repo_dir = os.path.join(base, "rocket-lander")
        if not os.path.exists(os.path.join(repo_dir, "full_mission", "env.py")):
            print("📥 রিপো clone হচ্ছে …")
            sh(["git", "clone", "--depth", "1", REPO_URL, repo_dir])
        if not os.path.exists(os.path.join(repo_dir, "full_mission", "env.py")):
            raise RuntimeError("clone ব্যর্থ — ইন্টারনেট চেক করো বা ম্যানুয়ালি রিপো নামাও")
        os.chdir(repo_dir)
    sys.path.insert(0, os.getcwd())
    from full_mission.env import FullMissionEnv, PRESETS
    if PRESET not in PRESETS:
        raise ValueError(f"PRESET='{PRESET}' ভুল — বৈধ: {list(PRESETS)}")
    from full_mission.scripted_test import (LandingController, MarsEdlController,
                                            GravityTurnController, TvcAttitude)
    print(f"✅ রিপো রেডি ({os.getcwd().rsplit('/')[-1]}), preset={PRESET}, phase={PHASE}")
    return FullMissionEnv, LandingController, MarsEdlController, GravityTurnController, TvcAttitude


# ── [৩] রিয়েল-স্পেক অডিট: সিমের ধ্রুবক vs প্রকাশিত বাস্তব ডেটা ─────────────
def stage3_realdata(PRESETS):
    spec = {
        "earth_falcon9": [
            ("গ্র্যাভিটি (m/s²)",            "gravity",       9.807,     "==="),
            ("ড্রাই মাস (kg)",               "dry_mass",      25_600.0,  "~"),      # F9 S1 dry
            ("প্রোপেলেন্ট (kg)",             "fuel_mass",     410_000.0, "গেম-সংকুচিত"),
            ("থ্রাস্ট (MN, SL)",             "max_thrust",    7.6e6,     "~"),      # 9×Merlin 1D
            ("ISP (s)",                     "isp",           300.0,     "~"),      # SL 282 / vac 311
            ("বডি এরিয়া (m²)",             "body_area",     10.52,     "~"),      # d=3.66 m
            ("TVC জিম্বাল (°)",             "gimbal_deg",    5.0,       "~"),
            ("০-সমুদ্র ঘনত্ব (kg/m³)",      "rho0",          1.225,     "~"),
            ("স্কেল-হাইট (m)",              "scale_height",  8_500.0,   "~"),      # US76 নিম্নস্তর
            ("প্যাড অর্ধ-প্রস্থ (m)",        "pad_half_width", 30.0,     "LZ-1"),
        ],
        "mars_edl": [
            ("গ্র্যাভিটি (m/s²)",            "gravity",       3.71,      "==="),
            ("০-মান ঘনত্ব (kg/m³)",         "rho0",          0.020,     "~"),      # MSL-পরিমিত
            ("স্কেল-হাইট (m)",              "scale_height",  11_100.0,  "~"),      # মিড-অ্যাটমোসফিয়ার
            ("ছাতার এরিয়া (m²)",           "chute_area",    179.0,     "সিম-বড়"), # MSL D=15.1 m
            ("dry_mass (kg)",               "dry_mass",      2_400.0,   "~"),
            ("max thrust (kN)",             "max_thrust",    500.0,     "~"),
            ("TVC জিম্বাল (°)",             "gimbal_deg",    8.0,       "~"),
        ],
    }
    cfg = dict(PRESETS[PRESET])
    print("🌍 রিয়েল-স্পেক অডিট (প্রকাশিত তথ্যের সাথে যাচাই):")
    for label, key, real, note in spec[PRESET]:
        sim = cfg.get(key)
        if sim is None or not real:
            continue
        ratio = sim / real
        print(f"   {label:<28} সিম={sim:>10,.1f} | বাস্তব={real:>10,.1f}"
              f" | অনুপাত={ratio:.2f} [{note}]")
    print("   (অনুপাত 1.0 = বাস্তবের সাথে হুবহু; 'গেম-সংকুচিত' মানে ছোট-এপিসোডের জন্য স্কেল করা)")


# ── [৪] ফিজিক্স সেলফ-চেক: স্ক্রিপ্টেড কন্ট্রোলার ২টা এপিসোড ─────────────
def stage3_selfcheck(FullMissionEnv, ctrl_cls):
    print("🔬 ফিজিক্স সেলফ-চেক …")
    oks, speeds = 0, []
    for seed in (0, 1):
        env = FullMissionEnv(PRESET, start_phase=PHASE, domain_randomization=False, seed=seed)
        ctrl = ctrl_cls()  # স্টেটফুল — প্রতি এপিসোডে নতুন ইনস্ট্যান্স বাধ্যতামূলক
        obs, _ = env.reset()
        done, info = False, {}
        while not done:
            obs, _, term, trunc, info = env.step(np.asarray(ctrl(obs, env), dtype=np.float32))
            done = term or trunc
        env.close()
        speeds.append(info["speed"])
        if info["outcome"] == "landed":
            oks += 1
    tag = "Mars EDL" if PRESET == "mars_edl" else "TVC ল্যান্ডিং"
    print(f"   {tag} স্ক্রিপ্টেড: {oks}/2 landed, touchdown speeds={['%.1f' % s for s in speeds]} m/s")
    if oks == 0:
        raise RuntimeError(f"সেলফ-চেক ফেল — {tag} ফিজিক্সে ল্যান্ডই করা যাচ্ছে না, ট্রেনিং বৃথা যাবে")
    print(f"✅ ফিজিক্স ল্যান্ডেবল ({oks}/2) — এবার AI শেখার পালা")


# ── [৪] PPO ট্রেনিং: কিউরিকুলাম র‍্যাম্প + ১০০k-চাঙ্ক চেকপয়েন্ট + রিজিউম ───
def stage4_train(FullMissionEnv):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)} — ব্যাচ {BATCH_SIZE:,}, রোলআউট {N_STEPS:,}/ইটারেশন")
        import warnings
        warnings.filterwarnings("ignore", message=".*run PPO on the GPU.*")
    else:
        print("🖥️ GPU নেই — CPU-তে চলছে")

    env = None
    if N_ENVS > 1:
        try:
            env = SubprocVecEnv([
                lambda: Monitor(FullMissionEnv(PRESET, start_phase=PHASE))
                for _ in range(N_ENVS)
            ])
        except Exception as e:
            print(f"⚠️ SubprocVecEnv ব্যর্থ ({type(e).__name__}) — DummyVecEnv-এ চলছে, ধীর কিন্তু নিশ্চিত")
    if env is None:
        env = DummyVecEnv([
            lambda: Monitor(FullMissionEnv(PRESET, start_phase=PHASE))
            for _ in range(N_ENVS)
        ])
    env.env_method("set_randomization_strength", 0.0)

    done_steps, model = 0, None
    if RESUME and os.path.exists(MODEL_ZIP + ".zip"):
        model = PPO.load(MODEL_ZIP)
        if os.path.exists(DONE_FILE):
            done_steps = int(json.load(open(DONE_FILE)))
        print(f"♻️ রিজিউ: {MODEL_ZIP}.zip (আগের অবস্থা {done_steps:,} স্টেপ)")
    if model is None:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            gamma=0.999, gae_lambda=0.95, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[NET_SIZE, NET_SIZE]),
            verbose=1, device=DEVICE,
        )

    ramp_frac, chunk = 0.6, max(100_000, N_STEPS * N_ENVS)
    ramp = ramp_frac * TRAIN_STEPS
    while done_steps < TRAIN_STEPS:
        n = min(chunk, TRAIN_STEPS - done_steps)
        model.learn(total_timesteps=n, reset_num_timesteps=False)
        done_steps += n
        frac = min(1.0, done_steps / max(1.0, ramp))
        strength = min(2.0, frac + max(0.0, (done_steps - ramp) / max(1.0, 0.4 * TRAIN_STEPS))) \
            if EXTREME else frac
        env.env_method("set_randomization_strength", strength)
        model.save(MODEL_ZIP)
        json.dump(done_steps, open(DONE_FILE, "w"))
        quick = run_eval(FullMissionEnv, model, 2 if done_steps < TRAIN_STEPS else 0, seed0=12345)
        label = (f"extreme s={strength:.1f}" if strength > 1.0
                 else ("full randomization" if frac >= 1.0 else f"curriculum {frac:.0%}"))
        if quick:
            landed = sum(1 for r in quick if r[0] == "landed")
            print(f"[{done_steps:,}/{TRAIN_STEPS:,} | {label}] চেকপয়েন্ট-ইভাল landed={landed}/{len(quick)}")
    env.close()
    print(f"✅ ট্রেনিং শেষ → {MODEL_ZIP}.zip")
    return model


def run_eval(mk_env, model, n_eps, seed0=12345):
    rows = []
    for ep in range(n_eps):
        env = mk_env(PRESET, start_phase=PHASE, seed=seed0 + ep)
        obs, _ = env.reset()
        ep_r, done, info = 0.0, False, {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_r += float(r)
            done = term or trunc
        env.close()
        rows.append((info["outcome"] or "timeout", ep_r, info["speed"], info["tilt"]))
    return rows


# ── [৫] ফাইনাল ইভাল টেবিল ──────────────────────────────────────────────────
def stage5_eval(FullMissionEnv, model):
    rows = run_eval(FullMissionEnv, model, EVAL_EPS, seed0=90001)
    landed = sum(1 for r in rows if r[0] == "landed")
    md = (f"### 📋 ফাইনাল ইভাল ({EVAL_EPS} নতুন সিড, deterministic)\n"
          "| Ep | Outcome | Reward | Touchdown speed | Tilt |\n|---|---|---|---|---|\n"
          + "\n".join(f"| {i+1} | {r[0]} | {r[1]:.1f} | {r[2]:.2f} | {r[3]:.1f}° |"
                       for i, r in enumerate(rows))
          + f"\n\n**Landings: {landed}/{len(rows)}**")
    print(md)
    return rows, landed


# ── [৬] GIF মিশন-রিপ্লে ────────────────────────────────────────────────────
def render(env, trail, thr, width=420, height=300):
    import matplotlib.figure, matplotlib.patches as mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    st = env.state
    pad_x, x, z = env._p["pad_x"], st["pos"][0], max(st["pos"][1], 0.0)
    half = max(abs(x - pad_x) / 2 + 250.0, z * 1.25, 600.0)
    fig = matplotlib.figure.Figure(figsize=(width / 80, height / 80), dpi=80)
    fig.patch.set_facecolor("#0b0e1a")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#0b0e1a")
    ax.plot([-half, half], [0, 0], color="#2a2f45", lw=2)
    ax.add_patch(mpatches.Rectangle((pad_x - env.cfg["pad_half_width"], 0),
                 2 * env.cfg["pad_half_width"], max(0.025 * half, 8.0),
                 facecolor="#e6c700", zorder=2))
    if trail:
        tx, tz = zip(*trail)
        ax.plot(tx, tz, color="#4da6ff", lw=0.9, alpha=0.55)
    L = max(0.07 * half, 25.0); W = L * 0.22
    c, s = math.cos(st["theta"]), math.sin(st["theta"])
    w = lambda px, pz: (x + px * c - pz * s, z + px * s + pz * c)
    ax.add_patch(mpatches.Polygon([w(-W, -L/2), w(W, -L/2), w(W, L/2), w(-W, L/2)],
                 closed=True, facecolor="#cc3333", edgecolor="#ff8080", zorder=5))
    if thr > 0.05:
        pl = L * (0.5 + 0.9 * thr)
        ax.add_patch(mpatches.Polygon([w(-W*0.7, -L/2), w(W*0.7, -L/2), w(0, -L/2 - pl)],
                     closed=True, facecolor="#ffb347", alpha=0.9, zorder=4))
    if env.chute_deployed:
        cr = L * 1.6
        ax.add_patch(mpatches.Arc(w(0, L), 2*cr, 2*cr, theta1=20, theta2=160,
                     color="#dddddd", lw=2.5, zorder=3))
    hud = (f"alt {z:,.0f} m   speed {math.hypot(st['vel'][0], st['vel'][1]):.1f}"
           f" m/s   fuel {env._fuel_frac:.0%}")
    ax.text(0.02, 0.02, hud, transform=ax.transAxes, color="#9fb3c8", fontsize=6)
    ax.set_xlim(-half, half); ax.set_ylim(-0.06 * half, 2 * half)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.tight_layout(pad=0.2); fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(np.uint8)


def stage6_replay(FullMissionEnv, model):
    from PIL import Image
    hists, stats = [], []
    for ep in range(N_REPLAY):
        env = FullMissionEnv(PRESET, start_phase=PHASE, seed=777_000 + ep)
        obs, _ = env.reset()
        ep_r, done, hist, info = 0.0, False, [], {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_r += float(r)
            done = term or trunc
            st = env.state
            hist.append((float(st["pos"][0]), float(st["pos"][1]), st["theta"],
                         float(np.ravel(a)[1]), env.chute_deployed))
        env.close()
        hists.append(hist)
        stats.append((info["outcome"] or "timeout", ep_r, info["speed"], info["tilt"]))

    frames = []
    for hist in hists:
        k = max(1, len(hist) // 80)
        skel = FullMissionEnv(PRESET, start_phase=PHASE, seed=0)
        skel.reset(seed=0)
        trail = []
        for i in range(0, len(hist), k):
            px, pz, th, thr, chute = hist[i]
            trail.append((px, pz))
            skel.state = {"pos": np.array([px, pz]), "vel": np.zeros(2),
                          "theta": th, "omega": 0.0, "fuel": skel._p["fuel0"]}
            skel.chute_deployed = chute
            frames.append(render(skel, trail, thr))
    gif_path = f"{MODEL_ZIP}_replay.gif"
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif_path, format="GIF", save_all=True, append_images=pil[1:],
                duration=90, loop=0)
    rows = "\n".join(f"| {i+1} | {r[0]} | {r[1]:.1f} | {r[2]:.2f} | {r[3]:.1f}° |"
                     for i, r in enumerate(stats))
    try:
        from IPython.display import display, HTML, Markdown
        buf = io.BytesIO()
        pil[0].save(buf, format="GIF", save_all=True, append_images=pil[1:],
                    duration=90, loop=0)
        display(HTML('<img src="data:image/gif;base64,' +
                     base64.b64encode(buf.getvalue()).decode() + '" width="420">'))
        display(Markdown(f"### 🎬 রিপ্লে ({N_REPLAY} এপিসোড)\n"
                         "| Ep | Outcome | Reward | Touchdown speed | Tilt |\n|---|---|---|---|---|\n" + rows))
    except ImportError:
        print(rows)
    print(f"🎬 GIF সেভ হয়েছে → {os.path.join(os.getcwd(), gif_path)}")


# ── [৭] (ঐচ্ছিক) Drive সেভ — Colab-এ Drive mount করা থাকলে অটো-ব্যাকআপ ─────
def stage7_drive():
    for f in sorted(glob.glob(MODEL_ZIP + "*")):
        drive = "/content/drive/MyDrive/rocket_lander_models"
        if os.path.isdir(drive) and not os.path.exists(os.path.join(drive, f)):
            import shutil
            shutil.copy2(f, os.path.join(drive, f))
            print(f"☁️ Drive-এ সেভ → {drive}/{f}")


# ── মেইন পাইপলাইন: এক সেলে পুরো ল্যাব ─────────────────────────────────────
def main_pipeline():
    stage1_deps()
    from full_mission.env import PRESETS as _PRESETS
    FME, LCtrl, MCtrl, _G, _T = stage2_repo()
    stage3_realdata(_PRESETS)
    t0 = time.time()
    stage3_selfcheck(FME, MCtrl if PRESET == "mars_edl" else LCtrl)
    print(f"   (সেলফ-চেক {time.time() - t0:.0f}s)")
    model = stage4_train(FME)
    stage5_eval(FME, model)
    stage6_replay(FME, model)
    stage7_drive()
    print("\n🎉 ফুল ল্যাব সম্পন্ন — মডেল: {}.zip, GIF: {}_replay.gif".format(MODEL_ZIP, MODEL_ZIP))


if __name__ == "__main__":
    main_pipeline()
