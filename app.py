"""
app.py  —  LipMatch Flask Backend
Full pipeline:
  Image → Face detect → Skin mask → K-Means → LAB tone → Undertone
       → MobileNetV2 skin type (with graceful fallback)
       → Random Forest shade group prediction (Ramjhun's classifier)
       → LAB→RGB colour boxes for frontend rendering
       → Ranked lipstick recommendations

Run:
    pip install flask flask-cors opencv-python numpy scikit-learn joblib
    python TrainShadeModel.py          ← run ONCE to train the RF model
    python app.py
"""

import os, sys, cv2, base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

app = Flask(__name__)
CORS(app)

# ── Lazy-loaded models ─────────────────────────────────────────────────────────
_cnn_model = None  # MobileNetV2  (SkinTypeModel.h5)
_rf_model = None  # Random Forest (rf_shade_model.pkl)
_rf_encoder = None  # LabelEncoder  (rf_label_encoder.pkl)


def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        path = os.path.join(ROOT, "model", "SkinTypeModel.h5")
        if os.path.exists(path):
            try:
                from tensorflow.keras.models import load_model

                _cnn_model = load_model(path)
                print("[INFO] SkinTypeModel.h5 loaded ✓")
            except Exception as e:
                print(f"[WARN] Could not load CNN model: {e}")
        else:
            print(f"[WARN] SkinTypeModel.h5 not found at {path}")
            print("       Run TrainSkinType.py first to generate it.")
    return _cnn_model


def get_rf_model():
    global _rf_model, _rf_encoder
    if _rf_model is None:
        import joblib

        mp = os.path.join(ROOT, "model", "rf_shade_model.pkl")
        ep = os.path.join(ROOT, "model", "rf_label_encoder.pkl")
        if os.path.exists(mp) and os.path.exists(ep):
            _rf_model = joblib.load(mp)
            _rf_encoder = joblib.load(ep)
            print("[INFO] RF shade model loaded ✓")
        else:
            print("[WARN] rf_shade_model.pkl not found.")
            print("       Run TrainShadeModel.py first.")
    return _rf_model, _rf_encoder


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR SCIENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# Sub-group LAB centres (from ShadeRangeEngine.py — kept in sync)
SUB_GROUP_CENTRES = {
    "Deep Coral": {"L": 40, "a": 32, "b": 28},
    "True Coral": {"L": 52, "a": 30, "b": 26},
    "Warm Nude": {"L": 62, "a": 18, "b": 22},
    "Peachy Nude": {"L": 70, "a": 14, "b": 20},
    "Deep Berry": {"L": 32, "a": 38, "b": -5},
    "True Berry": {"L": 42, "a": 35, "b": -3},
    "Cool Mauve": {"L": 55, "a": 22, "b": 2},
    "Soft Pink": {"L": 68, "a": 25, "b": 5},
    "True Red": {"L": 38, "a": 48, "b": 20},
    "Blue Red": {"L": 45, "a": 45, "b": 10},
    "Dusty Rose": {"L": 58, "a": 20, "b": 8},
    "Taupe Nude": {"L": 65, "a": 12, "b": 10},
}

# Which sub-groups belong to each primary group
PRIMARY_TO_SUBS = {
    "Coral": ["Deep Coral", "True Coral", "Warm Nude", "Peachy Nude"],
    "Nude": ["Warm Nude", "Peachy Nude", "Taupe Nude"],
    "Red": ["True Red", "Blue Red"],
    "Berry": ["Deep Berry", "True Berry"],
    "Mauve": ["Cool Mauve", "Dusty Rose"],
    "Pink": ["Soft Pink"],
}

# Undertone → preferred primary groups (colour theory)
UNDERTONE_PRIMARY = {
    "warm": ["Coral", "Nude", "Red"],
    "cool": ["Berry", "Mauve", "Pink"],
    "neutral": ["Red", "Nude", "Mauve"],
}

# Finish affinity by skin type
SKIN_TYPE_FINISH = {
    "dry": ["Satin", "Gloss"],
    "normal": ["Matte", "Satin", "Gloss", "Velvet"],
    "oily": ["Matte", "Velvet"],
}

CONTRAST_MAP = {"low": 0, "medium": 1, "high": 2}


def lab_to_rgb(L_ocv, a_ocv, b_ocv):
    """
    Convert OpenCV LAB (L∈[0,255], a/b∈[0,255] centred at 128)
    to sRGB uint8 (R,G,B).
    Returns (R, G, B) as ints 0-255.
    """
    lab_arr = np.array([[[L_ocv, a_ocv, b_ocv]]], dtype=np.uint8)
    rgb_arr = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)
    r, g, b = int(rgb_arr[0, 0, 0]), int(rgb_arr[0, 0, 1]), int(rgb_arr[0, 0, 2])
    return r, g, b


def sub_group_lab_to_rgb(sub_group):
    """
    Convert a sub-group's centre LAB to sRGB for rendering as a colour box.
    The SUB_GROUP_CENTRES use standard LAB; we need to convert to OpenCV encoding.
    Standard → OpenCV: L_ocv = L * 255/100, a_ocv = a + 128, b_ocv = b + 128
    """
    centre = SUB_GROUP_CENTRES.get(sub_group, {"L": 50, "a": 20, "b": 10})
    L_ocv = int(centre["L"] * 255 / 100)
    a_ocv = int(centre["a"] + 128)
    b_ocv = int(centre["b"] + 128)
    # Clamp to uint8
    L_ocv = max(0, min(255, L_ocv))
    a_ocv = max(0, min(255, a_ocv))
    b_ocv = max(0, min(255, b_ocv))
    r, g, b = lab_to_rgb(L_ocv, a_ocv, b_ocv)
    return {"R": r, "G": g, "B": b, "hex": f"#{r:02x}{g:02x}{b:02x}"}


def delta_e(lab1, lab2):
    """CIE76 Euclidean distance in standard LAB space."""
    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2))))


# ══════════════════════════════════════════════════════════════════════════════
# LIPSTICK CATALOGUE  (each shade has LAB + hex + finish)
# ══════════════════════════════════════════════════════════════════════════════
LIPSTICK_CATALOGUE = [
    # Coral
    {
        "name": "Sunset Coral",
        "primary": "Coral",
        "sub": "True Coral",
        "L": 52,
        "a": 30,
        "b": 26,
        "hex": "#E07A5F",
        "finish": "Satin",
    },
    {
        "name": "Deep Terracotta",
        "primary": "Coral",
        "sub": "Deep Coral",
        "L": 40,
        "a": 32,
        "b": 28,
        "hex": "#C0522A",
        "finish": "Matte",
    },
    {
        "name": "Peachy Dream",
        "primary": "Coral",
        "sub": "Peachy Nude",
        "L": 70,
        "a": 14,
        "b": 20,
        "hex": "#F4A882",
        "finish": "Gloss",
    },
    {
        "name": "Warm Glow",
        "primary": "Coral",
        "sub": "Warm Nude",
        "L": 62,
        "a": 18,
        "b": 22,
        "hex": "#D4896A",
        "finish": "Satin",
    },
    # Nude
    {
        "name": "Bare Minimum",
        "primary": "Nude",
        "sub": "Taupe Nude",
        "L": 65,
        "a": 12,
        "b": 10,
        "hex": "#C9A98A",
        "finish": "Matte",
    },
    {
        "name": "Your Lip But Better",
        "primary": "Nude",
        "sub": "Warm Nude",
        "L": 60,
        "a": 16,
        "b": 18,
        "hex": "#BC8B6E",
        "finish": "Gloss",
    },
    {
        "name": "Cashmere Touch",
        "primary": "Nude",
        "sub": "Peachy Nude",
        "L": 72,
        "a": 13,
        "b": 19,
        "hex": "#E8C4A8",
        "finish": "Satin",
    },
    # Red
    {
        "name": "Classic Red",
        "primary": "Red",
        "sub": "True Red",
        "L": 38,
        "a": 48,
        "b": 20,
        "hex": "#9B2335",
        "finish": "Matte",
    },
    {
        "name": "Ruby Noir",
        "primary": "Red",
        "sub": "Blue Red",
        "L": 35,
        "a": 45,
        "b": 8,
        "hex": "#7A1A2E",
        "finish": "Velvet",
    },
    {
        "name": "Fire Engine",
        "primary": "Red",
        "sub": "True Red",
        "L": 42,
        "a": 50,
        "b": 25,
        "hex": "#C0392B",
        "finish": "Gloss",
    },
    # Berry
    {
        "name": "Wild Berry",
        "primary": "Berry",
        "sub": "True Berry",
        "L": 42,
        "a": 35,
        "b": -3,
        "hex": "#6B3057",
        "finish": "Matte",
    },
    {
        "name": "Midnight Plum",
        "primary": "Berry",
        "sub": "Deep Berry",
        "L": 32,
        "a": 38,
        "b": -5,
        "hex": "#4A1942",
        "finish": "Velvet",
    },
    {
        "name": "Boysenberry",
        "primary": "Berry",
        "sub": "True Berry",
        "L": 40,
        "a": 36,
        "b": -2,
        "hex": "#5C2851",
        "finish": "Satin",
    },
    # Mauve
    {
        "name": "Dusty Rose",
        "primary": "Mauve",
        "sub": "Dusty Rose",
        "L": 58,
        "a": 20,
        "b": 8,
        "hex": "#A78295",
        "finish": "Matte",
    },
    {
        "name": "Cool Mauve",
        "primary": "Mauve",
        "sub": "Cool Mauve",
        "L": 55,
        "a": 22,
        "b": 2,
        "hex": "#957088",
        "finish": "Satin",
    },
    {
        "name": "Muted Violet",
        "primary": "Mauve",
        "sub": "Cool Mauve",
        "L": 52,
        "a": 24,
        "b": 0,
        "hex": "#8B6880",
        "finish": "Velvet",
    },
    # Pink
    {
        "name": "Bubble Gum",
        "primary": "Pink",
        "sub": "Soft Pink",
        "L": 68,
        "a": 25,
        "b": 5,
        "hex": "#E8A0BF",
        "finish": "Gloss",
    },
    {
        "name": "Petal Soft",
        "primary": "Pink",
        "sub": "Soft Pink",
        "L": 70,
        "a": 23,
        "b": 4,
        "hex": "#EDB8CF",
        "finish": "Satin",
    },
    {
        "name": "Cotton Candy",
        "primary": "Pink",
        "sub": "Soft Pink",
        "L": 72,
        "a": 21,
        "b": 6,
        "hex": "#F2C4D8",
        "finish": "Gloss",
    },
]


def recommend_lipsticks(
    skin_lab_std, undertone, skin_type, rf_primary_group=None, top_n=6
):
    """
    Score every shade in the catalogue and return the best top_n.

    Scoring formula:
        score = ΔE(skin_lab, shade_lab)          ← colour closeness
              + undertone_bonus  (−5 if family matches undertone)
              + rf_bonus         (−8 if family matches RF prediction)
              + finish_bonus     (−3 if finish suits skin type, else +3)

    rf_primary_group: the shade family predicted by the Random Forest.
    If available, shades in that family get a strong −8 bonus.
    """
    preferred_groups = UNDERTONE_PRIMARY.get(undertone.lower(), [])
    preferred_finishes = SKIN_TYPE_FINISH.get(skin_type.lower(), [])

    scored = []
    for shade in LIPSTICK_CATALOGUE:
        de = delta_e(skin_lab_std, (shade["L"], shade["a"], shade["b"]))
        bonus = 0
        if shade["primary"] in preferred_groups:
            bonus -= 5
        if rf_primary_group and shade["primary"] == rf_primary_group:
            bonus -= 8
        if shade["finish"] in preferred_finishes:
            bonus -= 3
        else:
            bonus += 3
        scored.append({**shade, "score": de + bonus, "delta_e": round(de, 2)})

    scored.sort(key=lambda x: x["score"])
    return scored[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PIPELINE  (cv2 + numpy + sklearn only — no matplotlib)
# ══════════════════════════════════════════════════════════════════════════════


def decode_image(b64):
    img_bytes = base64.b64decode(b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def detect_face(image_bgr):
    h, w = image_bgr.shape[:2]
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return (0, 0, w, h)
    x, y, fw, fh = faces[0]
    return (x, y, x + fw, y + fh)


def skin_mask(image_bgr, min_px=300):
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    _, mask = cv2.threshold(
        cv2.GaussianBlur(mask, (5, 5), 0), 127, 255, cv2.THRESH_BINARY
    )
    n = int(np.count_nonzero(mask))
    if n < min_px:
        raise ValueError(
            f"Only {n} skin pixels detected (need {min_px}+). "
            "Ensure your face is clearly visible and well-lit."
        )
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return rgb[mask > 0], n


def kmeans_tone(skin_rgb, k=3):
    from sklearn.cluster import KMeans

    reshaped = skin_rgb.reshape(1, -1, 3).astype(np.uint8)
    lab_pixels = (
        cv2.cvtColor(reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    )
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(lab_pixels)
    dom_idx = int(np.argmax(np.bincount(labels, minlength=k)))
    dom_lab = cv2.cvtColor(
        skin_rgb[labels == dom_idx].reshape(1, -1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    # Median per channel (OpenCV encoding)
    mL = float(np.median(dom_lab[:, 0]))
    ma = float(np.median(dom_lab[:, 1]))
    mb = float(np.median(dom_lab[:, 2]))
    med_rgb = cv2.cvtColor(
        np.array([[[mL, ma, mb]]], dtype=np.uint8), cv2.COLOR_LAB2RGB
    )[0, 0]
    return (
        {"L": mL, "a": ma, "b": mb},
        {"R": int(med_rgb[0]), "G": int(med_rgb[1]), "B": int(med_rgb[2])},
    )


def get_undertone(lab_ocv):
    """From OpenCV LAB (a/b centred at 128)."""
    a_s = lab_ocv["a"] - 128
    b_s = lab_ocv["b"] - 128
    if b_s > 5 and a_s < 10:
        return "warm", min(100.0, b_s / 20 * 100)
    elif b_s < -2:
        return "cool", min(100.0, abs(b_s) / 10 * 100)
    elif abs(b_s) <= 5 and abs(a_s) <= 10:
        return "neutral", 70.0
    else:
        return ("warm" if b_s > 0 else "neutral"), 60.0


def classify_skin_type(image_bgr):
    CLASS_NAMES = ["dry", "normal", "oily"]
    model = get_cnn_model()
    if model is None:
        # ── Rule-based fallback using skin masking ──────────────────────────
        # Without the CNN, we use a simple heuristic based on skin pixel
        # brightness variance (oily skin tends to have brighter, more uniform tone)
        try:
            px, _ = skin_mask(image_bgr, min_px=200)
            gray_var = float(
                np.var(
                    cv2.cvtColor(
                        px.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2GRAY
                    ).flatten()
                )
            )
            if gray_var > 1800:
                st, probs = "oily", [0.10, 0.25, 0.65]
            elif gray_var > 900:
                st, probs = "normal", [0.15, 0.70, 0.15]
            else:
                st, probs = "dry", [0.70, 0.20, 0.10]
        except Exception:
            st, probs = "normal", [0.15, 0.65, 0.20]
        return {
            "skin_type": st,
            "confidence": max(probs),
            "probabilities": probs,
            "mock": True,
            "note": "CNN model not loaded — using brightness heuristic",
        }
    try:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = np.expand_dims(cv2.resize(rgb, (224, 224)) / 255.0, 0)
        probs = model.predict(tensor, verbose=0)[0].tolist()
        idx = int(np.argmax(probs))
        return {
            "skin_type": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "probabilities": probs,
            "mock": False,
            "note": "CNN prediction",
        }
    except Exception as e:
        print(f"[WARN] CNN inference failed: {e}")
        return {
            "skin_type": "normal",
            "confidence": 0.60,
            "probabilities": [0.15, 0.60, 0.25],
            "mock": True,
            "note": f"CNN error — fallback: {e}",
        }


def predict_shade_group_rf(
    lab_ocv, contrast_level="medium", normal_pct=50, oily_pct=25, dry_pct=25
):
    """
    Use the trained Random Forest to predict primary_group from skin LAB values.
    Returns (primary_group_str, probabilities_dict) or (None, {}) if model missing.
    """
    rf, le = get_rf_model()
    if rf is None:
        return None, {}

    # Convert OpenCV LAB to standard LAB for features
    # (training used skin_L/skin_a/skin_b from SkinToneGenerator which uses standard scale)
    # Standard: L=0-100, a=-128..127, b=-128..127
    skin_L_std = lab_ocv["L"] * 100 / 255
    skin_a_std = lab_ocv["a"] - 128
    skin_b_std = lab_ocv["b"] - 128

    contrast_enc = CONTRAST_MAP.get(contrast_level.lower(), 1)

    # Build feature vector — match what TrainShadeModel.py uses
    X = np.array(
        [
            [
                skin_L_std,
                skin_a_std,
                skin_b_std,
                contrast_enc,
                normal_pct,
                oily_pct,
                dry_pct,
            ]
        ],
        dtype=np.float32,
    )

    # Drop last 3 features if model was trained without texture cols
    n_features = rf.n_features_in_
    X = X[:, :n_features]

    pred_idx = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    proba_dict = {cls: round(float(p), 3) for cls, p in zip(le.classes_, proba)}

    print(
        f"[RF] Predicted primary group: {pred_label} " f"(confidence: {max(proba):.2f})"
    )
    return pred_label, proba_dict


def build_colour_boxes(primary_group, undertone, skin_lab_std):
    """
    Build the coloured boxes to render on the frontend.
    Each box has:
        sub_group   — name
        rgb         — { R, G, B }   ← render as background-color: rgb(R,G,B)
        hex         — "#rrggbb"
        delta_e     — distance from user's skin tone
        is_primary  — True if this sub belongs to RF-predicted group
    """
    boxes = []
    for sub, centre in SUB_GROUP_CENTRES.items():
        colour = sub_group_lab_to_rgb(sub)
        de = delta_e(skin_lab_std, (centre["L"], centre["a"], centre["b"]))
        # Find which primary this sub belongs to
        sub_primary = next(
            (p for p, subs in PRIMARY_TO_SUBS.items() if sub in subs), "Other"
        )
        boxes.append(
            {
                "sub_group": sub,
                "primary": sub_primary,
                "rgb": colour,  # ← frontend renders this as colored box
                "hex": colour["hex"],
                "delta_e": round(de, 1),
                "is_primary": sub_primary == primary_group,
            }
        )

    # Sort: RF-predicted group first, then by closeness
    boxes.sort(key=lambda x: (0 if x["is_primary"] else 1, x["delta_e"]))
    return boxes


# ══════════════════════════════════════════════════════════════════════════════
# API ROUTES
# ══════════════════════════════════════════════════════════════════════════════


@app.route("/api/health", methods=["GET"])
def health():
    cnn_ready = os.path.exists(os.path.join(ROOT, "model", "SkinTypeModel.h5"))
    rf_ready = os.path.exists(os.path.join(ROOT, "model", "rf_shade_model.pkl"))
    return jsonify(
        {
            "status": "ok",
            "cnn_model": "loaded" if cnn_ready else "missing — run TrainSkinType.py",
            "rf_model": "loaded" if rf_ready else "missing — run TrainShadeModel.py",
        }
    )


@app.route("/api/analyse", methods=["POST"])
def analyse():
    """
    POST /api/analyse
    Body: { "image": "<base64>" }
    Returns full analysis + RF colour boxes + recommendations.
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]

        img = decode_image(b64)
        if img is None:
            return jsonify({"error": "Could not decode image"}), 400

        # 1. Face crop
        x1, y1, x2, y2 = detect_face(img)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            face = img

        # 2. Skin mask
        skin_px, px_count = skin_mask(face, min_px=300)

        # 3. K-Means → dominant LAB (OpenCV encoding)
        lab_ocv, rgb = kmeans_tone(skin_px, k=3)

        # 4. Undertone from LAB
        undertone, confidence = get_undertone(lab_ocv)

        # 5. Standard LAB for downstream (ΔE, RF features)
        skin_lab_std = (
            lab_ocv["L"] * 100 / 255,
            lab_ocv["a"] - 128,
            lab_ocv["b"] - 128,
        )

        # 6. Skin type (CNN or heuristic fallback)
        type_data = classify_skin_type(img)

        # 7. Derive contrast from skin L* (simple proxy)
        L_std = skin_lab_std[0]
        contrast_level = "high" if L_std < 45 else ("medium" if L_std < 65 else "low")

        # 8. Skin type texture probabilities for RF
        probs = type_data["probabilities"]
        dry_pct = round(probs[0] * 100, 1)
        normal_pct = round(probs[1] * 100, 1)
        oily_pct = round(probs[2] * 100, 1)

        # 9. Random Forest → shade group prediction
        rf_group, rf_proba = predict_shade_group_rf(
            lab_ocv, contrast_level, normal_pct, oily_pct, dry_pct
        )

        # 10. Build colour boxes for frontend rendering
        colour_boxes = build_colour_boxes(rf_group, undertone, skin_lab_std)

        # 11. Recommendations (enhanced with RF bonus)
        recs = recommend_lipsticks(
            skin_lab_std=skin_lab_std,
            undertone=undertone,
            skin_type=type_data["skin_type"],
            rf_primary_group=rf_group,
            top_n=6,
        )

        # ── Build response ──────────────────────────────────────────────────
        return jsonify(
            {
                "skin_tone": {
                    "lab_opencv": lab_ocv,
                    "lab_standard": {
                        "L": round(skin_lab_std[0], 1),
                        "a": round(skin_lab_std[1], 1),
                        "b": round(skin_lab_std[2], 1),
                    },
                    "rgb": rgb,
                    "hex": f"#{rgb['R']:02x}{rgb['G']:02x}{rgb['B']:02x}",
                    "undertone": undertone,
                    "confidence": round(confidence, 1),
                    "pixel_count": px_count,
                },
                "skin_type": {
                    "type": type_data["skin_type"],
                    "confidence": round(type_data["confidence"], 3),
                    "probabilities": {
                        "dry": round(probs[0], 3),
                        "normal": round(probs[1], 3),
                        "oily": round(probs[2], 3),
                    },
                    "mock": type_data.get("mock", False),
                    "note": type_data.get("note", ""),
                },
                "rf_prediction": {
                    "primary_group": rf_group,
                    "probabilities": rf_proba,
                    "model_loaded": rf_group is not None,
                    "colour_boxes": colour_boxes,  # ← array of {sub_group, rgb, hex, delta_e}
                },
                "recommendations": [
                    {
                        "name": r["name"],
                        "primary": r["primary"],
                        "sub": r["sub"],
                        "hex": r["hex"],
                        "finish": r["finish"],
                        "delta_e": r["delta_e"],
                        # Convert shade LAB → RGB so frontend can render coloured boxes
                        "shade_rgb": {
                            "R": int(int(r["L"]) * 255 // 100),  # approximate only
                            "hex": r["hex"],
                        },
                    }
                    for r in recs
                ],
            }
        )

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend-manual", methods=["POST"])
def recommend_manual():
    data = request.get_json(silent=True) or {}
    try:
        recs = recommend_lipsticks(
            skin_lab_std=(float(data["L"]), float(data["a"]), float(data["b"])),
            undertone=data.get("undertone", "neutral"),
            skin_type=data.get("skin_type", "normal"),
        )
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  LipMatch Backend")
    print("  http://localhost:5000")
    print("")
    print("  Setup checklist:")
    print("  1. pip install flask flask-cors opencv-python numpy scikit-learn joblib")
    print("  2. python TrainShadeModel.py   ← trains Random Forest")
    print("  3. python TrainSkinType.py     ← trains MobileNetV2 (optional)")
    print("  4. python app.py               ← starts server")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
