"""
IRON MAN HUD v4
═══════════════════════════════════════════════════════════════════
FACE-ANCHORED (moves with head):
  • LEFT EYE  – 4 tight concentric rings (innermost = solid, just bigger than iris)
  • RIGHT EYE – thin 2-ring scanner
  • Floating labels near forehead/cheek (distance, lock status, neural %)
  • Nose-tip targeting dot

SCREEN-FIXED (never moves):
  • LEFT panel  – Missile cross-section diagram with labels
  • RIGHT panel – Biometric + suit status with animated server bars & level meters
  • BOTTOM BAR  – Full-width status ticker
  • TOP BAR     – System header
  • CENTER      – Faint crosshair / depth grid
═══════════════════════════════════════════════════════════════════
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  COLOR PALETTE  (BGR)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C_CYAN    = (210, 200, 50)     # dominant HUD cyan
C_CYAN2   = (255, 235, 100)    # bright cyan highlight
C_GREEN   = (60,  210, 90)     # OK / active green
C_ORANGE  = (30,  140, 255)    # warning / missile accent
C_RED     = (45,  45,  220)    # alert red
C_WHITE   = (235, 240, 255)
C_DIM     = (85,  110, 135)
C_DDIM    = (45,  58,  72)     # very dim grid
C_DARK    = (12,  15,  20)     # panel fill

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MEDIAPIPE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOW-LEVEL HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def LM(lms, idx, W, H):
    return int(lms[idx].x * W), int(lms[idx].y * H)

def aline(dst, p1, p2, col, thick=1, a=1.0):
    if a >= 0.99:
        cv2.line(dst, p1, p2, col, thick, cv2.LINE_AA)
    else:
        ov = dst.copy(); cv2.line(ov, p1, p2, col, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, a, dst, 1-a, 0, dst)

def acircle(dst, c, r, col, thick=1, a=1.0):
    if r <= 0: return
    if a >= 0.99:
        cv2.circle(dst, c, r, col, thick, cv2.LINE_AA)
    else:
        ov = dst.copy(); cv2.circle(ov, c, r, col, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, a, dst, 1-a, 0, dst)

def aarc(dst, c, axes, rot, a1, a2, col, thick=1, a=1.0):
    if a >= 0.99:
        cv2.ellipse(dst, c, axes, rot, a1, a2, col, thick, cv2.LINE_AA)
    else:
        ov = dst.copy(); cv2.ellipse(ov, c, axes, rot, a1, a2, col, thick, cv2.LINE_AA)
        cv2.addWeighted(ov, a, dst, 1-a, 0, dst)

def panel_bg(dst, x, y, w, h, col_border=None, alpha=0.60):
    """Semi-transparent dark panel."""
    ov = dst.copy()
    cv2.rectangle(ov, (x,y), (x+w, y+h), C_DARK, -1)
    cv2.addWeighted(ov, alpha, dst, 1-alpha, 0, dst)
    if col_border:
        cv2.rectangle(dst, (x,y), (x+w, y+h), col_border, 1, cv2.LINE_AA)
        # corner ticks
        tk = min(12, w//6)
        for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            bx = x if sx<0 else x+w
            by = y if sy<0 else y+h
            cv2.line(dst,(bx,by),(bx-sx*tk,by),col_border,2,cv2.LINE_AA)
            cv2.line(dst,(bx,by),(bx,by-sy*tk),col_border,2,cv2.LINE_AA)

def gtxt(dst, text, pos, sc=0.36, col=C_CYAN, thick=1):
    """Glowing text via blur-add."""
    (tw,th),bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, sc, thick)
    if tw<=0 or th<=0: return
    pad=5
    p = np.zeros((th+bl+pad*2, tw+pad*2, 3), dtype=np.uint8)
    cv2.putText(p, text, (pad,th+pad), cv2.FONT_HERSHEY_DUPLEX, sc, col, thick+1, cv2.LINE_AA)
    g = cv2.GaussianBlur(p,(7,7),0)
    cv2.putText(g, text, (pad,th+pad), cv2.FONT_HERSHEY_DUPLEX, sc, col, thick, cv2.LINE_AA)
    x,y = pos[0]-pad, pos[1]-th-pad
    x2,y2 = x+g.shape[1], y+g.shape[0]
    if x>=0 and y>=0 and x2<dst.shape[1] and y2<dst.shape[0]:
        dst[y:y2,x:x2] = cv2.add(dst[y:y2,x:x2], g)

def stxt(dst, text, pos, sc=0.34, col=C_DIM, thick=1):
    cv2.putText(dst, text, pos, cv2.FONT_HERSHEY_DUPLEX, sc, col, thick, cv2.LINE_AA)

def seg_bar(dst, x, y, bw, bh, pct, col_fill, col_bg=C_DDIM, segs=12):
    """Segmented horizontal bar, pct 0-1."""
    cv2.rectangle(dst,(x,y),(x+bw,y+bh),col_bg,1)
    sw = (bw-2) // segs
    filled = int(pct * segs)
    for i in range(segs):
        bx = x+1+i*sw
        c  = col_fill if i < filled else C_DDIM
        cv2.rectangle(dst,(bx,y+1),(bx+sw-2,y+bh-2),c,-1)

def v_level(dst, x, y, bw, bh, pct, col):
    """Vertical level bar with tick marks."""
    cv2.rectangle(dst,(x,y),(x+bw,y+bh),C_DDIM,1)
    fh = int(bh * pct)
    cv2.rectangle(dst,(x+1,y+bh-fh),(x+bw-1,y+bh-1),col,-1)
    for i in range(5):
        ty = y + int(bh*(1-i/4))
        cv2.line(dst,(x-3,ty),(x,ty),C_DIM,1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FACE VIGNETTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def face_vignette(frame, lms, W, H):
    idxs = [10,338,297,332,284,251,389,356,454,323,361,288,
            397,365,379,378,400,377,152,148,176,149,150,136,
            172,58,132,93,234,127,162,21,54,103,67,109]
    pts  = np.array([[int(lms[i].x*W),int(lms[i].y*H)] for i in idxs],np.int32)
    mask = np.zeros((H,W),dtype=np.uint8)
    cv2.fillPoly(mask,[pts],255)
    mask = cv2.GaussianBlur(mask,(71,71),0)
    dark = (frame*0.58).astype(np.uint8)
    a3   = np.stack([mask/255.0]*3,axis=-1)
    frame[:] = (dark*a3 + frame*(1-a3)).astype(np.uint8)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EYE RINGS  (face-anchored)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eye_rings_left(dst, cx, cy, iris_r, t):
    """
    4 rings, tight spacing, innermost is SOLID just bigger than iris.
    R1 solid  |  R2 segmented CW  |  R3 arc-bracket CCW  |  R4 dashes CW-fast
    """
    # ── R1: SOLID filled ring (just outside iris) ────────────────────────
    r1 = iris_r + 4
    # glow fill
    ov = dst.copy()
    cv2.circle(ov,(cx,cy),r1,(40,35,8),-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.22,dst,0.78,0,dst)
    # solid bright rim
    cv2.circle(dst,(cx,cy),r1,C_CYAN2,2,cv2.LINE_AA)

    # ── R2: 14-segment rotating ring ────────────────────────────────────
    r2   = r1 + 10
    n2   = 14
    arc2 = int(360/n2) - 6
    sp2  = (t * 1.6) % 360
    for i in range(n2):
        a = int(sp2 + i*(360/n2))
        cv2.ellipse(dst,(cx,cy),(r2,r2),0,a,a+arc2,C_CYAN,1,cv2.LINE_AA)

    # ── R3: 3-arc bracket ring (counter-spin) ───────────────────────────
    r3  = r2 + 12
    sp3 = -(t*0.8) % 360
    for a0, a1 in [(0,85),(115,200),(230,330)]:
        aarc(dst,(cx,cy),(r3,r3),0,int(sp3+a0),int(sp3+a1),C_CYAN,2)
    # 8 tick marks on r3
    for i in range(8):
        ad = sp3 + i*(360/8)
        ar = math.radians(ad)
        x1=int(cx+(r3-4)*math.cos(ar)); y1=int(cy+(r3-4)*math.sin(ar))
        x2=int(cx+(r3+4)*math.cos(ar)); y2=int(cy+(r3+4)*math.sin(ar))
        cv2.line(dst,(x1,y1),(x2,y2),C_CYAN2 if i%2==0 else C_DIM,1,cv2.LINE_AA)

    # ── R4: fast dashed outer ring ───────────────────────────────────────
    r4  = r3 + 14
    n4  = 22
    sp4 = (t*2.8) % 360
    for i in range(n4):
        a = sp4 + i*(360/n4)
        cv2.ellipse(dst,(cx,cy),(r4,r4),0,int(a),int(a+360/n4*0.5),C_CYAN2,1,cv2.LINE_AA)

    # ── 4 orbiting corner brackets ───────────────────────────────────────
    r_orb = r4 + 12
    sp_o  = (t*0.45) % 360
    for base in [0,90,180,270]:
        ar = math.radians(base + sp_o)
        bx = int(cx + r_orb*math.cos(ar))
        by = int(cy + r_orb*math.sin(ar))
        cv2.circle(dst,(bx,by),3,C_CYAN2,-1,cv2.LINE_AA)
        aarc(dst,(bx,by),(7,7),0,base+sp_o-45,base+sp_o+45,C_CYAN,1)


def eye_rings_right(dst, cx, cy, iris_r, t):
    """Right eye: 2 subtle rings, asymmetric."""
    r1 = iris_r + 4
    cv2.circle(dst,(cx,cy),r1,C_DIM,1,cv2.LINE_AA)
    r2  = r1 + 14
    sp  = -(t*1.2) % 360
    aarc(dst,(cx,cy),(r2,r2),0,int(sp),int(sp+140),C_CYAN,2)
    aarc(dst,(cx,cy),(r2,r2),0,int(sp+180),int(sp+280),C_DIM,1,a=0.5)
    cv2.circle(dst,(cx,cy),3,C_CYAN2,-1,cv2.LINE_AA)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FLOATING FACE LABELS  (move with head)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _connector(dst, anchor, label_pos, col):
    """Small dot at anchor + line to label."""
    cv2.circle(dst, anchor, 2, col, -1, cv2.LINE_AA)
    aline(dst, anchor, label_pos, col, 1, a=0.45)

def floating_labels(dst, lms, W, H, scale, iris_dist, t):
    S = scale

    # ── 1. FOREHEAD — LOCK STATUS with bracket ───────────────────────────
    fhx, fhy = LM(lms, 10, W, H)
    ly = fhy - int(58*S)
    lx = fhx
    bk = 20
    for sx in [-1, 1]:
        cv2.line(dst,(lx+sx*bk, ly-10),(lx+sx*bk, ly+4), C_CYAN, 1, cv2.LINE_AA)
        cv2.line(dst,(lx+sx*bk, ly-10),(lx+sx*(bk-9), ly-10), C_CYAN, 1, cv2.LINE_AA)
    gtxt(dst, "BIO LOCK  OK", (lx-36, ly-12), 0.28, C_GREEN)

    # ── 2. ABOVE FOREHEAD — altitude / elevation readout ─────────────────
    ahx, ahy = LM(lms, 10, W, H)
    ay = ahy - int(88*S)
    alt_val = int(42 + 3*math.sin(t/28))
    stxt(dst, f"ALT  {alt_val:03d} M", (ahx - 30, ay), 0.28, C_DIM)
    aline(dst,(ahx, ahy - int(58*S)),(ahx, ay+2), C_DDIM, 1, a=0.35)

    # ── 3. LEFT CHEEK — distance estimate ────────────────────────────────
    lcx, lcy = LM(lms, 234, W, H)
    dist_val  = int(iris_dist * 2.1)
    tx = max(6, lcx - int(95*S))
    _connector(dst, (lcx - int(8*S), lcy), (tx + 68, lcy), C_DDIM)
    stxt(dst, f"DIST  {dist_val} mm", (tx, lcy + 4), 0.28, C_DIM)

    # ── 4. LEFT CHEEK LOWER — threat scan ────────────────────────────────
    lbx, lby = LM(lms, 172, W, H)
    tx2 = max(6, lbx - int(90*S))
    threat = "NONE" if (t // 60) % 3 != 0 else "LOW"
    col_t = C_GREEN if threat == "NONE" else C_ORANGE
    _connector(dst, (lbx - int(6*S), lby), (tx2 + 72, lby), C_DDIM)
    gtxt(dst, f"THREAT  {threat}", (tx2, lby + 4), 0.26, col_t)

    # ── 5. RIGHT CHEEK — neural sync ─────────────────────────────────────
    rcx, rcy = LM(lms, 454, W, H)
    nsync = 94.2 + 2*math.sin(t/35)
    rx = min(W - 130, rcx + int(14*S))
    _connector(dst, (rcx + int(8*S), rcy), (rx, rcy), C_DDIM)
    gtxt(dst, f"SYNC  {nsync:.1f}%", (rx, rcy + 4), 0.28, C_CYAN)

    # ── 6. RIGHT CHEEK UPPER — facial geometry ───────────────────────────
    rux, ruy = LM(lms, 323, W, H)
    rx2 = min(W - 150, rux + int(14*S))
    _connector(dst, (rux + int(6*S), ruy), (rx2, ruy), C_DDIM)
    stxt(dst, "GEO  STABLE", (rx2, ruy + 4), 0.26, C_DIM)

    # ── 7. RIGHT CHEEK LOWER — O2 / temp ─────────────────────────────────
    rlx, rly = LM(lms, 361, W, H)
    o2_val  = 97.5 + 0.5*math.sin(t/50)
    rx3 = min(W - 130, rlx + int(14*S))
    _connector(dst, (rlx + int(6*S), rly), (rx3, rly), C_DDIM)
    stxt(dst, f"O2  {o2_val:.1f}%", (rx3, rly + 4), 0.26, C_GREEN)

    # ── 8. NOSE BRIDGE — depth / focal ───────────────────────────────────
    nbx, nby = LM(lms, 6, W, H)
    focal = int(80 + 5*math.sin(t/20))
    stxt(dst, f"FOCAL  {focal}mm", (nbx + int(18*S), nby), 0.26, C_DDIM)

    # ── 9. LEFT EYE OUTER — iris data ────────────────────────────────────
    iox, ioy = LM(lms, 33, W, H)
    tx3 = max(6, iox - int(75*S))
    stxt(dst, f"IRIS  {int(iris_dist):03d}px", (tx3, ioy + 4), 0.26, C_DDIM)
    aline(dst,(iox - int(4*S), ioy),(tx3 + 58, ioy), C_DDIM, 1, a=0.30)

    # ── 10. RIGHT EYE OUTER — pupil dilation sim ─────────────────────────
    rex, rey = LM(lms, 263, W, H)
    dil = 3.2 + 0.4*math.sin(t/22)
    rx4 = min(W - 130, rex + int(14*S))
    stxt(dst, f"DIL  {dil:.1f}mm", (rx4, rey + 4), 0.26, C_DDIM)
    aline(dst,(rex + int(4*S), rey),(rx4, rey), C_DDIM, 1, a=0.30)

    # ── 11. CHIN — scan progress bar ─────────────────────────────────────
    chx, chy = LM(lms, 152, W, H)
    pct = (t % 90) / 90.0
    seg_bar(dst, chx - 38, chy + int(16*S), 76, 7, pct, C_CYAN, segs=8)
    stxt(dst, f"SCAN  {int(pct*100)}%", (chx - 38, chy + int(30*S)), 0.26, C_DIM)

    # ── 12. BELOW CHIN — heart rate sim ──────────────────────────────────
    bhx, bhy = LM(lms, 152, W, H)
    hr = int(72 + 4*math.sin(t/30))
    bhy2 = bhy + int(50*S)
    hr_col = C_GREEN if 60 <= hr <= 100 else C_RED
    gtxt(dst, f"HR  {hr} BPM", (bhx - 32, min(H-10, bhy2)), 0.28, hr_col)

    # ── 13. LEFT TEMPLE — temp readout ───────────────────────────────────
    tmx, tmy = LM(lms, 127, W, H)
    temp = 36.6 + 0.2*math.sin(t/60)
    tx4 = max(6, tmx - int(70*S))
    stxt(dst, f"TEMP  {temp:.1f}C", (tx4, tmy + 4), 0.26, C_DIM)

    # ── 14. RIGHT TEMPLE — power cell ────────────────────────────────────
    rtx, rty = LM(lms, 356, W, H)
    pwr = 98 + 2*math.sin(t/45)
    rx5 = min(W - 130, rtx + int(14*S))
    gtxt(dst, f"PWR  {pwr:.0f}%", (rx5, rty + 4), 0.26, C_CYAN)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LEFT PANEL — MISSILE DIAGRAM  (screen-fixed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_missile_panel(dst, W, H, t):
    pw, ph = 210, 270
    px, py = 14, 36                      # ← moved to top of screen

    panel_bg(dst, px, py, pw, ph, C_ORANGE)

    # Title
    gtxt(dst,"GUIDED MICRO-PROJECTILE",(px+6,py+16),0.28,C_ORANGE)
    cv2.line(dst,(px+4,py+20),(px+pw-4,py+20),C_DDIM,1)

    # ── Missile body diagram ─────────────────────────────────────────────
    mx   = px + 28
    my   = py + 80                       # sits comfortably near top of panel
    mlen = pw - 50
    mh   = 14

    # Nose cone
    nose_pts = np.array([(mx,    my),
                         (mx+28, my-mh),
                         (mx+28, my+mh)], np.int32)
    cv2.polylines(dst,[nose_pts],True,C_ORANGE,1,cv2.LINE_AA)

    # Main body
    bx1, bx2 = mx+28, mx+mlen-22
    cv2.rectangle(dst,(bx1,my-mh),(bx2,my+mh),C_ORANGE,1,cv2.LINE_AA)

    # Internal section lines
    for sx in [bx1+30, bx1+60, bx1+90]:
        if sx < bx2:
            cv2.line(dst,(sx,my-mh),(sx,my+mh),C_DDIM,1)

    # Engine nozzle (trapezoid)
    noz_pts = np.array([(bx2,    my-mh),
                        (bx2,    my+mh),
                        (bx2+22, my+mh-6),
                        (bx2+22, my-mh+6)], np.int32)
    cv2.polylines(dst,[noz_pts],True,C_ORANGE,1,cv2.LINE_AA)

    # Fins
    fin_x = bx1 + 70
    for sy in [-1, 1]:
        fin_pts = np.array([(fin_x,    my+sy*mh),
                            (fin_x-15, my+sy*(mh+16)),
                            (fin_x+20, my+sy*mh)], np.int32)
        cv2.polylines(dst,[fin_pts],True,C_ORANGE,1,cv2.LINE_AA)

    # Warhead hatches
    for hx in range(mx+6,mx+24,5):
        cv2.line(dst,(hx,my-6),(hx+4,my+6),C_DDIM,1)

    # Pulsing exhaust
    pulse = int(3*abs(math.sin(t/12)))
    for i in range(3):
        ex = bx2+22+i*4+pulse
        cv2.line(dst,(ex,my-mh+6+i*2),(ex+4+i,my+mh-6-i*2),C_RED,1,cv2.LINE_AA)

    # Callout labels
    labels = [
        (mx+14,  my-mh-4, "WARHEAD"),
        (bx1+20, my-mh-4, "GUIDANCE"),
        (bx1+55, my-mh-4, "PROPEL"),
        (bx1+85, my-mh-4, "FUEL"),
        (bx2+8,  my-mh-4, "NOZZLE"),
    ]
    for lx2, ly2, label in labels:
        if px < lx2 < px+pw:
            cv2.line(dst,(lx2,ly2),(lx2,ly2-6),C_DIM,1)
            stxt(dst,label,(max(px+2,lx2-18),ly2-8),0.20,C_DIM)

    # Stats
    sy0 = my + mh + 20
    rows_m = [("WEIGHT","4.2 KG"),("RANGE","800 M"),
              ("VELOCITY","340 M/S"),("STATUS","READY")]
    for i,(lbl,val) in enumerate(rows_m):
        stxt(dst,lbl,(px+8, sy0+i*18),0.26,C_DIM)
        gtxt(dst,val,(px+pw-80, sy0+i*18),0.26,C_GREEN if val=="READY" else C_ORANGE)

    # Charge bar
    charge = 0.72 + 0.15*math.sin(t/50)
    stxt(dst,"CHARGE",(px+8,sy0+82),0.24,C_DIM)
    seg_bar(dst,px+65,sy0+74,pw-72,8,charge,C_ORANGE,segs=10)

    # Lock indicator
    lock_on = (t//18)%2==0
    lk_col  = C_RED if lock_on else C_DDIM
    cv2.circle(dst,(px+pw-14,py+ph-14),5,lk_col,-1,cv2.LINE_AA)
    stxt(dst,"TGT",(px+pw-46,py+ph-8),0.24,lk_col)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LEFT LOWER — ARC REACTOR DIAGRAM  (screen-fixed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_arc_reactor_panel(dst, W, H, t):
    """
    Arc reactor cross-section diagram with concentric open rings at
    different gap angles — like an engineering schematic.
    """
    pw, ph = 210, 220
    px     = 14
    py     = H - ph - 28             # lower-left corner

    panel_bg(dst, px, py, pw, ph, C_CYAN)
    gtxt(dst,"ARC REACTOR  MK-VII",(px+6,py+16),0.28,C_CYAN2)
    cv2.line(dst,(px+4,py+20),(px+pw-4,py+20),C_DDIM,1)

    # Centre of the reactor diagram
    cx = px + pw//2
    cy = py + 42 + 68           # vertically centred in top 2/3 of panel

    # ── Pulsing core glow ────────────────────────────────────────────────
    pulse_r = int(3*abs(math.sin(t/18)))
    ov = dst.copy()
    cv2.circle(ov,(cx,cy),10+pulse_r,(180,160,40),-1,cv2.LINE_AA)
    cv2.addWeighted(ov,0.30,dst,0.70,0,dst)
    cv2.circle(dst,(cx,cy),10+pulse_r,C_CYAN2,1,cv2.LINE_AA)
    cv2.circle(dst,(cx,cy),4,C_WHITE,-1,cv2.LINE_AA)

    # ── Ring definitions: (radius, gap_start_deg, gap_span_deg, color, thickness)
    #    Each ring is a near-complete arc with one deliberate opening
    rings = [
        (18,   30,  60,  C_CYAN,  1),   # innermost, small gap top-right
        (28,  120,  45,  C_CYAN2, 2),   # second ring, gap right
        (38,  210,  70,  C_CYAN,  1),   # third ring, gap bottom-left
        (50,  300,  50,  C_DIM,   1),   # fourth ring, gap bottom
        (62,   60,  80,  C_CYAN2, 1),   # fifth ring, gap top-right wide
        (74,  170,  40,  C_CYAN,  1),   # sixth ring, gap left-ish
    ]

    spin_speeds = [1.2, -0.7, 0.9, -0.5, 0.4, -1.1]  # each ring spins differently

    for i, (r, gap_start, gap_span, col, thick) in enumerate(rings):
        # Rotate the gap position over time
        offset = (t * spin_speeds[i]) % 360
        # Arc goes from gap_end back around to gap_start (the complement)
        a_start = int((gap_start + gap_span + offset) % 360)
        a_end   = int((gap_start + offset) % 360)
        if a_end <= a_start:
            a_end += 360
        cv2.ellipse(dst,(cx,cy),(r,r),0,a_start,a_end,col,thick,cv2.LINE_AA)

        # Small tick at the gap edges
        for edge_a in [gap_start + offset, gap_start + gap_span + offset]:
            ar  = math.radians(edge_a % 360)
            ex1 = int(cx + (r-4)*math.cos(ar))
            ey1 = int(cy + (r-4)*math.sin(ar))
            ex2 = int(cx + (r+4)*math.cos(ar))
            ey2 = int(cy + (r+4)*math.sin(ar))
            cv2.line(dst,(ex1,ey1),(ex2,ey2),C_CYAN2,1,cv2.LINE_AA)

    # ── 6 radial spoke lines (faint, static) ────────────────────────────
    for a_deg in range(0,360,60):
        ar  = math.radians(a_deg + t*0.2)
        x1  = int(cx + 12*math.cos(ar))
        y1  = int(cy + 12*math.sin(ar))
        x2  = int(cx + 70*math.cos(ar))
        y2  = int(cy + 70*math.sin(ar))
        aline(dst,(x1,y1),(x2,y2),C_DDIM,1,a=0.35)

    # ── Stats below reactor ──────────────────────────────────────────────
    sy = cy + 82
    output_pct = 0.88 + 0.08*math.sin(t/40)
    temp_c     = int(3200 + 120*math.sin(t/55))
    flux       = 1.21 + 0.04*math.sin(t/25)

    rows_r = [
        ("OUTPUT",  f"{int(output_pct*100)}%",  C_CYAN),
        ("TEMP",    f"{temp_c} K",              C_ORANGE),
        ("FLUX",    f"{flux:.2f} GW",           C_CYAN2),
    ]
    for i,(lbl,val,col) in enumerate(rows_r):
        stxt(dst,lbl,(px+8, sy+i*18),0.25,C_DIM)
        gtxt(dst,val,(px+pw-82, sy+i*18),0.25,col)

    # Output bar
    seg_bar(dst,px+8,sy+58,pw-16,7,output_pct,C_CYAN,segs=12)
    stxt(dst,f"CORE STABLE",(px+8,sy+74),0.23,C_GREEN)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RIGHT PANEL — BIOMETRICS + SUIT + SERVER BARS  (screen-fixed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_right_panel(dst, W, H, t):
    pw, ph = 230, H - 80
    px, py = W - pw - 14, 40
    panel_bg(dst, px, py, pw, ph, C_CYAN)

    # ── Header ───────────────────────────────────────────────────────────
    gtxt(dst,"STARK  SYSTEMS",(px+8,py+16),0.36,C_CYAN2,1)
    cv2.line(dst,(px+4,py+20),(px+pw-4,py+20),C_DIM,1)

    # ── Biometrics section ───────────────────────────────────────────────
    gtxt(dst,"BIOMETRICS",(px+8,py+36),0.28,C_CYAN)
    hr  = int(72+4*math.sin(t/30))
    o2  = 97.5+0.5*math.sin(t/50)
    syn = 94+1.5*math.sin(t/40)
    bio = [("HR",f"{hr} BPM",hr/120,C_GREEN),
           ("O2",f"{o2:.1f}%",o2/100,C_GREEN),
           ("SYNC",f"{syn:.1f}%",syn/100,C_CYAN)]
    for i,(lbl,val,pct,col) in enumerate(bio):
        y0 = py+52+i*32
        stxt(dst,lbl,(px+8,y0+10),0.28,C_DIM)
        gtxt(dst,val,(px+pw-90,y0+10),0.28,col)
        seg_bar(dst,px+8,y0+14,pw-16,7,pct,col,segs=14)

    cv2.line(dst,(px+4,py+154),(px+pw-4,py+154),C_DDIM,1)

    # ── Suit status section ──────────────────────────────────────────────
    gtxt(dst,"SUIT STATUS",(px+8,py+168),0.28,C_CYAN)
    suit = [("POWER",   1.00, C_GREEN),
            ("SHIELD",  0.88, C_GREEN),
            ("THRUSTR", 1.00, C_GREEN),
            ("COMMS",   0.95, C_GREEN),
            ("WEAPONS", 0.72, C_ORANGE),
            ("COOLING", 0.60, C_ORANGE)]
    for i,(lbl,pct,col) in enumerate(suit):
        y0 = py+182+i*26
        stxt(dst,lbl,(px+8,y0+10),0.26,C_DIM)
        seg_bar(dst,px+70,y0+2,pw-80,10,pct,col,segs=10)
        stxt(dst,f"{int(pct*100)}%",(px+pw-32,y0+10),0.25,col)

    cv2.line(dst,(px+4,py+345),(px+pw-4,py+345),C_DDIM,1)

    # ── Server / load bars (vertical level meters) ───────────────────────
    gtxt(dst,"SERVER LOAD",(px+8,py+360),0.28,C_CYAN)
    n_srv = 8
    srv_w = (pw-24)//n_srv
    for i in range(n_srv):
        base_load = [0.85,0.62,0.91,0.44,0.77,0.55,0.88,0.70][i]
        noise     = 0.08*math.sin(t/8.0+i*1.3)
        load      = max(0.1, min(1.0, base_load + noise))
        col       = C_RED if load>0.85 else (C_ORANGE if load>0.65 else C_GREEN)
        vx        = px+12+i*srv_w
        vy        = py+372
        v_level(dst,vx,vy,srv_w-4,80,load,col)
        stxt(dst,f"S{i+1}",(vx,vy+92),0.20,C_DDIM)

    cv2.line(dst,(px+4,py+470),(px+pw-4,py+470),C_DDIM,1)

    # ── Network / data columns ───────────────────────────────────────────
    gtxt(dst,"DATA STREAM",(px+8,py+484),0.28,C_CYAN)
    for ci,cx_off in enumerate([8,60,115,168]):
        for ri in range(6):
            val = f"{(t*3+ci*43+ri*19)%1000:03d}"
            hi  = ri==(t//10)%6
            stxt(dst,val,(px+cx_off,py+500+ri*16),0.26,C_CYAN2 if hi else C_DDIM)

    # ── Timestamp ────────────────────────────────────────────────────────
    ts_y = py+ph-10
    cv2.line(dst,(px+4,ts_y-16),(px+pw-4,ts_y-16),C_DDIM,1)
    stxt(dst,time.strftime("T+ %H:%M:%S"),(px+8,ts_y),0.26,C_DIM)
    stxt(dst,f"FPS --",(px+pw-68,ts_y),0.26,C_DDIM)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TOP BAR  (screen-fixed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_top_bar(dst, W, fps_val, t):
    cv2.rectangle(dst,(0,0),(W,32),C_DARK,-1)
    cv2.line(dst,(0,32),(W,32),C_DIM,1)
    gtxt(dst,"STARK INDUSTRIES  //  J.A.R.V.I.S  v7.4",(10,22),0.34,C_CYAN)
    gtxt(dst,time.strftime("%Y-%m-%d  %H:%M:%S"),(W//2-100,22),0.34,C_DIM)
    gtxt(dst,f"FPS {fps_val:.0f}  //  SYS OK",(W-170,22),0.34,C_GREEN)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BOTTOM BAR  (screen-fixed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ticker_msgs = [
    "SCANNING ENVIRONMENT  //  THREAT LEVEL: NONE  //  ",
    "GUIDED MICRO-PROJECTILES: ARMED  //  ",
    "POWER CORE: 100%  //  ARC REACTOR STABLE  //  ",
    "FACIAL GEOMETRY LOCKED  //  BIOMETRIC AUTH: OK  //  ",
    "MK-50 WEIGHT ANALYSIS: NOMINAL  //  ",
    "REPULSOR CALIBRATION: COMPLETE  //  ",
    "COMM ARRAY: ACTIVE  //  ENCRYPTION: ON  //  ",
]
_ticker_str  = "  ".join(_ticker_msgs) * 3
_ticker_pos  = 0

def draw_bottom_bar(dst, W, H, t):
    global _ticker_pos
    bh = 28
    by = H - bh
    cv2.rectangle(dst,(0,by),(W,H),C_DARK,-1)
    cv2.line(dst,(0,by),(W,by),C_DIM,1)

    # Scrolling ticker
    _ticker_pos = (_ticker_pos + 1) % (len(_ticker_str) * 7)
    disp = _ticker_str[(_ticker_pos//7):(_ticker_pos//7)+120]
    stxt(dst, disp, (8, H-8), 0.30, C_DIM)

    # Right: static labels
    gtxt(dst,"MK-50  //  IRON MAN",(W-190,H-8),0.30,C_CYAN)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CENTER DEPTH GRID  (screen-fixed, very faint)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def draw_center_grid(dst, W, H):
    cx, cy = W//2, H//2
    # Faint crosshair
    aline(dst,(cx-60,cy),(cx-14,cy),C_DDIM,1,a=0.4)
    aline(dst,(cx+14,cy),(cx+60,cy),C_DDIM,1,a=0.4)
    aline(dst,(cx,cy-60),(cx,cy-14),C_DDIM,1,a=0.4)
    aline(dst,(cx,cy+14),(cx,cy+60),C_DDIM,1,a=0.4)
    acircle(dst,(cx,cy),6,C_DDIM,1,a=0.4)
    # Horizon line
    aline(dst,(0,cy),(W,cy),C_DDIM,1,a=0.15)
    # Depth perspective lines (vanishing point centre)
    for dx in [-200,-100,100,200]:
        aline(dst,(cx+dx,0),(cx,cy),C_DDIM,1,a=0.10)
        aline(dst,(cx+dx,H),(cx,cy),C_DDIM,1,a=0.10)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  RED FLOATING FACE ANNOTATIONS  (move with head)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

C_RED1 = (60,  70,  210)    # bright red
C_RED2 = (50,  55,  180)    # medium red
C_RED3 = (40,  42,  145)    # dim red
C_RDIM = (35,  38,  110)    # very dim red (connector lines)

def _rconnect(dst, anchor, label_pos):
    cv2.circle(dst, anchor, 2, C_RED2, -1, cv2.LINE_AA)
    aline(dst, anchor, label_pos, C_RDIM, 1, a=0.40)

def rtxt(dst, text, pos, sc=0.28, col=C_RED1, thick=1):
    cv2.putText(dst, text, pos, cv2.FONT_HERSHEY_DUPLEX, sc, col, thick, cv2.LINE_AA)

def floating_red_labels(dst, lms, W, H, scale, iris_dist, t):
    S = scale

    # ── 1. LEFT EYEBROW — cortex activity ────────────────────────────────
    ebx, eby = LM(lms, 55, W, H)
    ca = 78 + int(8*abs(math.sin(t/22)))
    lpos = (max(4, ebx - int(90*S)), eby - int(12*S))
    _rconnect(dst, (ebx - int(4*S), eby), (lpos[0]+70, lpos[1]))
    rtxt(dst, f"CORTEX  {ca}%",       lpos,                  0.26, C_RED1)
    rtxt(dst, f"ACT: {(t*2)%999:03d}", (lpos[0], lpos[1]+14), 0.22, C_RED3)

    # ── 2. RIGHT EYEBROW — retinal scan ──────────────────────────────────
    rebx, reby = LM(lms, 285, W, H)
    rpos = (min(W-140, rebx + int(10*S)), reby - int(12*S))
    _rconnect(dst, (rebx + int(4*S), reby), (rpos[0], rpos[1]))
    rtxt(dst, "RETINAL SCAN",     rpos,                   0.26, C_RED1)
    rtxt(dst, "AUTH  PASS",       (rpos[0], rpos[1]+14),  0.22, C_RED2)

    # ── 3. LEFT JAW — skeletal map ────────────────────────────────────────
    jwx, jwy = LM(lms, 136, W, H)
    jpos = (max(4, jwx - int(100*S)), jwy)
    _rconnect(dst, (jwx - int(6*S), jwy), (jpos[0]+80, jwy))
    rtxt(dst, "SKELETAL MAP",      jpos,                  0.26, C_RED2)
    rtxt(dst, f"NODE  {(t+7)%16:02d}/16", (jpos[0], jwy+14), 0.22, C_RED3)

    # ── 4. RIGHT JAW — muscular tension ──────────────────────────────────
    rjx, rjy = LM(lms, 365, W, H)
    rjpos = (min(W-150, rjx + int(10*S)), rjy)
    _rconnect(dst, (rjx + int(6*S), rjy), (rjpos[0], rjy))
    tension = int(18 + 6*math.sin(t/28))
    rtxt(dst, "MUSC TENSION",    rjpos,                  0.26, C_RED2)
    rtxt(dst, f"{tension}  N/cm²", (rjpos[0], rjy+14),   0.22, C_RED3)

    # ── 5. LEFT SIDE FACE — blood flow ───────────────────────────────────
    bfx, bfy = LM(lms, 93, W, H)
    bfpos = (max(4, bfx - int(85*S)), bfy)
    _rconnect(dst, (bfx - int(5*S), bfy), (bfpos[0]+68, bfy))
    flow = 4.2 + 0.3*math.sin(t/33)
    rtxt(dst, "BLOOD FLOW",       bfpos,                 0.26, C_RED1)
    rtxt(dst, f"{flow:.1f} L/min", (bfpos[0], bfy+14),   0.22, C_RED3)

    # ── 6. RIGHT SIDE FACE — nerve signal ────────────────────────────────
    nsx, nsy = LM(lms, 323, W, H)
    nspos = (min(W-150, nsx + int(10*S)), nsy + int(20*S))
    _rconnect(dst, (nsx + int(5*S), nsy + int(18*S)), (nspos[0], nspos[1]))
    nerve = int(312 + 14*math.sin(t/19))
    rtxt(dst, "NERVE SIG",        nspos,                  0.26, C_RED2)
    rtxt(dst, f"{nerve}  ms",     (nspos[0], nspos[1]+14), 0.22, C_RED3)

    # ── 7. UPPER LIP — vocal tract ───────────────────────────────────────
    vlx, vly = LM(lms, 13, W, H)
    vlpos = (min(W-160, vlx + int(16*S)), vly)
    _rconnect(dst, (vlx + int(8*S), vly), (vlpos[0], vly))
    rtxt(dst, "VOCAL TRACT",      vlpos,                  0.26, C_RED2)
    rtxt(dst, "IDLE",             (vlpos[0], vly+14),     0.22, C_RED3)

    # ── 8. LEFT UNDER-EYE — micro-expression ─────────────────────────────
    mex, mey = LM(lms, 117, W, H)
    mepos = (max(4, mex - int(80*S)), mey + int(10*S))
    _rconnect(dst, (mex - int(4*S), mey + int(8*S)), (mepos[0]+64, mepos[1]))
    expressions = ["NEUTRAL","FOCUSED","ALERT","CALM"]
    expr = expressions[(t//45) % len(expressions)]
    rtxt(dst, "MICRO-EXPR",       mepos,                  0.24, C_RED2)
    rtxt(dst, expr,               (mepos[0], mepos[1]+13), 0.22, C_RED1)

    # ── 9. RIGHT UNDER-EYE — saccade tracking ────────────────────────────
    sex, sey = LM(lms, 346, W, H)
    sepos = (min(W-155, sex + int(10*S)), sey + int(10*S))
    _rconnect(dst, (sex + int(4*S), sey + int(8*S)), (sepos[0], sepos[1]))
    sac = int(abs(math.sin(t/14)) * 180)
    rtxt(dst, "SACCADE",          sepos,                  0.24, C_RED2)
    rtxt(dst, f"{sac:03d}°  /s",  (sepos[0], sepos[1]+13), 0.22, C_RED3)

    # ── 10. PHILTRUM — facial symmetry score ─────────────────────────────
    phx, phy = LM(lms, 164, W, H)
    sym = 97.2 + 0.8*math.sin(t/60)
    phpos = (min(W-140, phx + int(16*S)), phy)
    _rconnect(dst, (phx + int(8*S), phy), (phpos[0], phy))
    rtxt(dst, f"SYMM  {sym:.1f}%", phpos, 0.24, C_RED2)

    # ── 11. LEFT NOSTRIL — respiration rate ──────────────────────────────
    nrx, nry = LM(lms, 240, W, H)
    resp = int(14 + 2*math.sin(t/55))
    nrpos = (max(4, nrx - int(88*S)), nry)
    _rconnect(dst, (nrx - int(4*S), nry), (nrpos[0]+70, nry))
    rtxt(dst, f"RESP  {resp}/min", nrpos, 0.24, C_RED3)

    # ── 12. LEFT CHEEKBONE — bone density estimate ───────────────────────
    bnx, bny = LM(lms, 116, W, H)
    bnpos = (max(4, bnx - int(80*S)), bny - int(18*S))
    _rconnect(dst, (bnx - int(4*S), bny - int(14*S)), (bnpos[0]+66, bnpos[1]))
    rtxt(dst, "BONE  1.24 g/cc", bnpos, 0.22, C_RED3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN LOOP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

t         = 0
fps_t     = time.time()
fps_n     = 0
fps_val   = 0.0

print("IRON MAN HUD v4  —  Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    t    += 1
    fps_n += 1
    if fps_n >= 30:
        fps_val = fps_n / (time.time()-fps_t)
        fps_t   = time.time()
        fps_n   = 0

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # ── HUD overlay ──────────────────────────────────────────────────────
    hud = np.zeros_like(frame)

    # ── SCREEN-FIXED elements on HUD ─────────────────────────────────────
    draw_center_grid(hud, W, H)
    draw_missile_panel(hud, W, H, t)
    draw_arc_reactor_panel(hud, W, H, t)
    draw_right_panel(hud, W, H, t)
    draw_top_bar(hud, W, fps_val, t)
    draw_bottom_bar(hud, W, H, t)

    # ── FACE-ANCHORED elements ────────────────────────────────────────────
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        lms  = face.landmark

        if len(lms) > 478:
            lix, liy = LM(lms,468,W,H)
            rix, riy = LM(lms,473,W,H)
        else:
            lix, liy = LM(lms,33, W,H)
            rix, riy = LM(lms,263,W,H)

        iris_dist = max(55, math.hypot(rix-lix, riy-liy))
        scale     = max(0.65, iris_dist/90.0)
        iris_r    = max(14, int(iris_dist * 0.27))   # approx physical iris radius

        # Darken face
        face_vignette(frame, lms, W, H)

        # Eye rings
        eye_rings_left (hud, lix, liy, iris_r, t)
        eye_rings_right(hud, rix, riy, iris_r, t)

        # Floating cyan labels
        floating_labels(hud, lms, W, H, scale, iris_dist, t)

        # Floating red annotations
        floating_red_labels(hud, lms, W, H, scale, iris_dist, t)

    else:
        msg = f"SCANNING{'.'*(t%4)}"
        tw  = cv2.getTextSize(msg,cv2.FONT_HERSHEY_DUPLEX,0.7,1)[0][0]
        gtxt(hud, msg, (W//2-tw//2+20, H//2), 0.7, C_CYAN, 2)

    # ── Final composite ───────────────────────────────────────────────────
    pulse = 0.82 + 0.10*math.sin(time.time()*1.8)
    out   = cv2.addWeighted(frame, 1.0, hud, pulse, 0)

    # Update FPS in right panel (direct write, avoids one-frame lag)
    stxt(out, f"FPS {fps_val:.0f}", (W-14-64, H-80-10), 0.26, C_CYAN2)

    cv2.imshow("IRON MAN HUD v4", out)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()