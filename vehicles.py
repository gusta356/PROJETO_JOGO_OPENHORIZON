"""Fisica de veiculos, colisao e trafego para o Horizon Panda3D.

Python puro (sem Panda3D) para ficar leve e testavel.

Convencoes (iguais ao Panda3D):
  * +X = direita, +Y = frente, Z = cima
  * heading em graus, positivo = anti-horario (setH do Panda).
    frente do carro = (-sin h, cos h)   direita = (cos h, sin h)

Resumo do que existe aqui
  * Vehicle    : modelo "bicicleta" (esterco -> taxa de guinada) + aderencia
                 lateral (slip angle) + superficie (asfalto/acostamento/grama).
  * sat / resolve : colisao OBB x OBB (SAT) com impulso, atrito e giro.
  * World      : carros + obstaculos estaticos + IA de trafego (IDM +
                 pure-pursuit) que enxerga o jogador e os outros carros.
"""
import math
import random

# ---------------------------------------------------------------- pista
ROAD_HALF = 6.0          # meia largura do asfalto
SHOULDER = 7.2           # asfalto + acostamento/zebra
MAIN_HALF_LEN = 800.0
CROSS_Y = 60.0
CROSS_HALF_LEN = 420.0
RING = 2.0 * MAIN_HALF_LEN
LANE_X = 3.0             # centro de cada faixa (mao direita: +X sobe, -X desce)
WORLD_LIMIT = 900.0

# (aderencia, arrasto constante, arrasto proporcional a velocidade)
SURF_ROAD = (1.0, 0.0, 0.0)
SURF_SHOULDER = (0.92, 0.4, 0.01)
SURF_GRASS = (0.55, 2.2, 0.09)


def surface_at(x, y):
    d = 999.0
    if abs(y) < MAIN_HALF_LEN:
        d = abs(x)
    if abs(x) < CROSS_HALF_LEN:
        d = min(d, abs(y - CROSS_Y))
    if d <= ROAD_HALF:
        return SURF_ROAD
    if d <= SHOULDER:
        return SURF_SHOULDER
    return SURF_GRASS


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def _toward_zero(v, amount):
    if v > 0:
        return max(0.0, v - amount)
    return min(0.0, v + amount)


# ---------------------------------------------------------------- corpos
class Static:
    """Obstaculo fixo alinhado aos eixos (casa, arvore, poste, mureta...)."""
    is_static = True
    inv_mass = 0.0
    inv_inertia = 0.0
    vx = vy = yaw = 0.0
    c = 1.0
    s = 0.0

    def __init__(self, x, y, hw, hl):
        self.x, self.y, self.hw, self.hl = x, y, hw, hl
        self.radius = math.hypot(hw, hl)

    def axes(self):
        return 1.0, 0.0, 0.0, 1.0   # right(x,y), forward(x,y)


class Vehicle:
    is_static = False

    def __init__(self, x, y, heading=0.0, scale=1.0, mass=1300.0, player=False):
        self.x, self.y = x, y
        self.heading = heading
        self.vx = self.vy = 0.0
        self.yaw = 0.0                      # rad/s, anti-horario positivo
        self.scale = scale
        self.hl = 2.22 * scale              # meio comprimento
        self.hw = 1.12 * scale              # meia largura
        self.radius = math.hypot(self.hl, self.hw)
        self.wheelbase = 2.9 * scale
        self.mass = mass * scale * scale
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia = 1.0 / (self.mass * ((2 * self.hl) ** 2 + (2 * self.hw) ** 2) / 12.0)
        self.is_player = player
        # ajustes
        self.rolling = 0.8
        self.aero = 0.0006
        self.engine_brake = 3.5 if player else 0.0
        self.lat_grip = 11.0                # 1/s: quao rapido o carro "agarra"
        self.max_lat_acc = 16.0             # m/s^2 de aderencia lateral
        self.scrub = 0.08                   # perda de velocidade ao arrastar pneu
        self.steer_max_deg = 34.0
        # comandos (preenchidos pelo jogador ou pela IA)
        self.drive = 0.0                    # m/s^2 aplicados a frente (negativo = re)
        self.brake = 0.0                    # m/s^2 de frenagem
        self.steer = 0.0                    # -1..1 (+ = direita)
        self.hb = 0.0                       # freio de mao 0..1
        # estado derivado (para visual/HUD)
        self.steer_angle = 0.0              # rad, + = direita
        self.acc_long = 0.0
        self.lean = 0.0                     # >0: corpo inclina para a direita
        self.surface = SURF_ROAD
        self.node = None
        self.wheels = []
        self.ai = None
        self.refresh_trig()

    # -- geometria
    def refresh_trig(self):
        h = math.radians(self.heading)
        self.c = math.cos(h)
        self.s = math.sin(h)

    def axes(self):
        return self.c, self.s, -self.s, self.c

    def forward_speed(self):
        return -self.vx * self.s + self.vy * self.c

    def speed(self):
        return math.hypot(self.vx, self.vy)

    # -- fisica
    def max_steer(self, vf):
        return math.radians(self.steer_max_deg * (1.0 + 0.4 * self.hb)) / (1.0 + (abs(vf) / 10.0) ** 2)

    def step(self, dt, surf):
        grip, dconst, dlin = surf
        self.surface = surf
        h = math.radians(self.heading)
        c, s = math.cos(h), math.sin(h)
        fx, fy, rx, ry = -s, c, c, s
        vf = self.vx * fx + self.vy * fy
        vl = self.vx * rx + self.vy * ry
        vf0 = vf

        # --- longitudinal
        vf += self.drive * dt
        if self.brake > 0.0:
            vf = _toward_zero(vf, self.brake * (0.4 + 0.6 * grip) * dt)
        drag = self.rolling + self.aero * vf * vf + dconst + dlin * abs(vf)
        if self.drive == 0.0 and self.brake == 0.0:
            drag += self.engine_brake
        vf = _toward_zero(vf, drag * dt)

        # --- direcao (modelo bicicleta): taxa de guinada = v/L * tan(esterco)
        ang = self.steer * self.max_steer(vf)
        self.steer_angle = ang
        target = -(vf / self.wheelbase) * math.tan(ang) * (1.0 + 0.25 * self.hb)
        cap = self.max_lat_acc * grip * (1.0 + 0.5 * self.hb) / max(abs(vf), 2.0)
        target = max(-cap, min(cap, target))
        self.yaw += (target - self.yaw) * min(1.0, 7.0 * (0.5 + 0.5 * grip) * dt)
        self.heading = wrap_deg(self.heading + math.degrees(self.yaw) * dt)

        # --- velocidade no mundo (eixos antigos) -> eixos novos
        vx = fx * vf + rx * vl
        vy = fy * vf + ry * vl
        h = math.radians(self.heading)
        c, s = math.cos(h), math.sin(h)
        fx, fy, rx, ry = -s, c, c, s
        vf = vx * fx + vy * fy
        vl = vx * rx + vy * ry
        sp = math.hypot(vx, vy)

        # --- aderencia lateral: gira o vetor velocidade em direcao ao nariz
        if sp > 0.05:
            sgn = 1.0 if vf >= 0.0 else -1.0
            beta = math.atan2(vl * sgn, abs(vf))
            k = self.lat_grip * grip * (1.0 - 0.82 * self.hb)
            nb = beta * math.exp(-k * dt)
            maxd = self.max_lat_acc * grip * (1.0 - 0.55 * self.hb) / max(sp, 3.0) * dt
            d = beta - nb
            if abs(d) > maxd:
                nb = beta - math.copysign(maxd, d)
            sp *= 1.0 - self.scrub * abs(beta - nb)
            cb, sb = math.cos(nb), math.sin(nb)
            vx = sp * sgn * (fx * cb + rx * sb)
            vy = sp * sgn * (fy * cb + ry * sb)
        elif self.drive == 0.0:
            vx = vy = 0.0

        self.vx, self.vy = vx, vy
        self.x += vx * dt
        self.y += vy * dt
        self.c, self.s = c, s

        vf_new = -vx * s + vy * c
        if dt > 0:
            a = (vf_new - vf0) / dt
            self.acc_long += (a - self.acc_long) * min(1.0, 8.0 * dt)
        self.lean += (self.yaw * vf_new - self.lean) * min(1.0, 8.0 * dt)


# ---------------------------------------------------------------- colisao
def corners(b):
    rx, ry, fx, fy = b.axes()
    out = []
    for sw, sl in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        out.append((b.x + rx * b.hw * sw + fx * b.hl * sl,
                    b.y + ry * b.hw * sw + fy * b.hl * sl))
    return out


def _inside(b, px, py, eps=0.02):
    rx, ry, fx, fy = b.axes()
    dx, dy = px - b.x, py - b.y
    return abs(dx * rx + dy * ry) <= b.hw + eps and abs(dx * fx + dy * fy) <= b.hl + eps


def sat(a, b):
    """Retorna (profundidade, nx, ny) com normal apontando de b para a, ou None."""
    arx, ary, afx, afy = a.axes()
    brx, bry, bfx, bfy = b.axes()
    dx, dy = a.x - b.x, a.y - b.y
    best = None
    for ux, uy in ((arx, ary), (afx, afy), (brx, bry), (bfx, bfy)):
        ra = abs(ux * arx + uy * ary) * a.hw + abs(ux * afx + uy * afy) * a.hl
        rb = abs(ux * brx + uy * bry) * b.hw + abs(ux * bfx + uy * bfy) * b.hl
        dist = dx * ux + dy * uy
        ov = ra + rb - abs(dist)
        if ov <= 0.0:
            return None
        if best is None or ov < best[0]:
            sg = 1.0 if dist >= 0.0 else -1.0
            best = (ov, ux * sg, uy * sg)
    return best


def contact_point(a, b):
    pts = [p for p in corners(a) if _inside(b, *p)] + [p for p in corners(b) if _inside(a, *p)]
    if not pts:
        return (a.x + b.x) * 0.5, (a.y + b.y) * 0.5
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


def resolve(a, b, depth, nx, ny, restitution=0.28, mu=0.35):
    """Separa os corpos e aplica impulso (linear + giro). Retorna a velocidade de impacto."""
    ima, imb = a.inv_mass, b.inv_mass
    tot = ima + imb
    if tot == 0.0:
        return 0.0
    push = depth + 0.004
    a.x += nx * push * ima / tot
    a.y += ny * push * ima / tot
    if imb:
        b.x -= nx * push * imb / tot
        b.y -= ny * push * imb / tot
    cx, cy = contact_point(a, b)
    rax, ray = cx - a.x, cy - a.y
    rbx, rby = cx - b.x, cy - b.y
    vax, vay = a.vx - a.yaw * ray, a.vy + a.yaw * rax
    vbx, vby = b.vx - b.yaw * rby, b.vy + b.yaw * rbx
    rvx, rvy = vax - vbx, vay - vby
    vn = rvx * nx + rvy * ny
    if vn >= 0.0:
        return 0.0
    raxn = rax * ny - ray * nx
    rbxn = rbx * ny - rby * nx
    denom = ima + imb + raxn * raxn * a.inv_inertia + rbxn * rbxn * b.inv_inertia
    j = -(1.0 + restitution) * vn / denom
    a.vx += j * nx * ima
    a.vy += j * ny * ima
    a.yaw += a.inv_inertia * j * raxn
    if imb:
        b.vx -= j * nx * imb
        b.vy -= j * ny * imb
        b.yaw -= b.inv_inertia * j * rbxn
    # atrito tangencial (raspar a lataria)
    tx, ty = -ny, nx
    vt = rvx * tx + rvy * ty
    raxt = rax * ty - ray * tx
    rbxt = rbx * ty - rby * tx
    dent = ima + imb + raxt * raxt * a.inv_inertia + rbxt * rbxt * b.inv_inertia
    jt = max(-mu * j, min(mu * j, -vt / dent))
    a.vx += jt * tx * ima
    a.vy += jt * ty * ima
    a.yaw += a.inv_inertia * jt * raxt
    if imb:
        b.vx -= jt * tx * imb
        b.vy -= jt * ty * imb
        b.yaw -= b.inv_inertia * jt * rbxt
    for v in (a, b):
        if not v.is_static:
            v.yaw = max(-3.0, min(3.0, v.yaw))
    return -vn


# ---------------------------------------------------------------- IA
class Driver:
    def __init__(self, lane_x, direction, v0):
        self.lane_x = lane_x
        self.dir = direction
        self.v0 = v0
        self.hit_timer = 0.0
        self.stuck = 0.0


def idm_accel(v, v0, gap, v_lead, a_max=2.4, b=3.2, T=1.3, s0=3.0):
    """Intelligent Driver Model: acelera livre ou segue o carro da frente."""
    v = max(v, 0.0)
    free = 1.0 - (v / max(v0, 0.1)) ** 4
    if gap is None:
        return a_max * free
    dv = v - v_lead
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a_max * b)))
    return max(-10.0, a_max * (free - (s_star / max(gap, 0.1)) ** 2))


AI_DT = 1.0 / 15.0
RR_MAX = 5.4 ** 2


class World:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.player = None
        self.traffic = []
        self.vehicles = []
        self.statics = []
        self.ai_timer = 0.0

    # -- construcao
    def add_static(self, x, y, hw, hl):
        st = Static(x, y, hw, hl)
        self.statics.append(st)
        return st

    def set_player(self, v):
        self.player = v
        self.vehicles.append(v)
        return v

    def add_traffic(self, lane_x, y, direction, scale, v0):
        v = Vehicle(lane_x, y, 0.0 if direction > 0 else 180.0, scale, 1200.0)
        v.ai = Driver(lane_x, direction, v0)
        v.vy = direction * v0 * self.rng.uniform(0.85, 1.0)
        self.traffic.append(v)
        self.vehicles.append(v)
        return v

    # -- IA
    def _leader(self, v):
        d = v.ai
        best_gap, best_v = None, 0.0
        for o in self.vehicles:
            if o is v:
                continue
            dy = (o.y - v.y) * d.dir
            if o.ai is not None:
                dy %= RING
            if dy <= 0.0 or dy > 120.0:
                continue
            dx = o.x - v.x
            ext_x = o.hw * abs(o.c) + o.hl * abs(o.s)
            if abs(dx) > v.hw + ext_x + 0.5:
                continue
            ext_y = o.hl * abs(o.c) + o.hw * abs(o.s)
            gap = dy - v.hl - ext_y
            if best_gap is None or gap < best_gap:
                best_gap, best_v = gap, o.vy * d.dir
        return best_gap, best_v

    def _control_traffic(self, dt):
        for v in self.traffic:
            d = v.ai
            vf = v.forward_speed()
            fx, fy, rx, ry = -v.s, v.c, v.c, v.s
            gap, vlead = self._leader(v)

            want_heading = 0.0 if d.dir > 0 else 180.0
            herr = abs(wrap_deg(v.heading - want_heading))
            lane_err = abs(v.x - d.lane_x)
            d.hit_timer = max(0.0, d.hit_timer - dt)
            v0 = d.v0
            if d.hit_timer > 0.0:
                v0 *= 0.5
            if lane_err > 1.8 or herr > 30.0:
                v0 = min(v0, 7.0)
            a = idm_accel(vf, v0, gap, vlead)

            drag = v.rolling + v.aero * vf * vf
            if a >= -drag:
                v.drive, v.brake = min(a + drag, 4.5), 0.0
            else:
                v.drive, v.brake = 0.0, min(-a - drag, 11.0)

            # pure pursuit: mira um ponto a frente no centro da faixa
            ld = min(26.0, max(7.0, 5.0 + 0.8 * max(vf, 0.0)))
            tx, ty = d.lane_x - v.x, d.dir * ld
            df = tx * fx + ty * fy
            dr = tx * rx + ty * ry
            if df > 0.5:
                ang = math.atan(v.wheelbase * 2.0 * dr / (df * df + dr * dr))
            else:
                ang = math.copysign(1.0, dr) * 0.6
            v.steer = max(-1.0, min(1.0, ang / max(v.max_steer(vf), 1e-3)))
            v.hb = 0.0

            # preso/fora da pista por muito tempo -> reaparece
            if v.speed() < 0.8 and lane_err > 2.0 or abs(v.x) > 60.0:
                d.stuck += dt
                if d.stuck > 5.0:
                    self._respawn(v)
            else:
                d.stuck = 0.0

    def _free_spot(self, lane_x, min_gap=26.0):
        for _ in range(30):
            y = self.rng.uniform(-MAIN_HALF_LEN + 5, MAIN_HALF_LEN - 5)
            ok = True
            for o in self.vehicles:
                if math.hypot(o.x - lane_x, o.y - y) < (80.0 if o.is_player else min_gap):
                    ok = False
                    break
            if ok:
                return y
        return None

    def _respawn(self, v):
        d = v.ai
        y = self._free_spot(d.lane_x)
        d.stuck = 0.0
        if y is None:
            return
        v.x, v.y = d.lane_x, y
        v.heading = 0.0 if d.dir > 0 else 180.0
        v.vx, v.vy, v.yaw = 0.0, d.dir * d.v0 * 0.8, 0.0
        v.refresh_trig()

    # -- passo da simulacao
    def step(self, dt):
        """Avanca dt segundos. Retorna lista de (veiculo, outro|None, velocidade_de_impacto)."""
        events = []
        self.ai_timer += dt
        if self.ai_timer >= AI_DT:
            self._control_traffic(self.ai_timer)
            self.ai_timer = 0.0

        vs = self.vehicles
        for v in vs:
            v.step(dt, surface_at(v.x, v.y))

        # veiculo x veiculo
        n = len(vs)
        for i in range(n):
            a = vs[i]
            for j in range(i + 1, n):
                b = vs[j]
                dx, dy = a.x - b.x, a.y - b.y
                if dx * dx + dy * dy > RR_MAX:
                    continue
                hit = sat(a, b)
                if hit is None:
                    continue
                imp = resolve(a, b, *hit)
                if imp > 0.0:
                    events.append((a, b, imp))
                    for v in (a, b):
                        if v.ai is not None and imp > 1.5:
                            v.ai.hit_timer = 3.0

        # veiculo x obstaculos fixos
        for v in vs:
            if v.ai is not None and abs(v.x) < 5.5:
                continue
            for st in self.statics:
                lim = st.radius + v.radius
                if abs(v.x - st.x) > lim or abs(v.y - st.y) > lim:
                    continue
                hit = sat(v, st)
                if hit is None:
                    continue
                imp = resolve(v, st, *hit, restitution=0.2, mu=0.25)
                if imp > 0.0:
                    # raspar/bater em objeto fixo tira velocidade
                    v.vx *= 0.97
                    v.vy *= 0.97
                    events.append((v, None, imp))
                    if v.ai is not None and imp > 1.5:
                        v.ai.hit_timer = 3.0

        # limites do mundo e "wrap" do trafego
        for v in vs:
            lim = WORLD_LIMIT
            if abs(v.x) > lim:
                v.x = math.copysign(lim, v.x)
                v.vx = 0.0
            if abs(v.y) > lim:
                v.y = math.copysign(lim, v.y)
                v.vy = 0.0
            if v.ai is not None:
                wrapped = False
                if v.ai.dir > 0 and v.y > MAIN_HALF_LEN:
                    v.y -= RING
                    wrapped = True
                elif v.ai.dir < 0 and v.y < -MAIN_HALF_LEN:
                    v.y += RING
                    wrapped = True
                if wrapped:   # novo "motorista" ao reaparecer: mantem variedade de velocidades
                    v.ai.v0 = self.rng.uniform(13.0, 21.0)
            v.refresh_trig()
        return events
