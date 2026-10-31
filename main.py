from direct.showbase.ShowBase import ShowBase
from direct.gui.DirectGui import DirectFrame, DirectLabel, DirectButton, DirectWaitBar
from panda3d.core import (
    AmbientLight, DirectionalLight, Vec3, Vec4, TextNode, WindowProperties,
    Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
    GeomVertexWriter, TransparencyAttrib, Fog
)
from direct.task import Task
import math, random, json, os

from vehicles import (
    World, Vehicle, LANE_X, RING, MAIN_HALF_LEN, SURF_GRASS, wrap_deg
)

SAVE_FILE = "save.json"


class HorizonPanda3D(ShowBase):
    def __init__(self):
        super().__init__()
        self.disableMouse()
        self.setBackgroundColor(0.48, 0.68, 0.88)
        self.win.setClearColor(Vec4(0.48, 0.68, 0.88, 1))
        props = WindowProperties()
        props.setTitle("Open Horizon Panda3D - Drive Edition")
        self.win.requestProperties(props)

        # fisica/colisao/trafego vivem em vehicles.py (World)
        self.world = World(seed=11)
        self.speed = 0.0            # velocidade a frente do jogador (m/s), atualizada todo frame
        self.max_speed = 52.0
        self.accel = 20.0
        self.brake = 32.0
        self.reverse_accel = 11.0
        self.heading = 0.0
        self.steer_input = 0.0
        self.hb = 0.0               # freio de mao suavizado (0..1)
        self.tilt_p = 0.0           # inclinacao visual da carroceria
        self.tilt_r = 0.0
        self.cam_heading = 0.0
        self.shake = 0.0
        self.base_fov = self.camLens.getFov()[0]
        self.customizer_open = False
        self.vehicle_type = "car"
        self.vehicle_selected = False
        self.transmission_mode = "manual"
        self.menu_vehicle_choice = "car"
        self.menu_transmission_choice = "manual"
        self.game_mode = "free"
        self.menu_game_mode_choice = "free"
        self.race_state = "inactive"
        self.race_timer = 0.0
        self.race_countdown = 0.0
        self.race_checkpoint_index = 0
        self.race_checkpoints = [-300.0, 100.0, 500.0]
        self.race_finish_y = 740.0
        self.race_start_y = -735.0
        self.race_rivals = []
        self.race_scene_nodes = []
        self.race_traffic_backup = None
        self.race_position = 1
        self.race_awarded = False
        self.wheelie = 0.0
        self.wheelie_angle = 0.0
        self.car_color_index = 0
        self.wheel_style_index = 0
        self.car_colors = [(0.055,0.22,0.82,1),(0.82,0.06,0.05,1),(0.05,0.62,0.22,1),(0.88,0.62,0.04,1),(0.10,0.10,0.12,1),(0.82,0.82,0.84,1),(0.48,0.12,0.68,1)]
        self.wheel_styles = ["Sport", "Classic", "Dark"]
        self.nitro = 100.0
        self.nitro_active = False
        # Câmbio manual: -1 = Ré, 0 = Neutro, 1..6 = marchas para frente.
        self.current_gear = 1
        self.max_gear = 6
        self.shift_cooldown = 0.0
        self.rpm = 900.0
        self.gear_speed_limits = {1: 11.0, 2: 19.0, 3: 27.0, 4: 33.5, 5: 37.5, 6: 48.0}
        self.gear_torque = {1: 1.34, 2: 1.16, 3: 1.00, 4: 0.90, 5: 0.82, 6: 0.74}
        self.camera_mode = 0
        self.credits = 5000
        self.radar_cooldown = 0.0
        self.keys = {k: False for k in ["w", "s", "a", "d", "shift", "space", "tab"]}
        self.radar_nodes = []

        self.setup_lighting()
        self.setup_world()
        self.setup_player()
        self.setup_traffic()
        self.setup_hud()
        self.setup_vehicle_menu()
        self.bind_inputs()
        self.load_game()
        self.taskMgr.add(self.update, "update")

    # ---------- basic mesh helpers ----------
    def make_mesh(self, name, vertices, faces, color=(1, 1, 1, 1), parent=None):
        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData(name, fmt, Geom.UHStatic)
        vdata.setNumRows(len(vertices))
        vw = GeomVertexWriter(vdata, "vertex")
        nw = GeomVertexWriter(vdata, "normal")
        cw = GeomVertexWriter(vdata, "color")

        # smooth-ish normals from the center
        cx = sum(v[0] for v in vertices) / max(1, len(vertices))
        cy = sum(v[1] for v in vertices) / max(1, len(vertices))
        cz = sum(v[2] for v in vertices) / max(1, len(vertices))
        for x, y, z in vertices:
            vw.addData3(x, y, z)
            n = Vec3(x - cx, y - cy, max(0.15, z - cz))
            if n.lengthSquared() < 0.001:
                n = Vec3(0, 0, 1)
            n.normalize()
            nw.addData3(n)
            cw.addData4(*color)

        prim = GeomTriangles(Geom.UHStatic)
        for face in faces:
            if len(face) == 3:
                prim.addVertices(*face)
            elif len(face) == 4:
                a, b, c, d = face
                prim.addVertices(a, b, c)
                prim.addVertices(a, c, d)
        prim.closePrimitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)
        node = GeomNode(name)
        node.addGeom(geom)
        np = (parent or self.render).attachNewNode(node)
        np.setTwoSided(True)
        return np

    def make_round_prism(self, name, length, width, height, color, parent=None, segments=12):
        """Rounded vehicle-like prism: elliptical rings from nose to tail."""
        verts = []
        rings = [
            (-length * 0.50, width * 0.32, height * 0.45),
            (-length * 0.42, width * 0.48, height * 0.62),
            (-length * 0.18, width * 0.52, height * 0.72),
            ( length * 0.18, width * 0.52, height * 0.72),
            ( length * 0.42, width * 0.46, height * 0.58),
            ( length * 0.50, width * 0.28, height * 0.42),
        ]
        for y, rx, rz in rings:
            for i in range(segments):
                a = math.tau * i / segments
                x = math.cos(a) * rx
                z = math.sin(a) * rz + height * 0.48
                verts.append((x, y, z))
        faces = []
        nr = len(rings)
        for r in range(nr - 1):
            for i in range(segments):
                j = (i + 1) % segments
                a = r * segments + i
                b = r * segments + j
                c = (r + 1) * segments + j
                d = (r + 1) * segments + i
                faces.append((a, b, c, d))
        faces.append(tuple(range(segments - 1, -1, -1)))
        last = (nr - 1) * segments
        faces.append(tuple(last + i for i in range(segments)))
        return self.make_mesh(name, verts, faces, color, parent)

    def make_cylinder(self, name, radius, depth, color, parent=None, segments=18, axis="x"):
        verts = []
        for side in (-1, 1):
            for i in range(segments):
                a = math.tau * i / segments
                c, s = math.cos(a), math.sin(a)
                if axis == "x":
                    verts.append((side * depth / 2, c * radius, s * radius))
                else:
                    verts.append((c * radius, side * depth / 2, s * radius))
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((i, j, segments + j, segments + i))
        # caps as triangles from first point fan (good enough visually)
        for i in range(1, segments - 1):
            faces.append((0, i + 1, i))
            faces.append((segments, segments + i, segments + i + 1))
        return self.make_mesh(name, verts, faces, color, parent)

    def box(self, name, pos, scale, color, parent=None, solid=False):
        # retained only for flat environment pieces such as asphalt and walls
        parent = parent or self.render
        m = self.loader.loadModel("models/box")
        m.reparentTo(parent)
        m.setName(name)
        m.setPos(*pos)
        m.setScale(*scale)
        m.setColor(*color)
        if solid:
            self.register_solid(m)
        return m

    def register_solid(self, np):
        """Registra o volume do objeto (em coordenadas do mundo) como obstaculo de colisao."""
        bounds = np.getTightBounds(self.render)
        if not bounds:
            return
        lo, hi = bounds
        self.world.add_static((lo.x + hi.x) / 2, (lo.y + hi.y) / 2,
                              max(0.05, (hi.x - lo.x) / 2), max(0.05, (hi.y - lo.y) / 2))

    def setup_lighting(self):
        amb = AmbientLight("ambient")
        amb.setColor(Vec4(0.28, 0.30, 0.34, 1))
        self.render.setLight(self.render.attachNewNode(amb))

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(1.18, 1.10, 0.92, 1))
        sun.setShadowCaster(True, 2048, 2048)
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-30, -58, 0)
        self.render.setLight(sun_np)

        fog = Fog("distance_fog")
        fog.setColor(0.48, 0.68, 0.88)
        fog.setLinearRange(260, 650)
        self.render.setFog(fog)

    # ---------- world ----------
    def setup_world(self):
        self.road_width = 12
        self.box("ground", (0, 0, -0.8), (900, 900, 0.6), (0.10, 0.34, 0.12, 1))

        # roads
        self.box("road_main", (0, 0, 0), (self.road_width / 2, 800, 0.08), (0.075, 0.078, 0.085, 1))
        self.box("road_cross", (0, 60, 0.02), (420, self.road_width / 2, 0.08), (0.075, 0.078, 0.085, 1))

        # shoulders and edge lines
        for x in (-self.road_width / 2 - 0.42, self.road_width / 2 + 0.42):
            self.box("shoulder", (x, 0, 0.05), (0.42, 800, 0.05), (0.32, 0.32, 0.34, 1))
        for x in (-self.road_width / 2 + 0.18, self.road_width / 2 - 0.18):
            self.box("edge_line", (x, 0, 0.13), (0.055, 798, 0.018), (0.96, 0.96, 0.93, 1))
        for y in range(-790, 791, 14):
            self.box("lane", (0, y, 0.14), (0.07, 4.2, 0.018), (0.95, 0.82, 0.16, 1))
        for x in range(-410, 411, 14):
            self.box("lane_cross", (x, 60, 0.15), (4.2, 0.07, 0.018), (0.95, 0.82, 0.16, 1))

        # detailed roadside: rumble strips, reflectors, drainage and signs
        for y in range(-795, 796, 6):
            stripe = (0.86, 0.10, 0.08, 1) if (y // 6) % 2 == 0 else (0.94, 0.94, 0.92, 1)
            for side in (-1, 1):
                self.box("rumble", (side*(self.road_width/2+0.78), y, 0.12), (0.28, 2.8, 0.035), stripe)
                self.make_cylinder("reflector", 0.06, 0.16, (1.0,0.80,0.20,1), self.render, 10, "x").setPos(side*(self.road_width/2+0.20), y, 0.20)
        for y in range(-780, 781, 30):
            for side in (-1, 1):
                self.box("drain", (side*(self.road_width/2+1.25), y, 0.08), (0.42, 1.6, 0.03), (0.18,0.18,0.19,1))
        for y, label_x in [(-620, -10.5), (-260, 10.5), (180, -10.5), (560, 10.5)]:
            self.box("sign_post", (label_x, y, 1.2), (0.05,0.05,1.2), (0.28,0.28,0.30,1), solid=True)
            self.box("sign", (label_x, y, 2.55), (0.9,0.10,0.55), (0.10,0.35,0.72,1))

        # guard rails in selected stretches
        for side in (-1, 1):
            gx = side * 8.0
            for y in range(-720, -500, 8):
                self.box("rail", (gx, y, 0.75), (0.055, 3.7, 0.08), (0.58, 0.60, 0.62, 1), solid=True)
                self.box("rail_post", (gx, y, 0.4), (0.05, 0.05, 0.38), (0.36, 0.38, 0.40, 1))

        random.seed(7)
        # houses with pitched roofs, doors, windows
        house_colors = [(0.70, 0.43, 0.28, 1), (0.62, 0.68, 0.74, 1), (0.80, 0.65, 0.36, 1), (0.54, 0.70, 0.52, 1)]
        for _ in range(150):
            side = random.choice([-1, 1])
            x = side * random.uniform(22, 180)
            y = random.uniform(-760, 760)
            if abs(y - 60) < 24:
                y += 40 if y < 60 else -40
            self.create_house(x, y, random.choice(house_colors), random.uniform(0.9, 1.25))

        # trees and roadside vegetation
        for _ in range(240):
            x = random.choice([-1, 1]) * random.uniform(14, 220)
            y = random.uniform(-780, 780)
            if abs(y - 60) < 14 and abs(x) < 120:
                continue
            self.create_tree(x, y, random.uniform(0.8, 1.5))

        # street lights
        for y in range(-740, 741, 45):
            for side in (-1, 1):
                x = side * 9.8
                self.box("lamp_pole", (x, y, 2.7), (0.055, 0.055, 2.7), (0.22, 0.23, 0.24, 1), solid=True)
                self.make_cylinder("lamp", 0.18, 0.38, (0.92, 0.88, 0.60, 1), self.render, 14, "x").setPos(x - side * 0.15, y, 5.35)

        # radars
        for y in (-600, -250, 180, 520):
            self.box("radar_pole", (7.5, y, 1.8), (0.10, 0.10, 1.8), (0.25, 0.25, 0.25, 1), solid=True)
            cam = self.make_round_prism("radar_cam", 0.65, 0.8, 0.55, (0.75, 0.77, 0.80, 1), self.render, 10)
            cam.setPos(7.2, y, 3.55)
            cam.setH(90)
            self.radar_nodes.append((Vec3(0, y, 0), 70.0))

        # distant buildings with varied silhouettes
        for x in (-130, 130):
            for y in (-650, -250, 250, 650):
                self.create_building(x, y, random.randint(10, 18))

    def create_house(self, x, y, wall_color, s=1.0):
        root = self.render.attachNewNode("house")
        root.setPos(x, y, 0)
        w, d, h = 5.2 * s, 6.2 * s, 3.8 * s
        self.box("walls", (0, 0, h / 2), (w / 2, d / 2, h / 2), wall_color, root, solid=True)
        # pitched roof mesh
        z = h
        verts = [(-w/2-0.25, -d/2-0.25, z), (w/2+0.25, -d/2-0.25, z),
                 (-w/2-0.25, d/2+0.25, z), (w/2+0.25, d/2+0.25, z),
                 (0, -d/2-0.25, z+1.7*s), (0, d/2+0.25, z+1.7*s)]
        faces = [(0,1,4), (2,5,3), (0,4,5,2), (1,3,5,4)]
        self.make_mesh("roof", verts, faces, (0.34, 0.10, 0.07, 1), root)
        self.box("door", (0, -d/2-0.03, 1.0*s), (0.55*s, 0.05, 1.0*s), (0.24, 0.13, 0.07, 1), root)
        for wx in (-1.55*s, 1.55*s):
            self.box("window", (wx, -d/2-0.055, 2.4*s), (0.55*s, 0.035, 0.52*s), (0.52, 0.78, 0.92, 1), root)

    def create_tree(self, x, y, s):
        trunk = self.make_cylinder("tree_trunk", 0.20*s, 2.6*s, (0.28, 0.16, 0.07, 1), self.render, 12, "y")
        trunk.setPos(x, y, 1.3*s)
        trunk.setP(90)
        crown = self.make_round_prism("tree_crown", 2.3*s, 2.5*s, 3.3*s, (0.08, 0.38, 0.12, 1), self.render, 12)
        crown.setPos(x, y, 2.8*s)
        self.world.add_static(x, y, 0.35*s, 0.35*s)   # tronco

    def create_building(self, x, y, h):
        root = self.render.attachNewNode("building")
        root.setPos(x, y, 0)
        self.box("tower", (0, 0, h/2), (7, 7, h/2), (0.34, 0.38, 0.43, 1), root, solid=True)
        # window strips make blocks look less plain
        for z in range(2, h, 3):
            for sx in (-7.03, 7.03):
                self.box("windows", (sx, 0, z), (0.03, 5.5, 0.55), (0.30, 0.58, 0.72, 1), root)

    # ---------- cars ----------
    def create_car_visual(self, parent, color, scale=1.0, player=False):
        # main rounded body
        body = self.make_round_prism("car_body", 4.5*scale, 2.2*scale, 1.15*scale, color, parent, 16)
        body.setZ(0.22*scale)

        # cabin/windshield as another rounded body, narrower and higher
        glass = self.make_round_prism("glass_cabin", 2.25*scale, 1.65*scale, 0.86*scale,
                                     (0.08, 0.15, 0.20, 0.96), parent, 14)
        glass.setPos(0, -0.05*scale, 0.88*scale)
        glass.setTransparency(TransparencyAttrib.MAlpha)

        # wheels - actual cylinders
        wheels = []
        for sx in (-1.05, 1.05):
            for sy in (-1.45, 1.45):
                w = self.make_cylinder("wheel", 0.40*scale, 0.34*scale, (0.025, 0.025, 0.028, 1), parent, 18, "x")
                w.setPos(sx*scale, sy*scale, 0.38*scale)
                wheels.append(w)
                # o aro e filho da roda: gira e esterca junto com ela
                self.make_cylinder("rim", 0.19*scale, 0.36*scale, (0.55, 0.57, 0.60, 1), w, 14, "x")

        # headlights / taillights
        for sx in (-0.68, 0.68):
            light = self.make_round_prism("headlight", 0.18*scale, 0.34*scale, 0.18*scale,
                                          (1.0, 0.94, 0.68, 1), parent, 10)
            light.setPos(sx*scale, 2.22*scale, 0.64*scale)
            tail = self.make_round_prism("taillight", 0.16*scale, 0.28*scale, 0.16*scale,
                                         (0.90, 0.03, 0.02, 1), parent, 10)
            tail.setPos(sx*scale, -2.18*scale, 0.62*scale)

        if player:
            # visible dashboard for cockpit view
            dash = self.make_round_prism("dashboard", 0.55*scale, 1.75*scale, 0.34*scale,
                                         (0.045, 0.048, 0.052, 1), parent, 12)
            dash.setPos(0, 0.75*scale, 1.14*scale)
            wheel = self.make_cylinder("steering", 0.30*scale, 0.07*scale,
                                       (0.06, 0.06, 0.065, 1), parent, 20, "y")
            wheel.setPos(-0.38*scale, 0.58*scale, 1.28*scale)
            wheel.setP(70)
        return wheels


    def create_motorcycle_visual(self, parent, color, scale=1.0, player=False):
        """Moto procedural: rodas redondas, tanque, banco, quadro e guidão."""
        # quadro
        frame = self.make_cylinder("bike_frame", 0.08*scale, 2.25*scale, (0.10,0.10,0.11,1), parent, 12, "y")
        frame.setPos(0, 0.02*scale, 0.62*scale)
        frame.setP(14)

        # tanque e carenagem
        tank = self.make_round_prism("bike_tank", 1.15*scale, 0.78*scale, 0.62*scale, color, parent, 14)
        tank.setPos(0, 0.35*scale, 0.78*scale)
        fairing = self.make_round_prism("bike_fairing", 0.78*scale, 0.68*scale, 0.78*scale, color, parent, 14)
        fairing.setPos(0, 0.98*scale, 0.80*scale)

        # banco
        seat = self.make_round_prism("bike_seat", 0.95*scale, 0.56*scale, 0.22*scale, (0.055,0.055,0.06,1), parent, 12)
        seat.setPos(0, -0.55*scale, 0.94*scale)

        # garfo dianteiro
        for sx in (-0.18, 0.18):
            fork = self.make_cylinder("fork", 0.035*scale, 0.95*scale, (0.52,0.54,0.56,1), parent, 10, "y")
            fork.setPos(sx*scale, 1.10*scale, 0.72*scale)
            fork.setP(-18)

        # rodas: traseira primeiro, dianteira depois
        wheels=[]
        for sy in (-1.05, 1.18):
            w=self.make_cylinder("bike_wheel", 0.46*scale, 0.20*scale, (0.02,0.02,0.022,1), parent, 20, "x")
            w.setPos(0, sy*scale, 0.48*scale)
            self.make_cylinder("bike_rim", 0.24*scale, 0.21*scale, (0.58,0.60,0.64,1), w, 16, "x")
            wheels.append(w)

        # guidão
        bar=self.make_cylinder("handlebar", 0.035*scale, 0.92*scale, (0.18,0.18,0.20,1), parent, 10, "x")
        bar.setPos(0, 0.92*scale, 1.32*scale)
        # farol
        lamp=self.make_round_prism("bike_headlight", 0.18*scale, 0.30*scale, 0.28*scale, (1.0,0.94,0.72,1), parent, 10)
        lamp.setPos(0, 1.37*scale, 1.02*scale)
        # lanterna
        tail=self.make_round_prism("bike_tail", 0.16*scale, 0.26*scale, 0.18*scale, (0.95,0.03,0.02,1), parent, 10)
        tail.setPos(0, -1.18*scale, 0.91*scale)
        return wheels

    def setup_player(self):
        self.car = self.render.attachNewNode("player_car")
        self.car.setPos(LANE_X, -700, 0.18)          # comeca na faixa direita, nao na linha central
        self.player_visual_root = self.car.attachNewNode("visual_root")
        self.player_wheels = self.create_car_visual(self.player_visual_root, self.car_colors[self.car_color_index], 1.0, True)
        self.player = Vehicle(LANE_X, -700.0, 0.0, 1.0, 1400.0, player=True)
        self.player.node = self.car
        self.player.wheels = self.player_wheels
        self.world.set_player(self.player)
        self.heading = 0.0
        self.cam_heading = 0.0

    def setup_traffic(self):
        colors = [(0.76, 0.05, 0.05, 1), (0.06, 0.58, 0.18, 1), (0.88, 0.64, 0.06, 1),
                  (0.46, 0.12, 0.66, 1), (0.78, 0.80, 0.84, 1), (0.05, 0.32, 0.60, 1)]
        per_lane = 24
        # mao direita: faixa +X sobe (+Y), faixa -X desce (-Y); carros espacados, sem sobreposicao
        for lane_x, direction in ((LANE_X, 1), (-LANE_X, -1)):
            for k in range(per_lane):
                y = -MAIN_HALF_LEN + (k + 0.5) * RING / per_lane + random.uniform(-8, 8)
                if direction > 0 and abs(y - self.player.y) < 40:
                    y += 45
                if y > MAIN_HALF_LEN:
                    y -= RING
                scale = random.uniform(0.88, 1.05)
                v = self.world.add_traffic(lane_x, y, direction, scale, random.uniform(13.0, 21.0))
                v.node = self.render.attachNewNode(f"traffic_{len(self.world.traffic)}")
                v.node.setPos(v.x, v.y, 0.18)
                v.node.setH(v.heading)
                v.wheels = self.create_car_visual(v.node, random.choice(colors), scale, False)

    # ---------- HUD / controls ----------
    def setup_hud(self):
        self.hud = DirectFrame(frameColor=(0, 0, 0, 0.48), frameSize=(-0.33, 0.33, -0.16, 0.16), pos=(-0.64, 0, 0.78))
        self.speed_label = DirectLabel(parent=self.hud, text="0 km/h", scale=0.07, pos=(0, 0, 0.06), text_align=TextNode.ACenter)
        self.gear_label = DirectLabel(parent=self.hud, text="MARCHA 1", scale=0.050, pos=(0, 0, -0.025), text_align=TextNode.ACenter)
        self.rpm_label = DirectLabel(parent=self.hud, text="900 RPM", scale=0.030, pos=(0, 0, -0.085), text_align=TextNode.ACenter)
        self.rpm_bar = DirectWaitBar(parent=self.hud, range=8000, value=900, scale=0.22,
                                     pos=(0, 0, -0.135), barColor=(0.12, 0.85, 0.20, 1),
                                     frameColor=(0.06, 0.06, 0.07, 0.9), frameSize=(-1.0, 1.0, -0.10, 0.10))
        self.rpm_scale = DirectLabel(parent=self.hud, text="1   2   3   4   5   6   7   8",
                                     scale=0.019, pos=(0, 0, -0.153), text_align=TextNode.ACenter)
        self.clutch_label = DirectLabel(text="EMBREAGEM: SOLTA", scale=0.038, pos=(-1.25, 0, 0.46), text_align=TextNode.ALeft)
        self.nitro_label = DirectLabel(text="NITRO 100%", scale=0.045, pos=(-1.25, 0, 0.62), text_align=TextNode.ALeft)
        self.credit_label = DirectLabel(text="CR 5000", scale=0.045, pos=(-1.25, 0, 0.54), text_align=TextNode.ALeft)
        self.camera_label = DirectLabel(text="CAM 1ª", scale=0.045, pos=(1.25, 0, 0.9), text_align=TextNode.ARight)
        self.race_status_label = DirectLabel(text="MUNDO LIVRE", scale=0.038, pos=(0, 0, 0.91),
                                             text_align=TextNode.ACenter, text_fg=(0.94, 0.96, 1, 1))
        self.message_label = DirectLabel(text="", scale=0.05, pos=(0, 0, 0.72), text_align=TextNode.ACenter)
        self.cam_button = DirectButton(text="CAMERA", scale=0.045, pos=(1.08, 0, -0.86), command=self.cycle_camera)
        self.save_button = DirectButton(text="SALVAR", scale=0.045, pos=(0.76, 0, -0.86), command=self.save_game)
        self.custom_button = DirectButton(text="CUSTOMIZAR", scale=0.045, pos=(0.34, 0, -0.86), command=self.toggle_customizer)

        self.custom_panel = DirectFrame(frameColor=(0.03,0.03,0.04,0.94), frameSize=(-0.58,0.58,-0.48,0.48), pos=(0,0,0))
        DirectLabel(parent=self.custom_panel, text="CUSTOMIZAÇÃO DO VEÍCULO", scale=0.06, pos=(0,0,0.36))
        self.color_name_label = DirectLabel(parent=self.custom_panel, text="COR", scale=0.05, pos=(0,0,0.20))
        DirectButton(parent=self.custom_panel, text="< COR", scale=0.045, pos=(-0.32,0,0.08), command=self.prev_color)
        DirectButton(parent=self.custom_panel, text="COR >", scale=0.045, pos=(0.32,0,0.08), command=self.next_color)
        self.wheel_name_label = DirectLabel(parent=self.custom_panel, text="RODA: Sport", scale=0.05, pos=(0,0,-0.08))
        DirectButton(parent=self.custom_panel, text="< RODA", scale=0.045, pos=(-0.32,0,-0.20), command=self.prev_wheel)
        DirectButton(parent=self.custom_panel, text="RODA >", scale=0.045, pos=(0.32,0,-0.20), command=self.next_wheel)
        DirectButton(parent=self.custom_panel, text="FECHAR", scale=0.05, pos=(0,0,-0.36), command=self.toggle_customizer)
        self.custom_panel.hide()

        self.minimap = DirectFrame(frameColor=(0.03, 0.03, 0.03, 0.76), frameSize=(-0.22, 0.22, -0.22, 0.22), pos=(1.05, 0, 0.52))
        self.map_player = DirectLabel(parent=self.minimap, text="▲", scale=0.06, pos=(0, 0, 0), text_fg=(1, 1, 0, 1))
        self.map_traffic = []
        for _ in range(12):
            lbl = DirectLabel(parent=self.minimap, text="•", scale=0.04, text_fg=(1, 0.2, 0.2, 1))
            self.map_traffic.append(lbl)


    def setup_vehicle_menu(self):
        # Menu principal com duas etapas: veículo + transmissão.
        self.vehicle_menu = DirectFrame(
            frameColor=(0.018, 0.024, 0.038, 0.98),
            frameSize=(-1.12, 1.12, -0.86, 0.82), pos=(0, 0, 0)
        )
        DirectFrame(parent=self.vehicle_menu, frameColor=(0.04,0.10,0.18,1), frameSize=(-1.12,1.12,-0.12,0.12), pos=(0,0,0.66))
        DirectLabel(parent=self.vehicle_menu, text="OPEN HORIZON", scale=0.092, pos=(0,0,0.68),
                    text_align=TextNode.ACenter, text_fg=(0.92,0.96,1,1))
        DirectLabel(parent=self.vehicle_menu, text="CONFIGURE SUA PILOTAGEM", scale=0.038, pos=(0,0,0.55),
                    text_align=TextNode.ACenter, text_fg=(0.55,0.75,1,1))

        # Painel de veículo
        DirectFrame(parent=self.vehicle_menu, frameColor=(0.035,0.045,0.065,1), frameSize=(-0.98,-0.04,-0.18,0.39), pos=(0,0,0))
        DirectLabel(parent=self.vehicle_menu, text="1  VEÍCULO", scale=0.05, pos=(-0.51,0,0.32), text_align=TextNode.ACenter)
        self.car_btn = DirectButton(parent=self.vehicle_menu, text="CARRO\nSHIFT: NITRO", scale=0.054, pos=(-0.73,0,0.08),
                                    frameSize=(-3.0,3.0,-1.45,1.45), command=self.select_menu_vehicle, extraArgs=["car"])
        self.moto_btn = DirectButton(parent=self.vehicle_menu, text="MOTO\nSHIFT: GRAU", scale=0.054, pos=(-0.29,0,0.08),
                                     frameSize=(-3.0,3.0,-1.45,1.45), command=self.select_menu_vehicle, extraArgs=["moto"])
        self.vehicle_choice_label = DirectLabel(parent=self.vehicle_menu, text="Selecionado: CARRO", scale=0.038,
                                                pos=(-0.51,0,-0.105), text_align=TextNode.ACenter, text_fg=(0.55,1,0.65,1))

        # Painel de transmissão
        DirectFrame(parent=self.vehicle_menu, frameColor=(0.035,0.045,0.065,1), frameSize=(0.04,0.98,-0.18,0.39), pos=(0,0,0))
        DirectLabel(parent=self.vehicle_menu, text="2  CÂMBIO", scale=0.05, pos=(0.51,0,0.32), text_align=TextNode.ACenter)
        self.auto_btn = DirectButton(parent=self.vehicle_menu, text="AUTOMÁTICO\nTroca sozinho", scale=0.050, pos=(0.29,0,0.08),
                                     frameSize=(-3.2,3.2,-1.55,1.55), command=self.select_transmission, extraArgs=["automatic"])
        self.manual_btn = DirectButton(parent=self.vehicle_menu, text="MANUAL\nTAB + Q / E", scale=0.050, pos=(0.73,0,0.08),
                                       frameSize=(-3.2,3.2,-1.55,1.55), command=self.select_transmission, extraArgs=["manual"])
        self.trans_choice_label = DirectLabel(parent=self.vehicle_menu, text="Selecionado: MANUAL", scale=0.038,
                                              pos=(0.51,0,-0.105), text_align=TextNode.ACenter, text_fg=(0.55,1,0.65,1))

        # Painel de modo de jogo
        DirectFrame(parent=self.vehicle_menu, frameColor=(0.035,0.045,0.065,1),
                    frameSize=(-0.98,0.98,-0.13,0.13), pos=(0,0,-0.30))
        DirectLabel(parent=self.vehicle_menu, text="3  MODO DE JOGO", scale=0.040,
                    pos=(0,0,-0.205), text_align=TextNode.ACenter, text_fg=(0.88,0.92,1,1))
        self.free_btn = DirectButton(parent=self.vehicle_menu, text="MUNDO LIVRE\nExplore o mapa", scale=0.044,
                                     pos=(-0.43,0,-0.36), frameSize=(-3.3,3.3,-1.25,1.25),
                                     command=self.select_game_mode, extraArgs=["free"])
        self.race_btn = DirectButton(parent=self.vehicle_menu, text="CORRIDA\nSprint + rivais", scale=0.044,
                                     pos=(0.43,0,-0.36), frameSize=(-3.3,3.3,-1.25,1.25),
                                     command=self.select_game_mode, extraArgs=["race"])
        self.game_mode_choice_label = DirectLabel(parent=self.vehicle_menu, text="Selecionado: MUNDO LIVRE",
                                                  scale=0.030, pos=(0,0,-0.455), text_align=TextNode.ACenter,
                                                  text_fg=(0.55,1,0.65,1))

        DirectLabel(parent=self.vehicle_menu, text="WASD dirigir  •  C câmera  •  ESPAÇO freio de mão  •  V abre este menu",
                    scale=0.030, pos=(0,0,-0.535), text_align=TextNode.ACenter, text_fg=(0.72,0.76,0.82,1))
        self.play_button = DirectButton(parent=self.vehicle_menu, text="JOGAR", scale=0.070, pos=(0,0,-0.665),
                                        frameSize=(-3.6,3.6,-0.85,0.85), command=self.start_from_menu)
        DirectLabel(parent=self.vehicle_menu, text="Escolha veículo, câmbio e modo para iniciar", scale=0.028,
                    pos=(0,0,-0.775), text_align=TextNode.ACenter, text_fg=(0.55,0.60,0.68,1))
        self.refresh_menu_selection()

    def refresh_menu_selection(self):
        if not hasattr(self, "vehicle_choice_label"):
            return
        self.vehicle_choice_label["text"] = "Selecionado: " + ("MOTO" if self.menu_vehicle_choice == "moto" else "CARRO")
        self.trans_choice_label["text"] = "Selecionado: " + ("AUTOMÁTICO" if self.menu_transmission_choice == "automatic" else "MANUAL")
        self.game_mode_choice_label["text"] = "Selecionado: " + ("CORRIDA" if self.menu_game_mode_choice == "race" else "MUNDO LIVRE")
        selected = (0.15, 0.42, 0.72, 1)
        normal = (0.12, 0.14, 0.18, 1)
        self.car_btn["frameColor"] = selected if self.menu_vehicle_choice == "car" else normal
        self.moto_btn["frameColor"] = selected if self.menu_vehicle_choice == "moto" else normal
        self.auto_btn["frameColor"] = selected if self.menu_transmission_choice == "automatic" else normal
        self.manual_btn["frameColor"] = selected if self.menu_transmission_choice == "manual" else normal
        self.free_btn["frameColor"] = selected if self.menu_game_mode_choice == "free" else normal
        self.race_btn["frameColor"] = selected if self.menu_game_mode_choice == "race" else normal

    def select_menu_vehicle(self, vehicle_type):
        self.menu_vehicle_choice = vehicle_type
        self.refresh_menu_selection()

    def select_transmission(self, mode):
        self.menu_transmission_choice = mode
        self.refresh_menu_selection()

    def select_game_mode(self, mode):
        self.menu_game_mode_choice = mode
        self.refresh_menu_selection()

    def start_from_menu(self):
        self.transmission_mode = self.menu_transmission_choice
        self.choose_vehicle(self.menu_vehicle_choice, close_menu=False)
        self.game_mode = self.menu_game_mode_choice
        self.vehicle_selected = True
        self.vehicle_menu.hide()
        if self.transmission_mode == "automatic":
            if self.current_gear <= 0:
                self.current_gear = 1
            self.message_label["text"] = "CÂMBIO AUTOMÁTICO"
        else:
            self.message_label["text"] = "CÂMBIO MANUAL: TAB + Q/E"
        if self.game_mode == "race":
            self.start_race()
        else:
            self.stop_race(restore_traffic=True)
            self.race_status_label["text"] = "MUNDO LIVRE • EXPLORE O MAPA"
        self.taskMgr.remove("clear-msg")
        self.doMethodLater(1.5, self.clear_message, "clear-msg")

    def choose_vehicle(self, vehicle_type, close_menu=True):
        self.vehicle_type = vehicle_type
        self.menu_vehicle_choice = vehicle_type
        self.vehicle_selected = True
        if close_menu:
            self.vehicle_menu.hide()
        # ajusta dimensões e comportamento físico
        if vehicle_type == "moto":
            self.player.hw = 0.46
            self.player.hl = 1.30
            self.player.radius = math.hypot(self.player.hw, self.player.hl)
            self.player.mass = 240.0
            self.player.inv_mass = 1.0 / self.player.mass
            self.player.inv_inertia = 1.0 / (self.player.mass * ((2*self.player.hl)**2 + (2*self.player.hw)**2) / 12.0)
            self.player.steer_max_deg = 38.0
            self.max_speed = 58.0
            self.accel = 23.0
            self.gear_speed_limits = {1: 12.0, 2: 20.0, 3: 29.0, 4: 38.0, 5: 48.0, 6: 58.0}
            self.gear_torque = {1: 1.38, 2: 1.18, 3: 1.02, 4: 0.90, 5: 0.80, 6: 0.72}
        else:
            self.player.hw = 1.12
            self.player.hl = 2.22
            self.player.radius = math.hypot(self.player.hw, self.player.hl)
            self.player.mass = 1400.0
            self.player.inv_mass = 1.0 / self.player.mass
            self.player.inv_inertia = 1.0 / (self.player.mass * ((2*self.player.hl)**2 + (2*self.player.hw)**2) / 12.0)
            self.player.steer_max_deg = 34.0
            self.max_speed = 52.0
            self.accel = 20.0
            self.gear_speed_limits = {1: 11.0, 2: 19.0, 3: 27.0, 4: 33.5, 5: 37.5, 6: 48.0}
            self.gear_torque = {1: 1.34, 2: 1.16, 3: 1.00, 4: 0.90, 5: 0.82, 6: 0.74}
        self.rebuild_player_visual()
        self.refresh_menu_selection()
        if close_menu:
            self.message_label["text"] = "MOTO SELECIONADA" if vehicle_type == "moto" else "CARRO SELECIONADO"
            self.doMethodLater(1.2, self.clear_message, "clear-msg")

    def toggle_vehicle_menu(self):
        if self.vehicle_menu.isHidden():
            self.menu_vehicle_choice = self.vehicle_type
            self.menu_transmission_choice = self.transmission_mode
            self.menu_game_mode_choice = self.game_mode
            self.refresh_menu_selection()
            self.vehicle_menu.show()
            self.vehicle_selected = False
        else:
            self.vehicle_menu.hide()
            self.vehicle_selected = True

    # ---------- race mode ----------
    def _race_gate(self, y, title, color):
        """Cria um portal visual atravessando a estrada, sem bloquear o veículo."""
        nodes = []
        for side in (-1, 1):
            post = self.box(f"race_{title}_post", (side * 5.35, y, 2.55),
                            (0.14, 0.16, 2.55), color, solid=False)
            nodes.append(post)
        beam = self.box(f"race_{title}_beam", (0, y, 5.12), (5.45, 0.16, 0.16), color, solid=False)
        stripe = self.box(f"race_{title}_stripe", (0, y, 0.17), (5.85, 0.20, 0.035),
                          (0.96, 0.86, 0.18, 1), solid=False)
        nodes.extend([beam, stripe])
        return nodes

    def _clear_race_visuals(self):
        for node in self.race_scene_nodes:
            if not node.isEmpty():
                node.removeNode()
        self.race_scene_nodes = []
        for rival in self.race_rivals:
            root = rival.get("root")
            if root is not None and not root.isEmpty():
                root.removeNode()
        self.race_rivals = []

    def stop_race(self, restore_traffic=True):
        self._clear_race_visuals()
        self.race_state = "inactive"
        self.race_timer = 0.0
        self.race_countdown = 0.0
        self.race_checkpoint_index = 0
        self.race_awarded = False
        if restore_traffic and self.race_traffic_backup is not None:
            self.world.traffic, self.world.vehicles = self.race_traffic_backup
            self.race_traffic_backup = None
            for vehicle in self.world.traffic:
                if vehicle.node is not None:
                    vehicle.node.show()
        if hasattr(self, "race_status_label"):
            self.race_status_label["text"] = "MUNDO LIVRE" if self.game_mode == "free" else "CORRIDA"

    def start_race(self):
        """Inicia um sprint de 1,47 km com dois rivais, checkpoints e recompensa."""
        self.stop_race(restore_traffic=True)
        if self.race_traffic_backup is None:
            self.race_traffic_backup = (list(self.world.traffic), list(self.world.vehicles))
        # Retira o trânsito normal temporariamente para montar uma largada limpa.
        for vehicle in self.world.traffic:
            if vehicle.node is not None:
                vehicle.node.hide()
        self.world.traffic = []
        self.world.vehicles = [self.player]

        self.race_start_y = -735.0
        self.race_checkpoints = [-300.0, 100.0, 500.0]
        self.race_finish_y = 740.0
        self.race_checkpoint_index = 0
        self.race_timer = 0.0
        self.race_countdown = 3.0
        self.race_state = "countdown"
        self.race_position = 1
        self.race_awarded = False

        # Largada em três faixas paralelas da pista principal.
        start_grid = [(LANE_X, self.race_start_y, None), (0.0, self.race_start_y - 1.0, (0.88, 0.12, 0.08, 1)),
                      (-LANE_X, self.race_start_y - 2.0, (0.10, 0.35, 0.90, 1))]
        self.car.setPos(LANE_X, self.race_start_y, 0.18)
        self.heading = 0.0
        self.cam_heading = 0.0
        self.player.x, self.player.y = LANE_X, self.race_start_y
        self.player.heading = 0.0
        self.player.vx = self.player.vy = self.player.yaw = 0.0
        self.player.drive = self.player.brake = self.player.steer = self.player.hb = 0.0
        self.player.refresh_trig()
        self.speed = 0.0
        self.current_gear = 1
        self.rpm = 900.0
        self.nitro = 100.0

        self.race_scene_nodes.extend(self._race_gate(self.race_start_y + 12, "START", (0.10, 0.75, 0.30, 1)))
        for index, checkpoint_y in enumerate(self.race_checkpoints, 1):
            self.race_scene_nodes.extend(self._race_gate(checkpoint_y, f"CP{index}", (0.12, 0.58, 0.95, 1)))
        self.race_scene_nodes.extend(self._race_gate(self.race_finish_y, "FINISH", (0.95, 0.18, 0.10, 1)))

        # Rivais são carros 3D animados no percurso; não interferem na física do jogador.
        for index, (x, y, color) in enumerate(start_grid[1:]):
            root = self.render.attachNewNode(f"race_rival_{index+1}")
            self.create_car_visual(root, color, 0.96, False)
            root.setPos(x, y, 0.18)
            root.setH(0)
            self.race_rivals.append({
                "root": root, "x": x, "y": y,
                "target_speed": (32.0, 35.5)[index],
                "speed": 0.0,
                "finish_time": None,
            })

        self.race_status_label["text"] = "CORRIDA • PREPARE-SE"
        self.message_label["text"] = "CORRIDA: 1.475 m • 3 CHECKPOINTS"
        self.taskMgr.remove("clear-msg")
        self.doMethodLater(2.0, self.clear_message, "clear-msg")

    def update_race(self, dt):
        if self.race_state == "countdown":
            self.race_countdown -= dt
            count = max(1, int(math.ceil(self.race_countdown)))
            self.race_status_label["text"] = f"LARGADA EM {count}..."
            if self.race_countdown <= 0.0:
                self.race_state = "racing"
                self.race_timer = 0.0
                self.race_status_label["text"] = "VALENDO!"
                self.message_label["text"] = "VAI!"
                self.taskMgr.remove("clear-msg")
                self.doMethodLater(0.8, self.clear_message, "clear-msg")
            return

        if self.race_state != "racing":
            return

        self.race_timer += dt
        for rival in self.race_rivals:
            # Saída progressiva para os rivais não arrancarem instantaneamente.
            rival["speed"] = min(rival["target_speed"], rival["speed"] + 11.0 * dt)
            rival["y"] += rival["speed"] * dt
            rival["root"].setPos(rival["x"], rival["y"], 0.18)
            if rival["finish_time"] is None and rival["y"] >= self.race_finish_y:
                rival["finish_time"] = self.race_timer

        player_y = self.car.getY()
        while (self.race_checkpoint_index < len(self.race_checkpoints)
               and player_y >= self.race_checkpoints[self.race_checkpoint_index]):
            self.race_checkpoint_index += 1
            self.message_label["text"] = f"CHECKPOINT {self.race_checkpoint_index}/{len(self.race_checkpoints)}"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.9, self.clear_message, "clear-msg")

        self.race_position = 1 + sum(1 for rival in self.race_rivals if rival["y"] > player_y)
        cp_text = f"CP {self.race_checkpoint_index}/{len(self.race_checkpoints)}"
        time_text = self._format_race_time(self.race_timer)
        self.race_status_label["text"] = f"CORRIDA  •  {self.race_position}º/3  •  {cp_text}  •  {time_text}"

        if player_y >= self.race_finish_y:
            self.race_state = "finished"
            self.race_position = 1 + sum(1 for rival in self.race_rivals
                                         if rival["finish_time"] is not None and rival["finish_time"] < self.race_timer)
            rewards = {1: 1500, 2: 750, 3: 350}
            reward = rewards.get(self.race_position, 350)
            self.credits += reward
            self.race_awarded = True
            self.race_status_label["text"] = (f"CORRIDA FINALIZADA  •  {self.race_position}º LUGAR  •  "
                                               f"{self._format_race_time(self.race_timer)}")
            self.message_label["text"] = f"{self.race_position}º LUGAR! +{reward} CR"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(4.0, self.clear_message, "clear-msg")

    @staticmethod
    def _format_race_time(seconds):
        total = max(0.0, float(seconds))
        minutes = int(total // 60)
        secs = total - minutes * 60
        return f"{minutes:02d}:{secs:05.2f}"

    def bind_inputs(self):
        for key in ["w", "s", "a", "d", "shift", "space", "tab"]:
            self.accept(key, self.set_key, [key, True])
            self.accept(key + "-up", self.set_key, [key, False])
        self.accept("arrow_up", self.set_key, ["w", True]); self.accept("arrow_up-up", self.set_key, ["w", False])
        self.accept("arrow_down", self.set_key, ["s", True]); self.accept("arrow_down-up", self.set_key, ["s", False])
        self.accept("arrow_left", self.set_key, ["a", True]); self.accept("arrow_left-up", self.set_key, ["a", False])
        self.accept("arrow_right", self.set_key, ["d", True]); self.accept("arrow_right-up", self.set_key, ["d", False])
        self.accept("q", self.shift_down)
        self.accept("e", self.shift_up)
        self.accept("c", self.cycle_camera)
        self.accept("m", self.toggle_customizer)
        self.accept("v", self.toggle_vehicle_menu)
        self.accept("f5", self.save_game)
        self.accept("escape", self.userExit)

    def set_key(self, key, value):
        self.keys[key] = value

    def gear_text(self):
        if self.current_gear < 0:
            return "R"
        if self.current_gear == 0:
            return "N"
        return str(self.current_gear)

    def shift_up(self):
        if self.transmission_mode != "manual":
            self.message_label["text"] = "CÂMBIO AUTOMÁTICO ATIVO"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.7, self.clear_message, "clear-msg")
            return
        if not self.vehicle_selected or self.shift_cooldown > 0.0:
            return
        if not self.keys.get("tab", False):
            self.message_label["text"] = "APERTE TAB PARA USAR A EMBREAGEM"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.8, self.clear_message, "clear-msg")
            return
        if self.current_gear < self.max_gear:
            self.current_gear += 1
            self.shift_cooldown = 0.22
            self.message_label["text"] = f"MARCHA {self.gear_text()}"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.65, self.clear_message, "clear-msg")

    def shift_down(self):
        if self.transmission_mode != "manual":
            self.message_label["text"] = "CÂMBIO AUTOMÁTICO ATIVO"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.7, self.clear_message, "clear-msg")
            return
        if not self.vehicle_selected or self.shift_cooldown > 0.0:
            return
        if not self.keys.get("tab", False):
            self.message_label["text"] = "APERTE TAB PARA USAR A EMBREAGEM"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.8, self.clear_message, "clear-msg")
            return
        # Só permite engatar ré quando o veículo está praticamente parado.
        if self.current_gear == 0 and self.player.forward_speed() > 1.5:
            self.message_label["text"] = "PARE PARA ENGATAR A RÉ"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.8, self.clear_message, "clear-msg")
            return
        if self.current_gear > -1:
            self.current_gear -= 1
            self.shift_cooldown = 0.22
            self.message_label["text"] = f"MARCHA {self.gear_text()}"
            self.taskMgr.remove("clear-msg")
            self.doMethodLater(0.65, self.clear_message, "clear-msg")

    def update_automatic_transmission(self, vf, dt):
        if self.transmission_mode != "automatic" or self.shift_cooldown > 0.0:
            return
        speed = abs(vf)
        # No automático: W seleciona direção para frente e S seleciona ré quando parado.
        if self.keys["s"] and speed < 0.8 and not self.keys["w"]:
            if self.current_gear != -1:
                self.current_gear = -1
                self.shift_cooldown = 0.16
            return
        if self.keys["w"] and self.current_gear <= 0:
            self.current_gear = 1
            self.shift_cooldown = 0.16
            return
        if self.current_gear <= 0:
            return
        limit = self.gear_speed_limits[self.current_gear]
        # Sobe quando chega perto do fim útil da marcha; reduz para recuperar torque.
        if self.keys["w"] and self.current_gear < self.max_gear and (speed > limit * 0.985 or self.rpm > 7350):
            self.current_gear += 1
            self.shift_cooldown = 0.18
        elif self.current_gear > 1:
            prev_limit = self.gear_speed_limits[self.current_gear - 1]
            if speed < prev_limit * 0.62 and self.rpm < 2600:
                self.current_gear -= 1
                self.shift_cooldown = 0.16

    def update_engine_rpm(self, vf, dt):
        throttle = self.keys["w"]
        if self.current_gear < 0:
            ratio = min(1.05, abs(vf) / 13.0)
            target = 900.0 + ratio * 5800.0
        elif self.current_gear == 0:
            target = 900.0 + (2500.0 if throttle else 0.0)
        else:
            limit = self.gear_speed_limits[self.current_gear]
            ratio = min(1.08, abs(vf) / max(limit, 0.1))
            target = 900.0 + ratio * 6500.0
            if throttle and abs(vf) < 1.5:
                target = max(target, 1500.0)
        self.rpm += (target - self.rpm) * min(1.0, 8.0 * dt)

    def cycle_camera(self):
        self.camera_mode = (self.camera_mode + 1) % 3
        self.camera_label["text"] = f"CAM {self.camera_mode + 1}ª"

    def toggle_customizer(self):
        self.customizer_open = not self.customizer_open
        if self.customizer_open:
            self.custom_panel.show()
            self.player.vx *= 0.2
            self.player.vy *= 0.2
        else:
            self.custom_panel.hide()
        self.refresh_custom_labels()

    def refresh_custom_labels(self):
        color_names = ["Azul", "Vermelho", "Verde", "Amarelo", "Preto", "Prata", "Roxo"]
        self.color_name_label["text"] = f"COR: {color_names[self.car_color_index]}"
        self.wheel_name_label["text"] = f"RODA: {self.wheel_styles[self.wheel_style_index]}"

    def rebuild_player_visual(self):
        if hasattr(self, "player_visual_root"):
            self.player_visual_root.removeNode()
        self.player_visual_root = self.car.attachNewNode("visual_root")
        if self.vehicle_type == "moto":
            self.player_wheels = self.create_motorcycle_visual(self.player_visual_root, self.car_colors[self.car_color_index], 1.0, True)
        else:
            self.player_wheels = self.create_car_visual(self.player_visual_root, self.car_colors[self.car_color_index], 1.0, True)
        self.player.wheels = self.player_wheels
        # rim styling by tinting rim-like cylinders based on wheel choice
        if self.wheel_style_index == 1:
            for w in self.player_wheels: w.setScale(1, 1.08, 1.08)
        elif self.wheel_style_index == 2:
            for w in self.player_wheels: w.setColor(0.01,0.01,0.012,1)

    def next_color(self):
        self.car_color_index = (self.car_color_index + 1) % len(self.car_colors)
        self.rebuild_player_visual(); self.refresh_custom_labels()

    def prev_color(self):
        self.car_color_index = (self.car_color_index - 1) % len(self.car_colors)
        self.rebuild_player_visual(); self.refresh_custom_labels()

    def next_wheel(self):
        self.wheel_style_index = (self.wheel_style_index + 1) % len(self.wheel_styles)
        self.rebuild_player_visual(); self.refresh_custom_labels()

    def prev_wheel(self):
        self.wheel_style_index = (self.wheel_style_index - 1) % len(self.wheel_styles)
        self.rebuild_player_visual(); self.refresh_custom_labels()

    # ---------- update ----------
    def update(self, task):
        dt = min(globalClock.getDt(), 0.05)
        p = self.player
        vf = p.forward_speed()

        if not self.vehicle_selected:
            p.drive = p.brake = p.steer = 0.0
            self.update_camera(dt)
            self.update_hud()
            return Task.cont

        if self.game_mode == "race" and self.race_state == "countdown":
            self.update_race(dt)
            p.drive = p.brake = p.steer = p.hb = 0.0
            self.sync_vehicles(dt)
            self.update_camera(dt)
            self.update_hud()
            self.update_minimap()
            return Task.cont

        # --- entradas -> comandos do veículo + câmbio manual
        self.shift_cooldown = max(0.0, self.shift_cooldown - dt)
        self.update_automatic_transmission(vf, dt)
        self.update_engine_rpm(vf, dt)

        # No carro SHIFT ativa nitro; na moto SHIFT controla o grau.
        # Ambos respeitam a marcha manual selecionada.
        self.nitro_active = (self.vehicle_type == "car" and self.keys["shift"] and self.keys["w"]
                             and self.nitro > 0 and vf > 5 and self.current_gear > 0
                             and self.shift_cooldown <= 0.0)
        wheelie_target = 1.0 if (self.vehicle_type == "moto" and self.keys["shift"] and self.keys["w"]
                                 and vf > 4.0 and self.current_gear > 0) else 0.0
        self.wheelie += (wheelie_target - self.wheelie) * min(1.0, (4.8 if wheelie_target else 3.2) * dt)

        drive = brake = 0.0
        if self.current_gear > 0:
            gear_limit = self.gear_speed_limits[self.current_gear]
            cmax = gear_limit * (1.18 if self.nitro_active else 1.0)
            torque = self.gear_torque[self.current_gear]
            if self.keys["w"]:
                if vf < -0.5:
                    brake = self.brake
                elif self.shift_cooldown <= 0.0:
                    ratio = max(vf, 0.0) / max(cmax, 0.1)
                    fall = max(0.0, 1.0 - ratio ** 4)
                    drive = self.accel * torque * fall
                    if self.nitro_active:
                        drive += 22.0 * fall
            if self.keys["s"]:
                brake = max(brake, self.brake)

        elif self.current_gear == 0:
            # Neutro: W sobe o giro, mas não transmite força. S continua freando.
            if self.keys["s"]:
                brake = self.brake

        else:  # Ré
            reverse_pressed = self.keys["s"] if self.transmission_mode == "automatic" else self.keys["w"]
            brake_pressed = self.keys["w"] if self.transmission_mode == "automatic" else self.keys["s"]
            if reverse_pressed:
                if vf > 0.5:
                    brake = self.brake
                elif self.shift_cooldown <= 0.0:
                    drive = -self.reverse_accel * max(0.0, 1.0 - (-vf / 13.0) ** 2)
            if brake_pressed:
                brake = self.brake

        if self.nitro_active:
            self.nitro = max(0, self.nitro - 20 * dt)
        else:
            self.nitro = min(100, self.nitro + 4.5 * dt)

        # freio de mao: entra/sai suave, solta a traseira e freia um pouco
        hb_target = 1.0 if (self.keys["space"] and abs(vf) > 8) else 0.0
        self.hb += (hb_target - self.hb) * min(1.0, (7.0 if hb_target else 3.5) * dt)
        brake = max(brake, 5.0 * self.hb)

        target_steer = (-1.0 if self.keys["a"] else 0.0) + (1.0 if self.keys["d"] else 0.0)
        if self.vehicle_type == "moto":
            target_steer *= (1.0 - 0.72 * self.wheelie)
        response = 5.8 if target_steer != 0 else 8.5
        self.steer_input += (target_steer - self.steer_input) * min(1.0, response * dt)

        p.drive, p.brake, p.steer, p.hb = drive, brake, self.steer_input, self.hb

        # --- fisica em passos curtos (evita atravessar carros em alta velocidade)
        steps = max(1, math.ceil(dt / 0.02))
        h = dt / steps
        events = []
        for _ in range(steps):
            events += self.world.step(h)

        self.speed = p.forward_speed()
        self.heading = p.heading
        crash = max((imp for a, b, imp in events if (a is p or b is p)), default=0.0)
        if crash > 3.0:
            self.on_player_crash(crash)

        self.sync_vehicles(dt)
        self.handle_radars(dt)
        if self.game_mode == "race":
            self.update_race(dt)
        self.update_camera(dt)
        self.update_hud()
        self.update_minimap()
        return Task.cont

    def on_player_crash(self, impact):
        self.message_label["text"] = "COLISÃO!"
        self.taskMgr.remove("clear-msg")
        self.doMethodLater(1.2, self.clear_message, "clear-msg")
        self.shake = max(self.shake, min(1.0, impact / 16.0))

    def sync_vehicles(self, dt):
        """Copia o estado da fisica para os nos do Panda (posicao, giro, rodas, inclinacao)."""
        p = self.player
        offroad = p.surface is SURF_GRASS and p.speed() > 4
        for v in self.world.vehicles:
            if v is not p and abs(v.x - p.x) + abs(v.y - p.y) > 260:
                v.node.setPos(v.x, v.y, 0.18)
                v.node.setH(v.heading)
                continue
            vf = v.forward_speed()
            z = 0.18
            if v is p:
                bump = min(1.0, p.speed() / 25.0) if offroad else 0.0
                tp = max(-3.5, min(3.5, v.acc_long * 0.10)) + random.uniform(-0.9, 0.9) * bump
                tr = max(-4.5, min(4.5, v.lean * 0.25)) + random.uniform(-0.9, 0.9) * bump
                k = min(1.0, 10.0 * dt)
                self.tilt_p += (tp - self.tilt_p) * k
                self.tilt_r += (tr - self.tilt_r) * k
                if self.vehicle_type == "moto":
                    # grau visual com limite; retorna suavemente ao chão ao soltar SHIFT
                    self.wheelie_angle += ((-28.0 * self.wheelie) - self.wheelie_angle) * min(1.0, 7.0 * dt)
                    self.player_visual_root.setP(self.tilt_p + self.wheelie_angle)
                    # moto inclina nas curvas
                    self.player_visual_root.setR(max(-18.0, min(18.0, self.tilt_r * 2.2)))
                else:
                    self.player_visual_root.setP(self.tilt_p)
                    self.player_visual_root.setR(self.tilt_r)
                z += random.uniform(-0.03, 0.03) * bump
            v.node.setPos(v.x, v.y, z)
            v.node.setH(v.heading)
            spin = -vf * dt * 143.0 / v.scale
            steer_deg = -math.degrees(v.steer_angle)
            for i, w in enumerate(v.wheels):
                w.setP(w.getP() + spin)
                if (v is p and self.vehicle_type == "moto" and i == 1) or (not (v is p and self.vehicle_type == "moto") and i in (1, 3)):
                    w.setH(steer_deg)

    def handle_radars(self, dt):
        self.radar_cooldown = max(0, self.radar_cooldown - dt)
        if self.radar_cooldown > 0:
            return
        p = self.car.getPos()
        for pos, limit in self.radar_nodes:
            if (p - pos).length() < 8 and abs(self.speed) * 3.6 > limit:
                fine = 250
                self.credits = max(0, self.credits - fine)
                self.radar_cooldown = 4.0
                self.message_label["text"] = f"RADAR! -{fine} CR"
                self.doMethodLater(1.8, self.clear_message, "clear-msg")
                break

    def clear_message(self, task):
        self.message_label["text"] = ""
        return Task.done

    def update_camera(self, dt=0.016):
        p = self.car.getPos()
        h = math.radians(self.heading)
        forward = Vec3(-math.sin(h), math.cos(h), 0)
        if self.camera_mode == 0:  # cockpit
            cam_pos = p + forward * 0.52 + Vec3(0, 0, 1.42)
            look_at = p + forward * 24 + Vec3(0, 0, 1.05)
        elif self.camera_mode == 1:  # bonnet
            cam_pos = p + forward * 1.78 + Vec3(0, 0, 1.48)
            look_at = p + forward * 26 + Vec3(0, 0, 0.85)
        else:  # chase: segue o carro com um pouco de atraso (da sensacao de peso)
            self.cam_heading += wrap_deg(self.heading - self.cam_heading) * min(1.0, 5.0 * dt)
            ch = math.radians(self.cam_heading)
            cf = Vec3(-math.sin(ch), math.cos(ch), 0)
            cam_pos = p - cf * 8.7 + Vec3(0, 0, 4.1)
            look_at = p + cf * 5.5 + Vec3(0, 0, 0.95)
        if self.shake > 0.01:
            cam_pos += Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)) * 0.22 * self.shake
            self.shake *= math.exp(-6.0 * dt)
        self.camera.setPos(cam_pos)
        self.camera.lookAt(look_at)
        # campo de visao abre com a velocidade
        extra = min(14.0, max(0.0, abs(self.speed) - 22.0) / 45.0 * 14.0) + (4.0 if self.nitro_active else 0.0)
        self.camLens.setFov(self.base_fov + extra)

    def update_hud(self):
        kmh = int(abs(self.speed) * 3.6)
        self.speed_label["text"] = f"{kmh} km/h"
        self.gear_label["text"] = f"MARCHA {self.gear_text()}"
        rpm_int = int(max(800, min(7900, self.rpm)))
        self.rpm_label["text"] = f"{rpm_int} RPM" + ("  CORTE" if rpm_int >= 7400 else "")
        self.rpm_bar["value"] = rpm_int
        if rpm_int >= 7000:
            self.rpm_bar["barColor"] = (0.95, 0.12, 0.08, 1)
        elif rpm_int >= 5500:
            self.rpm_bar["barColor"] = (0.95, 0.62, 0.08, 1)
        else:
            self.rpm_bar["barColor"] = (0.12, 0.85, 0.20, 1)
        if self.vehicle_type == "moto":
            self.nitro_label["text"] = f"GRAU SHIFT  {int(self.wheelie*100)}%"
        else:
            self.nitro_label["text"] = f"NITRO {int(self.nitro)}%" + ("  BOOST" if self.nitro_active else "")
        self.credit_label["text"] = f"CR {self.credits}"
        if self.game_mode == "free" and self.race_state == "inactive":
            self.race_status_label["text"] = "MUNDO LIVRE • EXPLORE O MAPA"
        if self.transmission_mode == "automatic":
            self.clutch_label["text"] = "CÂMBIO: AUTOMÁTICO"
        else:
            self.clutch_label["text"] = "EMBREAGEM: PRESSIONADA" if self.keys.get("tab", False) else "EMBREAGEM: SOLTA"

    def update_minimap(self):
        p = self.car.getPos()
        mx = max(-0.19, min(0.19, p.x / 430 * 0.19))
        my = max(-0.19, min(0.19, p.y / 800 * 0.19))
        self.map_player.setPos(mx, 0, my)
        self.map_player.setR(-self.heading)
        pp = self.car.getPos()
        near = sorted(self.world.traffic, key=lambda v: (v.x - pp.x) ** 2 + (v.y - pp.y) ** 2)
        for i, lbl in enumerate(self.map_traffic):
            if i >= len(near):
                lbl.hide(); continue
            tx = max(-0.19, min(0.19, near[i].x / 430 * 0.19))
            ty = max(-0.19, min(0.19, near[i].y / 800 * 0.19))
            lbl.setPos(tx, 0, ty)
            lbl.show()

    def save_game(self):
        data = {
            "pos": [self.car.getX(), self.car.getY(), self.car.getZ()],
            "heading": self.heading,
            "credits": self.credits,
            "nitro": self.nitro,
            "camera_mode": self.camera_mode,
            "car_color_index": self.car_color_index,
            "wheel_style_index": self.wheel_style_index,
            "vehicle_type": self.vehicle_type,
            "current_gear": self.current_gear,
            "transmission_mode": self.transmission_mode,
            "game_mode": self.game_mode,
        }
        try:
            with open(SAVE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self.message_label["text"] = "JOGO SALVO"
            self.doMethodLater(1.2, self.clear_message, "clear-msg")
        except Exception as e:
            print("Save error:", e)

    def load_game(self):
        if not os.path.exists(SAVE_FILE):
            return
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            pos = data.get("pos", [LANE_X, -700, 0.18])
            self.car.setPos(*pos)
            self.heading = float(data.get("heading", 0))
            self.cam_heading = self.heading
            self.player.x, self.player.y = pos[0], pos[1]
            self.player.heading = self.heading
            self.player.vx = self.player.vy = self.player.yaw = 0.0
            self.player.refresh_trig()
            self.credits = int(data.get("credits", 5000))
            self.nitro = float(data.get("nitro", 100))
            self.camera_mode = int(data.get("camera_mode", 0)) % 3
            self.car_color_index = int(data.get("car_color_index", 0)) % len(self.car_colors)
            self.wheel_style_index = int(data.get("wheel_style_index", 0)) % len(self.wheel_styles)
            saved_vehicle = data.get("vehicle_type", "car")
            if saved_vehicle in ("car", "moto"):
                self.vehicle_type = saved_vehicle
            self.current_gear = max(-1, min(self.max_gear, int(data.get("current_gear", 1))))
            saved_trans = data.get("transmission_mode", "manual")
            if saved_trans in ("manual", "automatic"):
                self.transmission_mode = saved_trans
                self.menu_transmission_choice = saved_trans
            saved_mode = data.get("game_mode", "free")
            if saved_mode in ("free", "race"):
                self.game_mode = saved_mode
                self.menu_game_mode_choice = saved_mode
            self.menu_vehicle_choice = self.vehicle_type
            self.rebuild_player_visual()
            self.refresh_custom_labels()
            self.camera_label["text"] = f"CAM {self.camera_mode + 1}ª"
            self.refresh_menu_selection()
        except Exception as e:
            print("Load error:", e)


if __name__ == "__main__":
    game = HorizonPanda3D()
    game.run()
