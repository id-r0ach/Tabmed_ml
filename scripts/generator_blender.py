import bpy
import math
import random
from pathlib import Path
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# ==========================================
# 1. Инициализация и настройка OptiX Cycles
# ==========================================
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

scene.render.engine = 'CYCLES'
cycles = scene.cycles
cycles.device = 'GPU'
cycles.samples = 32
cycles.preview_samples = 16
cycles.use_denoising = True

prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'OPTIX'
prefs.get_devices()
for dev in prefs.devices:
    if dev.type == 'OPTIX':
        dev.use = True

# Физический мир с повышенным демпфированием, чтобы таблетки не разлетались
if not scene.rigidbody_world:
    bpy.ops.rigidbody.world_add()
rbw = scene.rigidbody_world
rbw.substeps_per_frame = 25
rbw.solver_iterations = 40

# ==========================================
# 2. Стол (Пассивная физика)
# ==========================================
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.05))
ground = bpy.context.active_object
ground.name = "Table"
ground.scale = (2.0, 2.0, 0.1)
bpy.ops.rigidbody.object_add()
ground.rigid_body.type = 'PASSIVE'
ground.rigid_body.collision_shape = 'BOX'
ground.rigid_body.friction = 0.95

# Материал столешницы (матовый нейтральный графит)
mat_ground = bpy.data.materials.new(name="Table_Mat")
mat_ground.use_nodes = True
bsdf_g = mat_ground.node_tree.nodes.get("Principled BSDF")
bsdf_g.inputs['Base Color'].default_value = (0.15, 0.15, 0.17, 1.0)
bsdf_g.inputs['Roughness'].default_value = 0.8
ground.data.materials.append(mat_ground)

# ==========================================
# 3. Фабрика процедурных материалов
# ==========================================
def get_chalk_material(name, color_rgb):
    """Шейдер матового прессованного мела с легким подповерхностным рассеиванием"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.75
    # Легкий эффект порошка
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.08
    return mat

def get_gelatin_material(name, color_rgb):
    """Шейдер глянцевой желатиновой капсулы с лаковым слоем"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.12
    bsdf.inputs['IOR'].default_value = 1.48
    if 'Coat Weight' in bsdf.inputs:
        bsdf.inputs['Coat Weight'].default_value = 1.0
        bsdf.inputs['Coat Roughness'].default_value = 0.05
    return mat

# Базовые цветовые палитры фармакологии
PALETTE = {
    "white": (0.92, 0.92, 0.92),
    "yellow": (0.95, 0.82, 0.15),
    "blue": (0.05, 0.35, 0.85),
    "red": (0.85, 0.10, 0.10),
    "green": (0.12, 0.65, 0.25),
    "orange": (0.95, 0.45, 0.05)
}

# ==========================================
# 4. Процедурные генераторы геометрии
# ==========================================
def create_round_tablet(name, loc, rot, has_score=True):
    """Круглая таблетка с фаской и процедурной риской (класс 1: tablet)"""
    radius = random.uniform(0.009, 0.012)
    depth = random.uniform(0.004, 0.006)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64, radius=radius, depth=depth, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name

    # Булева насечка (риска по центру)
    if has_score:
        bpy.ops.mesh.primitive_cube_add(
            size=1.0, 
            location=(loc[0], loc[1], loc[2] + depth * 0.48),
            rotation=rot
        )
        cutter = bpy.context.active_object
        cutter.scale = (radius * 2.2, 0.0006, depth * 0.25)

        bool_mod = obj.modifiers.new(name="Score", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = cutter

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Score")
        bpy.data.objects.remove(cutter, do_unlink=True)

    # Скругление фаски
    bevel = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel.width = 0.0009
    bevel.segments = 3
    bpy.ops.object.modifier_apply(modifier="Bevel")
    bpy.ops.object.shade_smooth()

    # Назначаем случайный цвет мела
    color_name = random.choice(["white", "yellow", "green", "orange"])
    obj.data.materials.append(get_chalk_material(f"Mat_{name}", PALETTE[color_name]))
    return obj

def create_oval_tablet(name, loc, rot):
    """Вытянутая овальная таблетка (класс 1: tablet)"""
    obj = create_round_tablet(name, loc, rot, has_score=False)
    # Вытягиваем в эллипс
    obj.scale = (1.55, 1.0, 1.0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    return obj

def create_two_tone_capsule(name, loc, rot):
    """Двухцветная желатиновая капсула со стыком (класс 0: capsule)"""
    radius = random.uniform(0.0045, 0.0055)
    depth = random.uniform(0.016, 0.020)

    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48, radius=radius, depth=depth, location=loc, rotation=rot
    )
    obj = bpy.context.active_object
    obj.name = name

    # Скругляем полусферические торцы
    bevel = obj.modifiers.new(name="Capsule_Bevel", type='BEVEL')
    bevel.width = radius * 0.95
    bevel.segments = 8
    bpy.ops.object.modifier_apply(modifier="Capsule_Bevel")
    bpy.ops.object.shade_smooth()

    # Два цвета (половинки капсулы)
    col1 = random.choice(["blue", "red", "green", "orange"])
    col2 = random.choice(["white", "yellow"])
    mat_top = get_gelatin_material(f"Mat_Top_{name}", PALETTE[col1])
    mat_bottom = get_gelatin_material(f"Mat_Bot_{name}", PALETTE[col2])
    obj.data.materials.append(mat_top)
    obj.data.materials.append(mat_bottom)

    # Назначаем полигонам разные слоты по локальной оси Z (до/после экватора)
    for poly in obj.data.polygons:
        poly.material_index = 0 if poly.center.z >= 0 else 1

    return obj

# ==========================================
# 5. Спавн пачки лекарств (от 3 до 6 штук)
# ==========================================
pills_data = []
spawn_types = ['round', 'round_score', 'oval', 'capsule', 'capsule']
random.shuffle(spawn_types)
num_items = random.randint(3, 5)

for i in range(num_items):
    p_type = spawn_types[i]
    loc = (random.uniform(-0.015, 0.015), random.uniform(-0.015, 0.015), 0.03 + i * 0.018)
    rot = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))

    if p_type == 'round':
        obj = create_round_tablet(f"pill_{i}", loc, rot, has_score=False)
        cls_id = 1
    elif p_type == 'round_score':
        obj = create_round_tablet(f"pill_{i}", loc, rot, has_score=True)
        cls_id = 1
    elif p_type == 'oval':
        obj = create_oval_tablet(f"pill_{i}", loc, rot)
        cls_id = 1
    else:
        obj = create_two_tone_capsule(f"pill_{i}", loc, rot)
        cls_id = 0

    # Настройка активного твердого тела
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE'
    obj.rigid_body.collision_shape = 'CONVEX_HULL'
    obj.rigid_body.mass = 0.003
    obj.rigid_body.friction = 0.8
    obj.rigid_body.linear_damping = 0.65
    obj.rigid_body.angular_damping = 0.65

    pills_data.append({"obj": obj, "class_id": cls_id})

# ==========================================
# 6. Симуляция падения и заморозка
# ==========================================
for frame in range(1, 75):
    scene.frame_set(frame)

depsgraph = bpy.context.evaluated_depsgraph_get()
for item in pills_data:
    obj = item["obj"]
    eval_obj = obj.evaluated_get(depsgraph)
    obj.matrix_world = eval_obj.matrix_world.copy()
    obj.rigid_body.enabled = False

scene.frame_set(1)
bpy.context.view_layer.update()

# ==========================================
# 7. Свет и Камера со случайным ракурсом
# ==========================================
# Мягкий ключевой свет
light_data = bpy.data.lights.new(name="Main_Light", type='POINT')
light_data.energy = random.uniform(50.0, 75.0)
light_data.shadow_soft_size = 0.035
light_obj = bpy.data.objects.new(name="Main_Light", object_data=light_data)
scene.collection.objects.link(light_obj)
light_obj.location = (random.uniform(-0.1, 0.1), random.uniform(-0.15, -0.05), random.uniform(0.28, 0.38))

# Камера
cam_data = bpy.data.cameras.new(name="Camera")
cam_data.lens = 50
cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Случайный наклон и радиус камеры вокруг центра
cam_dist = random.uniform(0.20, 0.26)
cam_angle = random.uniform(math.radians(35), math.radians(65))
cam_z = cam_dist * math.sin(cam_angle)
cam_y = -cam_dist * math.cos(cam_angle)
cam_x = random.uniform(-0.04, 0.04)
cam_obj.location = (cam_x, cam_y, cam_z)

# Фокусировка на центр скопления
target = bpy.data.objects.new("Focus_Target", None)
scene.collection.objects.link(target)
target.location = (0, 0, 0.005)

track = cam_obj.constraints.new(type='TRACK_TO')
track.target = target
track.track_axis = 'TRACK_NEGATIVE_Z'
track.up_axis = 'UP_Y'

bpy.context.view_layer.update()
scene.render.resolution_x = 640
scene.render.resolution_y = 640

# ==========================================
# 8. Алгоритм Convex Hull и экспорт YOLO-seg
# ==========================================
def get_monotone_convex_hull(points):
    """Andrew's Monotone Chain 2D Convex Hull"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def calculate_yolo_seg_and_bbox(scene, camera, obj):
    mat_world = obj.matrix_world
    mesh = obj.data
    coords_2d = [world_to_camera_view(scene, camera, mat_world @ v.co) for v in mesh.vertices]
    visible = [(c.x, c.y) for c in coords_2d if c.z > 0]

    if len(visible) < 4:
        return None, None

    # Ограничение по экрану
    clamped = [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in visible]
    hull = get_monotone_convex_hull(clamped)

    if len(hull) < 3:
        return None, None

    # В Blender Y направлен снизу вверх, в YOLO — сверху вниз
    # Разреживаем полигон до ~16-24 точек для легковесного формата YOLO-seg
    step = max(1, len(hull) // 20)
    sampled_hull = hull[::step]

    polygon_yolo = []
    for x, y in sampled_hull:
        polygon_yolo.extend([round(x, 6), round(1.0 - y, 6)])

    # Расчет Bbox из полигона
    xs = [x for x, y in hull]
    ys = [y for x, y in hull]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw, bh = max_x - min_x, max_y - min_y

    if bw < 0.01 or bh < 0.01:
        return None, None

    xc = min_x + bw / 2.0
    yc = 1.0 - (min_y + bh / 2.0)
    bbox_yolo = (xc, yc, bw, bh)

    return polygon_yolo, bbox_yolo

# ==========================================
# 9. Рендеринг и сохранение
# ==========================================
out_dir = Path(r"D:\Tabmed\Tabmed_ml\synthetic_3d_output")
img_dir = out_dir / "images"
lbl_dir = out_dir / "labels"
lbl_seg_dir = out_dir / "labels_seg"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)
lbl_seg_dir.mkdir(parents=True, exist_ok=True)

img_path = img_dir / "synth_0001.png"
scene.render.filepath = str(img_path)
bpy.ops.render.render(write_still=True)

lines_bbox = []
lines_seg = []

for item in pills_data:
    poly, bbox = calculate_yolo_seg_and_bbox(scene, cam_obj, item["obj"])
    if poly is not None:
        cls_id = item["class_id"]
        # Сохранение стандартного BBox (YOLOv8/v11 detect)
        lines_bbox.append(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        # Сохранение полигонов (YOLOv8/v11/v26 segment)
        poly_str = " ".join(f"{coord:.6f}" for coord in poly)
        lines_seg.append(f"{cls_id} {poly_str}")

with open(lbl_dir / "synth_0001.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines_bbox) + "\n")

with open(lbl_seg_dir / "synth_0001.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines_seg) + "\n")

print(f"\n[УСПЕХ] Готово! Размечено объектов: {len(lines_bbox)}")
