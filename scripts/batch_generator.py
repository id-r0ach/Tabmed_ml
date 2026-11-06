import bpy
import math
import random
import sys
import json
from pathlib import Path
from mathutils import Vector

# ==========================================
# 1. Палитра эмиссионных цветов для ID-пасса
# ==========================================
ID_PALETTE_RGB = [
    (1.0, 0.0, 0.0),    # 0: Красный
    (0.0, 1.0, 0.0),    # 1: Зеленый
    (0.0, 0.0, 1.0),    # 2: Синий
    (1.0, 1.0, 0.0),    # 3: Желтый
    (1.0, 0.0, 1.0),    # 4: Пурпурный
    (0.0, 1.0, 1.0),    # 5: Голубой
    (1.0, 0.5, 0.0),    # 6: Оранжевый
    (0.5, 0.0, 1.0),    # 7: Фиолетовый
    (0.0, 0.5, 1.0),    # 8: Лазурный
    (0.5, 1.0, 0.0),    # 9: Лайм
    (1.0, 0.0, 0.5),    # 10: Розовый
    (0.0, 1.0, 0.5),    # 11: Мятный
]

PHARMA_PALETTE = {
    "white": (0.93, 0.93, 0.93),
    "chalk_white": (0.87, 0.86, 0.82),
    "yellow": (0.95, 0.82, 0.12),
    "blue": (0.08, 0.35, 0.85),
    "red": (0.85, 0.12, 0.12),
    "green": (0.12, 0.65, 0.25),
    "orange": (0.95, 0.45, 0.05),
    "pink": (0.92, 0.45, 0.55)
}

# ==========================================
# 2. Шейдеры материалов
# ==========================================
def create_table_material():
    mat = bpy.data.materials.new(name="Surface_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")

    surface_types = ["dark", "light", "wood_tint", "skin_tint"]
    choice = random.choice(surface_types)

    if choice == "dark":
        base_col = (random.uniform(0.08, 0.18), random.uniform(0.08, 0.18), random.uniform(0.09, 0.20), 1.0)
        rough = random.uniform(0.6, 0.85)
    elif choice == "light":
        base_col = (random.uniform(0.75, 0.90), random.uniform(0.75, 0.90), random.uniform(0.75, 0.90), 1.0)
        rough = random.uniform(0.3, 0.6)
    elif choice == "skin_tint":
        base_col = (random.uniform(0.70, 0.85), random.uniform(0.50, 0.65), random.uniform(0.40, 0.55), 1.0)
        rough = random.uniform(0.5, 0.7)
    else:
        base_col = (random.uniform(0.25, 0.40), random.uniform(0.18, 0.28), random.uniform(0.12, 0.20), 1.0)
        rough = random.uniform(0.4, 0.7)

    tex_noise = nodes.new(type="ShaderNodeTexNoise")
    tex_noise.inputs['Scale'].default_value = random.uniform(150.0, 300.0)
    tex_noise.inputs['Detail'].default_value = 4.0

    color_ramp = nodes.new(type="ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].color = (base_col[0]*0.9, base_col[1]*0.9, base_col[2]*0.9, 1.0)
    color_ramp.color_ramp.elements[1].color = base_col

    links.new(tex_noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = rough
    return mat

def get_chalk_mat(name, color_rgb):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = random.uniform(0.60, 0.85)
    if 'Subsurface Weight' in bsdf.inputs:
        bsdf.inputs['Subsurface Weight'].default_value = 0.08
    return mat

def get_gelatin_mat(name, color_rgb):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = random.uniform(0.08, 0.18)
    bsdf.inputs['IOR'].default_value = 1.48
    if 'Coat Weight' in bsdf.inputs:
        bsdf.inputs['Coat Weight'].default_value = 1.0
        bsdf.inputs['Coat Roughness'].default_value = 0.04
    return mat

# ==========================================
# 3. Фабрика процедурных мешей
# ==========================================
def spawn_round_tablet(loc, rot, has_score=False):
    radius = random.uniform(0.008, 0.012)
    depth = random.uniform(0.0035, 0.006)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth, location=loc, rotation=rot)
    obj = bpy.context.active_object

    if has_score:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(loc[0], loc[1], loc[2] + depth*0.48), rotation=rot)
        cutter = bpy.context.active_object
        cutter.scale = (radius * 2.2, 0.0006, depth * 0.3)
        bool_m = obj.modifiers.new(name="Cut", type='BOOLEAN')
        bool_m.operation = 'DIFFERENCE'
        bool_m.object = cutter
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier="Cut")
        bpy.data.objects.remove(cutter, do_unlink=True)

    bev = obj.modifiers.new(name="Bevel", type='BEVEL')
    bev.width = 0.0008
    bev.segments = 3
    bpy.ops.object.modifier_apply(modifier="Bevel")
    bpy.ops.object.shade_smooth()

    col = PHARMA_PALETTE[random.choice(["white", "chalk_white", "yellow", "pink", "green"])]
    obj.data.materials.append(get_chalk_mat("Tab_Mat", col))
    return obj

def spawn_oval_tablet(loc, rot):
    obj = spawn_round_tablet(loc, rot, has_score=False)
    obj.scale = (random.uniform(1.4, 1.7), 1.0, 1.0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    return obj

def spawn_capsule(loc, rot):
    radius = random.uniform(0.0042, 0.0055)
    depth = random.uniform(0.015, 0.021)
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=loc, rotation=rot)
    obj = bpy.context.active_object

    bev = obj.modifiers.new(name="CapBevel", type='BEVEL')
    bev.width = radius * 0.95
    bev.segments = 8
    bpy.ops.object.modifier_apply(modifier="CapBevel")
    bpy.ops.object.shade_smooth()

    col1 = random.choice(["blue", "red", "green", "orange", "yellow"])
    col2 = random.choice(["white", "yellow", "orange"])
    obj.data.materials.append(get_gelatin_mat("Top", PHARMA_PALETTE[col1]))
    obj.data.materials.append(get_gelatin_mat("Bot", PHARMA_PALETTE[col2]))

    for p in obj.data.polygons:
        p.material_index = 0 if p.center.z >= 0 else 1
    return obj

# ==========================================
# 4. Рендер одной сцены (Cycles Beauty + ID Mask)
# ==========================================
def render_single_scene(output_img_path):
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

    if not scene.rigidbody_world:
        bpy.ops.rigidbody.world_add()
    rbw = scene.rigidbody_world
    rbw.substeps_per_frame = 25
    rbw.solver_iterations = 40

    # Стол
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.05))
    table = bpy.context.active_object
    table.name = "Table"
    table.scale = (2.0, 2.0, 0.1)
    bpy.ops.rigidbody.object_add()
    table.rigid_body.type = 'PASSIVE'
    table.rigid_body.collision_shape = 'BOX'
    table.rigid_body.friction = random.uniform(0.75, 0.95)
    table.data.materials.append(create_table_material())

    # Спавн таблеток (от 3 до 8 шт)
    pills = []
    n_pills = random.randint(3, 8)
    types = ['round', 'round_score', 'oval', 'capsule', 'capsule']

    for i in range(n_pills):
        t = random.choice(types)
        loc = (random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02), 0.025 + i * 0.016)
        rot = (random.uniform(0, 3.14), random.uniform(0, 3.14), random.uniform(0, 3.14))

        if t == 'round':
            obj = spawn_round_tablet(loc, rot, has_score=False)
            cls_id = 1
        elif t == 'round_score':
            obj = spawn_round_tablet(loc, rot, has_score=True)
            cls_id = 1
        elif t == 'oval':
            obj = spawn_oval_tablet(loc, rot)
            cls_id = 1
        else:
            obj = spawn_capsule(loc, rot)
            cls_id = 0

        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = 'ACTIVE'
        obj.rigid_body.collision_shape = 'CONVEX_HULL'
        obj.rigid_body.mass = 0.003
        obj.rigid_body.friction = 0.75
        obj.rigid_body.linear_damping = 0.65
        obj.rigid_body.angular_damping = 0.65
        pills.append((obj, cls_id))

    # Симуляция падения
    for frame in range(1, 70):
        scene.frame_set(frame)

    deps = bpy.context.evaluated_depsgraph_get()
    for obj, _ in pills:
        eval_obj = obj.evaluated_get(deps)
        obj.matrix_world = eval_obj.matrix_world.copy()
        obj.rigid_body.enabled = False

    scene.frame_set(1)
    bpy.context.view_layer.update()

    # Освещение
    light_data = bpy.data.lights.new(name="Light", type='POINT')
    light_data.energy = random.uniform(50.0, 85.0)
    light_data.shadow_soft_size = random.uniform(0.02, 0.05)
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (random.uniform(-0.15, 0.15), random.uniform(-0.18, -0.05), random.uniform(0.25, 0.40))

    # Камера
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.lens = random.uniform(45, 58)
    cam_obj = bpy.data.objects.new(name="Camera", object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    dist = random.uniform(0.18, 0.27)
    pitch = random.uniform(math.radians(18), math.radians(75))
    cam_z = dist * math.sin(pitch)
    cam_y = -dist * math.cos(pitch)
    cam_x = random.uniform(-0.05, 0.05)
    cam_obj.location = (cam_x, cam_y, cam_z)

    target = bpy.data.objects.new("Target", None)
    scene.collection.objects.link(target)
    target.location = (0, 0, 0.005)

    track = cam_obj.constraints.new(type='TRACK_TO')
    track.target = target
    track.track_axis = 'TRACK_NEGATIVE_Z'
    track.up_axis = 'UP_Y'

    bpy.context.view_layer.update()
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640

    # 1. Beauty Pass (JPEG)
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 95
    scene.render.use_file_extension = False
    scene.render.filepath = str(output_img_path)
    bpy.ops.render.render(write_still=True)

    # 2. ID Mask Pass (Эмиссионные цвета без теней)
    light_obj.hide_render = True
    if scene.world and scene.world.node_tree:
        bg_node = scene.world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs['Strength'].default_value = 0.0

    mat_black = bpy.data.materials.new(name="ID_Black")
    mat_black.use_nodes = True
    mat_black.node_tree.nodes.clear()
    emit_b = mat_black.node_tree.nodes.new(type='ShaderNodeEmission')
    emit_b.inputs['Color'].default_value = (0, 0, 0, 1)
    out_b = mat_black.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    mat_black.node_tree.links.new(emit_b.outputs['Emission'], out_b.inputs['Surface'])
    table.data.materials.clear()
    table.data.materials.append(mat_black)

    metadata = []
    for idx, (obj, cls_id) in enumerate(pills):
        col_rgb = ID_PALETTE_RGB[idx % len(ID_PALETTE_RGB)]
        mat_id = bpy.data.materials.new(name=f"ID_Mat_{idx}")
        mat_id.use_nodes = True
        mat_id.node_tree.nodes.clear()
        emit = mat_id.node_tree.nodes.new(type='ShaderNodeEmission')
        emit.inputs['Color'].default_value = (*col_rgb, 1.0)
        emit.inputs['Strength'].default_value = 1.0
        out_n = mat_id.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        mat_id.node_tree.links.new(emit.outputs['Emission'], out_n.inputs['Surface'])
        obj.data.materials.clear()
        obj.data.materials.append(mat_id)

        metadata.append({
            "class_id": cls_id,
            "color_rgb": [int(round(c * 255)) for c in col_rgb]
        })

    cycles.samples = 1
    cycles.preview_samples = 1
    cycles.use_denoising = False
    cycles.max_bounces = 0
    cycles.diffuse_bounces = 0
    cycles.glossy_bounces = 0
    cycles.transmission_bounces = 0

    try:
        scene.view_settings.view_transform = 'Raw'
    except Exception:
        scene.view_settings.view_transform = 'Standard'

    mask_path = output_img_path.with_name(f"{output_img_path.stem}_mask.png")
    meta_path = output_img_path.with_name(f"{output_img_path.stem}_meta.json")

    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = str(mask_path)
    bpy.ops.render.render(write_still=True)

    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf)

# ==========================================
# 5. CLI Точка входа
# ==========================================
if __name__ == "__main__":
    args = sys.argv
    count = 10
    if "--" in args:
        idx = args.index("--")
        if len(args) > idx + 1:
            count = int(args[idx + 1])

    root_dir = Path(r"D:\Tabmed\Tabmed_ml\synthetic_dataset")
    val_count = max(1, int(count * 0.2))
    train_count = count - val_count
    splits = [("train", train_count), ("val", val_count)]
    current_idx = 1

    for split_name, total in splits:
        img_dir = root_dir / "images" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)

        for _ in range(total):
            img_file = img_dir / f"synth_{current_idx:06d}.jpg"
            render_single_scene(img_file)
            print(f"[{split_name.upper()} {current_idx}/{count}] Отрендерен: {img_file.name}")
            current_idx += 1

    print(f"\n[УСПЕХ] Рендер сцен в Blender завершен.")
