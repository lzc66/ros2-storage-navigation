# linorobot2_gazebo

Gazebo simulation package for linorobot2. Contains worlds, models, launch files, and tools for running the robot in simulation.

## Tools

### image_to_gazebo

A GUI tool that converts a plain floor-plan or occupancy-map image (PNG, JPG, BMP, etc.) into a ready-to-use Gazebo world. It lets you calibrate the image's real-world scale and set the coordinate origin interactively, then generates the 3D mesh, model SDF, and world SDF files automatically.

Under the hood it calls `map_to_gazebo`, which traces every dark (occupied) pixel in the image and extrudes it into a 3D wall mesh exported as an STL file.

#### Running

```bash
ros2 run linorobot2_gazebo image_to_gazebo
```

The **Map Image Processor** window (1000 × 700 px) opens with a **Controls** sidebar on the left and an **Image View** canvas on the right.

#### Workflow

Follow the steps below in order:

---

**1. Load Image**

Click **Load Image** and select your floor-plan file (PNG, JPG, JPEG, BMP, GIF, or TIFF).

The file dialog opens at `linorobot2_gazebo/linorobot2_gazebo/images/` by default — place your images there for easy access.

The image is displayed on the canvas. X (red, pointing right) and Y (green, pointing up) reference axes are drawn in the bottom-left corner.

---

**2. Set Meters Per Pixel**

This step calibrates the real-world scale of the image.

1. Click **Set Meters Per Pixel**. The status bar prompts you to click two points.
2. Click the first point on the canvas (a red dot appears).
3. Click the second point (another red dot and a connecting line appear).
4. A dialog asks for the real-world distance between the two points **in metres**. Enter the known distance and click **OK**.

The resolution (metres/pixel) is calculated and shown in the **Map Info** panel.

---

**3. Set Origin**

This step defines where the Gazebo world coordinate origin `(0, 0)` falls on the image.

1. Click **Set Origin**. The status bar prompts you to click a point.
2. Click the desired origin location on the image (e.g. a doorway, room corner, or the robot's starting position).

A small circle with red (X) and green (Y) arrows is drawn at the clicked point. The computed world-frame origin `[x, y, 0.0]` is shown in the **Map Info** panel.

> The origin follows the ROS `map_server` convention: the pixel you click becomes `(0, 0)` in the world frame, with Y pointing up in the image.

---

**4. Set Wall Height**

In the **Wall Height** field (default `1.0` m), enter the desired extrusion height for the walls.

---

**5. Generate World**

Click **Generate World**. A dialog appears with three fields:

| Field | Description |
|-------|-------------|
| **World Name** | A human-readable name (spaces and CamelCase are converted to `snake_case`). A live preview shows the final name. |
| **Model Directory** | Directory where the model folder will be created. Defaults to `linorobot2_gazebo/models/`. |
| **World SDF Directory** | Directory where the world SDF file will be written. Defaults to `linorobot2_gazebo/worlds/`. |

Click **Generate** (or press Enter). A progress splash is shown while the mesh is being built. When complete, a success dialog confirms the output paths.

#### Output Files

For a world named `my_map` the following files are created:

```
models/
└── my_map/
    ├── model.config
    ├── my_map.sdf
    └── meshes/
        └── my_map.stl

worlds/
└── my_map_world.sdf
```

#### Using the Generated World in Gazebo

After rebuilding (or if you used `--symlink-install`), launch the simulation with:

```bash
ros2 launch linorobot2_gazebo gazebo.launch.py world_name:=my_map_world
```

The `world_name` argument resolves to `linorobot2_gazebo/worlds/<world_name>.sdf`, so pass the base filename without the `.sdf` extension.

To use a world SDF located outside the package, pass the full path instead:

```bash
ros2 launch linorobot2_gazebo gazebo.launch.py world_path:=/absolute/path/to/my_map_world.sdf
```

---

### create_worlds_from_maps

A batch CLI tool that converts **all** ROS map YAML files in `linorobot2_navigation/maps/` into Gazebo worlds in one command. It is the non-interactive counterpart to `image_to_gazebo` — useful for regenerating every world after maps have been updated.

For each YAML file it calls `map_to_gazebo` to extrude the occupancy-map image into a 3D wall mesh, then writes the model SDF and world SDF files.

#### Running

```bash
ros2 run linorobot2_gazebo create_worlds_from_maps
```

The tool automatically locates `linorobot2_navigation/maps/` by searching the workspace, then writes all generated models to `linorobot2_gazebo/models/` and all world SDF files to `linorobot2_gazebo/worlds/`. No arguments are needed.

#### Output Files

For each map YAML named `<map_name>.yaml` the following files are created (same layout as `image_to_gazebo`):

```
models/
└── <map_name>/
    ├── model.config
    ├── <map_name>.sdf
    └── meshes/
        └── <map_name>.stl

worlds/
└── <map_name>.sdf
```

#### Using the Generated Worlds in Gazebo

After rebuilding (or with `--symlink-install`), launch any generated world with:

```bash
ros2 launch linorobot2_gazebo gazebo.launch.py world_name:=<map_name>
```
