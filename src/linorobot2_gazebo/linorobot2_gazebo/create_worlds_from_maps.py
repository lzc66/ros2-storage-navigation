import os
import sys
import yaml
import argparse
from linorobot2_gazebo.map_to_gazebo import process_maps

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SCRIPT_DIR)


def _resolve_src_pkg_dir(pkg_name: str) -> str | None:
    """
    Locate <workspace>/src/linorobot2/<pkg_name>/ regardless of whether the
    script is running from the colcon install tree or directly from source.

    Installed layout:
      <ws>/install/linorobot2_gazebo/lib/python3.12/site-packages/linorobot2_gazebo/create_worlds_from_maps.py
      parts: ['', ..., 'install', 'linorobot2_gazebo', 'lib', 'python3.x', 'site-packages',
              'linorobot2_gazebo', 'create_worlds_from_maps.py']

    Source layout:
      <ws>/src/linorobot2/linorobot2_gazebo/linorobot2_gazebo/create_worlds_from_maps.py
      parts: ['', ..., 'src', 'linorobot2', 'linorobot2_gazebo', 'linorobot2_gazebo',
              'create_worlds_from_maps.py']

    Finds the first ('install'|'src') + 'linorobot2_gazebo' pair, takes everything
    before it as the workspace root, then returns <workspace>/src/linorobot2/<pkg_name>/
    if that directory actually exists.
    """
    parts = os.path.abspath(__file__).split(os.sep)
    for marker in ('install', 'src'):
        for i, part in enumerate(parts[:-1]):
            if part == marker and parts[i + 1] == 'linorobot2_gazebo':
                workspace_root = os.sep.join(parts[:i]) or os.sep
                candidate = os.path.join(workspace_root, 'src', 'linorobot2', pkg_name)
                if os.path.isdir(candidate):
                    return candidate
    return None


_GAZEBO_SRC_PKG_DIR = _resolve_src_pkg_dir('linorobot2_gazebo')
_NAV_SRC_PKG_DIR = _resolve_src_pkg_dir('linorobot2_navigation')

_DEFAULT_MODELS_DIR = (
    os.path.join(_GAZEBO_SRC_PKG_DIR, 'models')
    if _GAZEBO_SRC_PKG_DIR else os.path.join(_PKG_DIR, 'models')
)
_DEFAULT_WORLDS_DIR = (
    os.path.join(_GAZEBO_SRC_PKG_DIR, 'worlds')
    if _GAZEBO_SRC_PKG_DIR else os.path.join(_PKG_DIR, 'worlds')
)
_DEFAULT_MAPS_DIR = (
    os.path.join(_NAV_SRC_PKG_DIR, 'maps')
    if _NAV_SRC_PKG_DIR else None
)


def main():
    parser = argparse.ArgumentParser(
        description='Create Gazebo worlds from all maps in linorobot2_navigation/maps.'
    )
    parser.add_argument(
        '--map_dir', type=str,
        default=_DEFAULT_MAPS_DIR,
        help=f'Directory containing YAML map files (default: {_DEFAULT_MAPS_DIR})'
    )
    parser.add_argument(
        '--model_dir', type=str,
        default=_DEFAULT_MODELS_DIR,
        help=f'Gazebo model output directory (default: {_DEFAULT_MODELS_DIR})'
    )
    parser.add_argument(
        '--world_dir', type=str,
        default=_DEFAULT_WORLDS_DIR,
        help=f'World SDF output directory (default: {_DEFAULT_WORLDS_DIR})'
    )
    parser.add_argument(
        '--height', type=float, default=1.5,
        help='Height of the 3D map mesh in meters (default: 1.5)'
    )
    args = parser.parse_args()

    if args.map_dir is None:
        print('Error: Could not resolve linorobot2_navigation maps directory. '
              'Please provide --map_dir.')
        sys.exit(1)

    if not os.path.isdir(args.map_dir):
        print(f'Error: Map directory {args.map_dir} not found or is not a directory.')
        sys.exit(1)

    yaml_files = [
        os.path.join(args.map_dir, f)
        for f in os.listdir(args.map_dir)
        if f.lower().endswith('.yaml') or f.lower().endswith('.yml')
    ]

    if not yaml_files:
        print(f'Error: No YAML files found in {args.map_dir}')
        sys.exit(1)

    print(f'Found {len(yaml_files)} YAML file(s) in {args.map_dir}')

    map_info_list = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as stream:
                map_info = yaml.safe_load(stream)

            map_name = os.path.splitext(os.path.basename(yaml_file))[0]
            map_info['map_name'] = map_name
            if os.path.exists(os.path.join(args.model_dir, map_name)):
                print(f'Skipping map: {map_name} (model already exists in {args.model_dir})')
                continue

            if not os.path.isabs(map_info['image']):
                yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
                map_info['image'] = os.path.join(yaml_dir, map_info['image'])

            map_info_list.append(map_info)
            print(f'Added map: {map_name}')
        except Exception as e:
            print(f'Error loading YAML file {yaml_file}: {str(e)}')

    if not map_info_list:
        print('No valid map files found. Exiting.')
        sys.exit(1)

    process_maps(map_info_list, args.model_dir, args.world_dir, args.height)


if __name__ == '__main__':
    main()
