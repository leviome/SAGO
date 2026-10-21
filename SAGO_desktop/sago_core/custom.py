import open3d as o3d
import numpy as np


def customcam_to_frustum(cam, color=(1, 0, 0), scale=0.1, w=1024, h=1024):
    # Get intrinsic and extrinsic
    # try:
    #     intrinsic = cam.K
    # except AttributeError:
    intrinsic = np.array([[886.81001348, 0., 512.],
                          [0., 886.81001348, 512.],
                          [0., 0., 1.]])
    # Build 4x4 extrinsic matrix from R, T
    extr = np.eye(4)
    extr[:3, :3] = cam.R
    extr[:3, 3] = cam.T
    width, height = w, h

    # Compute image corners in normalized camera coordinates
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ])
    corners = (corners - [cx, cy, 0]) / [fx, fy, 1]
    corners = np.hstack([corners[:, :2], np.ones((4, 1))]) * scale

    origin = np.zeros((1, 3))
    points = np.vstack([origin, corners])
    # Transform to world coordinates
    points = (extr[:3, :3] @ points.T + extr[:3, 3:4]).T

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_plane_mesh(point, normal, size=1, color=[0, 0, 1, 0.5]):
    # Create two orthogonal vectors to the normal
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, [0, 0, 1]):
        tangent = np.array([1, 0, 0])
    else:
        tangent = np.cross(normal, [0, 0, 1])
        tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    # Four corners of the plane
    corners = []
    for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        corner = point + (tangent * dx + bitangent * dy) * size
        corners.append(corner)
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(corners)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [2, 3, 0]])
    mesh.paint_uniform_color(color[:3])
    mesh.compute_vertex_normals()
    if len(color) == 4:
        mesh.vertex_colors = o3d.utility.Vector3dVector([color[:3]] * 4)
    return mesh


def visualize_customcams(cams, look_at_point=None, plane_normal=None,
                         show_world_axis=True, draw_cube=True, w=1024, h=1024):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if show_world_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(axis)

    if look_at_point is not None:
        # look_at_point: [x, y, z]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector([look_at_point])
        pcd.colors = o3d.utility.Vector3dVector([[1, 1, 0]])  # yellow
        vis.add_geometry(pcd)

        # Draw a sphere with look_at_point as the origin
        # sphere_radius = np.linalg.norm(cams[0].T - np.array(look_at_point))
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, create_uv_map=True)
        # sphere.translate(look_at_point)

        # vis.add_geometry(sphere)

        if plane_normal is not None:
            plane = create_plane_mesh(np.array(look_at_point), np.array(plane_normal), size=5,
                                      color=[0.5, 0.5, 0.5, 0.1])
            vis.add_geometry(plane)

    if draw_cube:
        # Draw a cube with look_at_point as the center and align it with the plane
        cube = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.5)
        cube.translate(np.array(look_at_point) - np.array([0.25, 0.25, 0.25]))  # Center the cube

        # Align the cube with the plane's normal
        plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
        tangent = np.cross(plane_normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 1e-6:  # Handle edge case where plane_normal is [0, 0, 1]
            tangent = np.array([1, 0, 0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(plane_normal, tangent)
        rotation_matrix = np.array([tangent, bitangent, plane_normal]).T
        cube.rotate(rotation_matrix, center=cube.get_center())

        cube.paint_uniform_color([1, 0, 0])  # Red color
        vis.add_geometry(cube)

    colors = [
        [0, 0, 0],  # Black
        [1, 0, 1],  # Magenta
        [0, 0, 1],  # Blue
    ]

    for i, cam in enumerate(cams):
        color = colors[2] if i > 1 else colors[i]
        frustum = customcam_to_frustum(cam, color=color, scale=0.5, w=w, h=h)
        vis.add_geometry(frustum)

    # Draw a line and a circle connecting cams[1:]
    if len(cams) > 1:
        line_points = np.array([cams[0].T, cams[1].T])
        line = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_points),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        line.colors = o3d.utility.Vector3dVector([[0, 1, 1]])  # Cyan color
        vis.add_geometry(line)

        cam_positions = np.array([cam.T for cam in cams[1:]])
        center = cam_positions.mean(axis=0)
        radius = np.linalg.norm(cam_positions - center, axis=1).mean()
        theta = np.linspace(0, 2 * np.pi, 100)

        # Create a rotation matrix to align the circle with the plane
        plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
        tangent = np.cross(plane_normal, [0, 0, 1])
        if np.linalg.norm(tangent) < 1e-6:  # Handle edge case where plane_normal is [0, 0, 1]
            tangent = np.array([1, 0, 0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(plane_normal, tangent)

        # Generate circle points in the plane
        circle_points = np.array([
            center + radius * (np.cos(t) * tangent + np.sin(t) * bitangent)
            for t in theta
        ])
        lines = [[i, i + 1] for i in range(len(circle_points) - 1)] + [[len(circle_points) - 1, 0]]
        circle = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(circle_points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        circle.colors = o3d.utility.Vector3dVector([[1, 0.5, 0] for _ in lines])  # Orange color
        vis.add_geometry(circle)

    vis.run()
    vis.destroy_window()
