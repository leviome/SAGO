class SceneCard:
    def __init__(self, scene_id, root, scene_type,
                 up_vector=None,
                 radius=None, lookat=None, forward_vec=None
                 ):
        """
        The definition for SAGO's input
        Args:
            scene_id (str): scene_id
            root (str): root
            scene_type (str): ["dynamic", "3dgs", "2dgs"]
            up_vector (tuple): up_vector
            radius (float): radius
            lookat (tuple): lookat
            forward_vec (tuple): forward_vec
        """
        self.scene_id = scene_id
        self.root = root
        self.up_vector = up_vector
        self.scene_type = scene_type

        self.recommended_radius = radius
        self.recommended_lookat = lookat
        self.recommended_forward_vec = forward_vec

    def print_scene_info(self):
        print("========Scene Card========")
        print("SCENE ID:", self.scene_id)
        print("SCENE ROOT:", self.root)
        print("SCENE TYPE:", self.scene_type)
        print("UP VECTOR:", self.up_vector)
        print("RECOMMENDED RADIUS:", self.recommended_radius)
        print("RECOMMENDED LOOKAT:", self.recommended_lookat)
        print("RECOMMENDED FORWARD VECTOR:", self.recommended_forward_vec)
        print("==========================")
        print()


scenes = dict()
scenes["scene0"] = SceneCard(
    "scene0", root="/home/pc/gaussian-splatting/output/for_seg1",
    scene_type="3dgs", up_vector=(-0.06, -0.96, -0.26),
    lookat=(0.38, 1.88, -0.55), forward_vec=(-0.71,  0.65,  0.27), radius=5)

scenes["desktop"] = SceneCard(
    "desktop", root="/home/pc/gaussian-splatting/output/figurines_desktop",
    scene_type="3dgs", up_vector=(-0.04, -1.00, -0.09),
    lookat=(0.36, -1.60, -0.91), forward_vec=(-0.58,  0.56,  0.59), radius=5)
scenes["scene1"] = SceneCard(
    "scene1", root="/home/pc/gaussian-spla"
                   "tting/output/pku_group/",
    scene_type="3dgs", up_vector=(0.0116, -0.9938, 0.1105))
scenes["scene2"] = SceneCard("scene2",
                             root="/home/pc/gaussian-splatting/output/3d-ovs/bed",
                             scene_type="3dgs", up_vector=(-0.03, -0.72, -0.69))
scenes["scene3"] = SceneCard("scene3",
                             root="/home/pc/gaussian-splatting/output/3d-ovs/sofa",
                             scene_type="3dgs", up_vector=(-0.0368, -0.7625, -0.6459))
scenes["scene4"] = SceneCard("scene4",
                             root="/home/pc/gaussian-splatting/output/bear/",
                             scene_type="3dgs", up_vector=(0.08, -0.85, -0.53), radius=10)
scenes["scene5"] = SceneCard("scene5",
                             root="/home/pc/gaussian-splatting/output/mip360/garden/",
                             scene_type="3dgs", up_vector=(0, -0.89, -0.46))
scenes["scene6"] = SceneCard("scene6",
                             root="/home/pc/gaussian-splatting/output/lerf/teatime",
                             scene_type="3dgs", up_vector=(0.02, -0.85, -0.52))
scenes["scene7"] = SceneCard("scene7",
                             root="/home/pc/gaussian-splatting/output/pku_origin",
                             scene_type="3dgs", up_vector=(0.0074, -0.9905, 0.1374), radius=5)
scenes["scene8"] = SceneCard("scene8",
                             root="/home/pc/gaussian-splatting/output/3d-ovs/bench",
                             scene_type="3dgs", up_vector=(-0.0109, -0.6912, -0.7226))
scenes["scene9"] = SceneCard("scene9",
                             root="/home/pc/gaussian-splatting/output/lerf/ramen",
                             scene_type="3dgs", up_vector=(-0.03, -0.69, -0.72))
scenes["scene10"] = SceneCard("scene10",
                              root="/home/pc/gaussian-splatting/output/llff/horns/",
                              scene_type="3dgs", up_vector=(0.02, -0.99, -0.17))
scenes["scene11"] = SceneCard("scene11",
                              root="/home/pc/dynamic/gaussian-splatting/output/Tanks/Family",
                              scene_type="3dgs", up_vector=(0.01, -0.99, -0.17))
scenes["scene12"] = SceneCard("scene12",
                              root="/home/pc/gaussian-splatting/output/Tank/truck/",
                              scene_type="3dgs", up_vector=(0.02, -0.99, -0.17),
                              forward_vec=(0.31, 0.44, -0.84), radius=13)
scenes["scene13"] = SceneCard("scene13",  #
                              root="/home/pc/dynamic/gaussian-splatting/output/Tanks/Horse",
                              scene_type="3dgs", up_vector=(-0.09, -0.99, -0.11))
scenes["scene14"] = SceneCard("scene14",
                              root="/home/pc/gaussian-splatting/output/scannetpp/036bce3393/",
                              scene_type="3dgs", up_vector=(-0., -1., -0.),
                              lookat=(1.74, 1.20, 3.85), forward_vec=(0.45, 0.25, 0.86), radius=8)
scenes["scene15"] = SceneCard("scene15",
                              root="/home/pc/gaussian-splatting/output/llff/fortress",
                              scene_type="3dgs", up_vector=(-0.05, -0.76, -0.65),
                              lookat=(1.43, -1.24, 18.16), forward_vec=(0.02, -0.45, 0.89), radius=34)

scenes["scene16"] = SceneCard("scene16",
                              root="/home/pc/Data/FastGS/output/vr01",
                              scene_type="3dgs", up_vector=(-0.68, 0.19,  -0.71),
                              lookat=(1.55, -0.35,  1.78), forward_vec=(0.02, -0.45, 0.89), radius=3)

scenes["scene17"] = SceneCard("scene17",
                              root="/home/pc/dynamic/gaussian-splatting/output/vr02",
                              scene_type="3dgs", up_vector=(-0.68, 0.19,  -0.71),
                              lookat=(1.55, -0.35,  1.78), forward_vec=(0.02, -0.45, 0.89), radius=7)

scenes["haptic_bear"] = SceneCard("haptic_bear",
                              root="/home/pc/dynamic/gaussian-splatting/output/haptic_bear2",
                              scene_type="3dgs", up_vector=(-0.95, -0.29, -0.10),
                              lookat=(0.09, 0.48, 1.64), forward_vec=(-0.09,  0.31,  0.95), radius=3)
scenes["6b509"] = SceneCard("6b509",
                              root="/home/pc/dynamic/gaussian-splatting/output/6b509",
                              scene_type="3dgs", up_vector=(0.07,  -1.00, 0.03),
                              lookat=(-3.77, -0.39,  9.88), forward_vec=(0.02, -0.45, 0.89), radius=10)

scenes["Dy01"] = SceneCard("Dy01",
                           root="/home/pc/Data/SportTrack3D/dataset/Dance_Dunhuang_Pair_1080/",
                           scene_type="dynamic", up_vector=(0.01, -0.98, 0.17),
                           forward_vec=(0.19, 0.43, 0.88), radius=10)

scenes["Dy02"] = SceneCard("Dy02",
                           root="/home/pc/Data/SportTrack3D/dataset/VRU_gz/",
                           scene_type="dynamic", up_vector=(-0.08, -0.93, -0.36),
                           lookat=(3.28, -0.84, 10.90), forward_vec=(0.62, 0.02, 0.78), radius=12.8)

scenes["Dy03"] = SceneCard("Dy03",
                           root="/home/pc/Data/SportTrack3D/dataset/VRU_dg4/",
                           scene_type="dynamic", up_vector=(-0.18, -0.88, -0.45),
                           lookat=(8.32, -0.42, 10.09), forward_vec=(0.74, 0.02, 0.67), radius=14.7)

scenes["Dy04"] = SceneCard("Dy04",
                           root="/home/pc/Data/DyGSset/Basketball_full_1",
                           scene_type="dynamic", up_vector=(-0.1, -0.93, -0.36),
                           lookat=(0.26, 1.05, 0.65), forward_vec=(-0.33, 0.10, 0.94), radius=2.8)

scenes["Dy05"] = SceneCard("Dy05",
                           root="/home/pc/Data/SportTrack3D/dataset/mobile_stage/",
                           scene_type="dynamic", up_vector=(0.0, -0.97, -0.24),
                           lookat=(-1.65, 0.09, 2.65), forward_vec=(0.06, 0.21, 0.98), radius=4.8)

scenes["Dy06"] = SceneCard("Dy06",
                           root="/home/pc/Data/SportTrack3D/dataset/utsz-football",
                           scene_type="dynamic", up_vector=(0.08, -1., -0.02),
                           lookat=(-2.97, 0.20, 1.68), forward_vec=(-0.60, 0.05, 0.80), radius=2)

scenes["Dy07"] = SceneCard("Dy07",
                           root="/home/pc/Data/SportTrack3D/dataset/1080_Dance_Dunhuang_Single",
                           scene_type="dynamic", up_vector=(-0.01, -0.97, 0.24),
                           lookat=(0.33, -0.15, 1.04), forward_vec=(-0.12, 0.33, 0.93), radius=4.8)

scenes["Dy08"] = SceneCard("Dy08",
                           root="/home/pc/Data/SportTrack3D/coffee_martini",
                           scene_type="dynamic", up_vector=(0.02, -0.99, -0.17), )

scenes["gs2d_0"] = SceneCard("gs2d_0",
                             root="/home/pc/haptic/2d-gaussian-splatting/output/749d4c31-4",
                             scene_type="2dgs", up_vector=(0.49,  -0.80,  -0.34))


# referring

scenes["gr_scannet_0000"] = SceneCard(
    "gr_scannet_0000", root="/home/pc/GaussianRAG/datasets/zerosplat-dataset/GR-ScanNet/gr-scannet/scene0000_00",
    scene_type="3dgs", up_vector=(0.28,  -0.01, 0.96),
    lookat=(4.76, 2.20, 0.55), forward_vec=(-0.71,  0.65,  0.27), radius=5)

scenes["gr_scannet_0062"] = SceneCard(
    "gr_scannet_0062", root="/home/pc/GaussianRAG/datasets/zerosplat-dataset/GR-ScanNet/gr-scannet/scene0062_00",
    scene_type="3dgs", up_vector=(0.28,  -0.01, 0.96),
    lookat=(4.76, 2.20, 0.55), forward_vec=(-0.71,  0.65,  0.27), radius=5)


