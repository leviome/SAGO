import os
from collections import deque
from itertools import islice

from sago_core.sago_utils import *


class TrackAnyGaussian:
    def __init__(self, root, tracker, up_vector, prompt, obj_id=0,
                 prompt_cam=None, result_save_dir=None, device="cuda:0"):
        self.root = root
        self.tracker = tracker
        self.obj_id = obj_id
        self.up_vector = torch.FloatTensor(up_vector)
        self.device = device
        self.save_dir = result_save_dir

        self.padding = 5

        # self.gaussians = GaussianModel(sh_degree=0, disable_xyz_log_activation=True)
        # frame_plys = sorted(os.listdir(osp.join(root, "point")))
        # self.frame_plys = [osp.join(root, "point", p) for p in frame_plys]

        self.prompt = prompt
        self.prompt_cam = prompt_cam

        self.seg_level = 0
        self.initial_memory = None
        self.initial_ptr = None
        self.prev_mems = None
        self.prev_ptrs = None
        self.object_center = None
        self.initial_mask = None
        self.background_aug = False

        self.memory_cache_length = 20

        self.debug_folder = "./debug/tag/"
        os.makedirs(self.debug_folder, exist_ok=True)

        self.previous_mask_3d = None
        self.previous_cylinder = None

        self.background_color = torch.tensor([0., 0., 0.])

    def set_prompt(self, prompt):
        self.prompt = prompt

    def render_all(self, cams, gaussians, frame_id):
        # self.gaussians.load_ply(self.frame_plys[frame_id])
        for i, cam in enumerate(cams):
            renderings = render_simple(cam, gaussians)
            img = render2img(renderings["render"])
            folder = osp.join(self.debug_folder, "scene", f"frame_{frame_id:05d}")
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(osp.join(folder, f"cam_{i:03d}.png"), img)

    def spatial_temporal_track_all(self, gaussians, cam_id, prompt, cams, frame_id):
        # self.gaussians.load_ply(self.frame_plys[frame_id])
        if frame_id == 0:
            renderings = render_simple(cams[cam_id], gaussians)
            initial_view = render2img(renderings["render"])
            h, w, _ = initial_view.shape
            mask, memory, ptr = segmentation_simple(self.tracker, initial_view, prompt,
                                                    seg_level=self.seg_level)
            bg_color = distinct_background(renderings["render"], mask)
            self.background_color = bg_color
            self.initial_memory = memory
            self.initial_ptr = ptr
            self.initial_mask = mask
            self.prev_mems, self.prev_ptrs = deque([memory[0]]), deque([ptr[0]])
            viz_mask(initial_view, mask.cpu().numpy(), prefix=self.debug_folder,
                     box=prompt,
                     save_name=f"frame_{frame_id:03d}_cam_{cam_id:02d}.png")

        for cid, cam in enumerate(cams):
            if frame_id != 0:
                if cid != 32: continue
            renderings = render_simple(cam, self.gaussians)
            view = render2img(renderings["render"])
            mask = self.temporal_track(view, frame_id, viz=True)
            viz_mask(view, mask.cpu().numpy(), prefix=self.debug_folder,
                     save_name=f"frame_{frame_id:03d}_cam_{cid:02d}.png")
        # self.prev_mems, self.prev_ptrs = (deque([self.initial_memory[0]]),
        #                                   deque([self.initial_ptr[0]]))

    def tag_step(self, gaussians, cam, frame_id,
                 viz=False, cam_track=False, prev_mask3d=None):
        # self.gaussians.load_ply(self.frame_plys[frame_id])
        mask_3d = None if prev_mask3d is None else ~prev_mask3d
        if frame_id != 0:
            pos3d, radius, height = self.previous_cylinder
            region_mask_3d = area_tracking(gaussians, pos3d, radius, self.up_vector, height=None)
            if mask_3d is not None:
                mask_3d &= region_mask_3d
            else:
                mask_3d = region_mask_3d
            if cam_track:
                # cam_pos = get_camera_pos_3d(cam)
                radius, yaw, pitch = params2sphere(cam, self.up_vector, pos3d)
                cam = sphere2params(radius, yaw, pitch, pos3d, self.up_vector, cam.FoVx,
                                    like_cam=cam,
                                    device=self.device)

        renderings = render_simple(cam, gaussians,
                                   gaussian_mask=[mask_3d], render_fg_only=True,
                                   bg_color=self.background_color)

        img_fid = render2img(renderings["render"])
        depth_fid = renderings["depth"]
        if frame_id == 0:
            h, w, _ = img_fid.shape
            mask, memory, ptr = segmentation_simple(self.tracker, img_fid, self.prompt,
                                                    seg_level=self.seg_level)
            bg_color = distinct_background(renderings["render"], mask)
            self.background_color = bg_color
            self.initial_memory = memory
            self.initial_ptr = ptr
            self.initial_mask = mask
            self.prev_mems, self.prev_ptrs = deque([memory[0]]), deque([ptr[0]])

            if viz:
                viz_mask(img_fid, mask.cpu().numpy(),
                         save_name="frame_mask_%05d.png" % 0, prefix=self.debug_folder,
                         box=self.prompt)
        else:
            mask = self.temporal_track(img_fid, frame_id, viz=True)
            mask = clear_mask_scatters(mask)

            if len(self.prev_mems) > self.memory_cache_length:
                self.prev_mems = deque(islice(self.prev_mems, self.memory_cache_length))
                self.prev_ptrs = deque(islice(self.prev_ptrs, self.memory_cache_length))

        self.object_center = mask_average_location(mask, depth_fid[0],
                                                   load_3dgs_camera(cam, device=self.device))

        mask_3d = self.spatial_track(gaussians, cam, mask, mask_3d, fid=frame_id)

        pos3d, radius, height = get_area(gaussians, mask_3d, self.up_vector, ratio=1.2)
        # if pos3d is None:
        #     pos3d, radius, height = self.previous_cylinder
        #     radius = radius * 1.5
        #     viz_mask(img_fid, mask.cpu().numpy(), prefix=self.debug_folder,
        #              save_name=f"frame_debug.png")

        self.previous_mask_3d = mask_3d
        self.previous_cylinder = [pos3d, radius, height]

        if viz:
            col_1 = viz_mask(img_fid, mask.cpu().numpy(), prefix=self.debug_folder, text="frame %05d" % frame_id)
            # image_save(f"mask_{frame_id:03d}_cam00.png",
            #            mask.cpu().numpy(), prefix=self.debug_folder)
            render = render_simple(viewpoint_camera=cam, pc=gaussians,
                                   gaussian_mask=[mask_3d], mutlicolor=False,
                                   up=self.up_vector,
                                   obj_rm=True,
                                   show_box=True,
                                   show_gau=False)
            img = render["render"]
            if len(render["3d_box_proj_list"]) > 0:

                for pp in render["3d_box_proj_list"]:
                    img = draw_3d_box(img, pp, edge_color=(255, 0, 0), show_points_color=True)

            col_2 = render2img(img)
            # cv2.imwrite(osp.join(self.debug_folder, "render_bg_%05d.png" % frame_id), render2img(img))

            render = render_simple(viewpoint_camera=cam, pc=gaussians,
                                   gaussian_mask=[mask_3d], mutlicolor=False,
                                   up=self.up_vector,
                                   bg="white",
                                   render_fg_only=True,
                                   obj_rm=False,
                                   show_box=True,
                                   show_gau=False)
            img = render["render"]
            col_3 = render2img(img)
            # cv2.imwrite(osp.join(self.debug_folder, "render_fg_%05d.png" % frame_id), render2img(img))

            img_merge = np.concatenate([col_1, col_2, col_3], axis=0)
            image_save(f"merge_{self.obj_id}_{frame_id:05d}.png", img_merge, prefix=self.debug_folder)
        return mask_3d

    @torch.no_grad()
    def temporal_track(self, frame_image, frame_id, viz=False):
        h, w, _ = frame_image.shape
        img_fid_code = self.tracker.image_encoder(frame_image)
        (obj_score, best_mask_idx, preds, new_mem, new_ptr) = self.tracker.sam.step_video_masking(
            img_fid_code, self.initial_memory, self.initial_ptr, self.prev_mems, self.prev_ptrs)
        self.prev_mems.appendleft(new_mem)
        self.prev_ptrs.appendleft(new_ptr)
        hr_mask = self.tracker.make_hires_mask_uint8(
            preds[0, best_mask_idx[0]], 0.0, height=h, width=w)
        if hr_mask is None:
            print("lost!")
        return hr_mask

    @torch.no_grad()
    def spatial_track(self, gaussians, cam, mask, mask_3d=None, fid=0):
        mask_filter = MaskShapedFrustumFilter(gaussians,
                                              padding=self.padding, device=self.device,
                                              # self validation
                                              self_val=True, initial_cam=cam, initial_mask=mask)
        mask_3d = mask_filter.filtering(mask, mask_3d, cam, self_val=False, first_view_prompt=True)

        fg_set = masking_for_fg(gaussians, mask_3d)
        # render = render_simple(viewpoint_camera=cam, pc=fg_set)
        # cv2.imwrite("render.png", render2img(render["render"]))
        yaw = np.pi / 9
        pitch = np.pi / 6

        drone_0 = VirtualDroneWithMemory(drone_id=0, up_vector=self.up_vector,
                                         initial_view=cam, look_at=self.object_center,
                                         background_color=self.background_color,
                                         memory=self.initial_memory, ptr=self.initial_ptr)

        drone_1 = VirtualDroneWithMemory(drone_id=1, up_vector=self.up_vector,
                                         initial_view=cam, look_at=self.object_center,
                                         background_color=self.background_color,
                                         memory=self.initial_memory, ptr=self.initial_ptr)

        # _mask_3d = self.launch_bidirectional_drones(drone_0, drone_1, mask_filter,
        #                                             yaw, pitch, fg_set, mask_3d)

        # yaws = [np.pi / 4, np.pi / 2]
        # yaws = [np.pi / 4]
        yaws = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        for i, yaw in enumerate(yaws):
            # if i == 0:
            #     fg_set = masking_for_fg(self.gaussians, _mask_3d)
            # else:
            fg_set = masking_for_fg(gaussians, mask_3d)
            pos3d = fg_set.get_xyz.mean(dim=0).cpu()
            # drone_0.debug_verbose = True
            # drone_1.debug_verbose = True
            mask_3d = self.launch_bidirectional_drones(drone_0, drone_1, mask_filter,
                                                       yaw, pitch, fg_set, mask_3d,
                                                       lookat_point=pos3d, step_num=i + 1)
        if fid == 0:
            # 3D Memory Augmentation
            p_mem_drone_0, p_ptr_drone_0 = drone_0.prev_mems, drone_0.prev_ptrs
            p_mem_drone_1, p_ptr_drone_1 = drone_1.prev_mems, drone_1.prev_ptrs
            self.prev_mems += p_mem_drone_0
            self.prev_ptrs += p_ptr_drone_0
            self.prev_mems += p_mem_drone_1
            self.prev_ptrs += p_ptr_drone_1

        return mask_3d

    def launch_bidirectional_drones(self, drone_0, drone_1, mask_filter, yaw, pitch, fg_set, mask_3d,
                                    lookat_point=None, step_num=0):
        _mask_3d = mask_3d.clone() if mask_3d is not None else None

        cam_pose_0 = drone_0.return_camera_pose(yaw=yaw, pitch=pitch, center=lookat_point)
        cam_pose_1 = drone_1.return_camera_pose(yaw=-yaw, pitch=pitch, center=lookat_point)

        memory_pkgs = [self.initial_memory, self.initial_ptr, self.prev_mems, self.prev_ptrs]

        mask_0 = drone_0.tracking_in_new_view(fg_set, cam_pose_0, step_num=step_num, prompt_aug=self.background_aug,
                                              tracker=self.tracker, memory_pkgs=memory_pkgs)
        mask_1 = drone_1.tracking_in_new_view(fg_set, cam_pose_1, step_num=step_num, prompt_aug=self.background_aug,
                                              tracker=self.tracker, memory_pkgs=memory_pkgs)

        _mask_3d = mask_filter.filtering(mask_0, _mask_3d, cam_pose_0)
        _mask_3d = mask_filter.filtering(mask_1, _mask_3d, cam_pose_1)

        return _mask_3d


class VirtualDrone:
    def __init__(self, drone_id, initial_view, look_at, up_vector, device="cuda:0",
                 background_color=torch.tensor([0., 0., 0.])):
        self.drone_id = drone_id
        self.up_vector = torch.FloatTensor(up_vector)
        self.device = device
        ext = initial_view.world_view_transform.T
        R = ext[:3, :3]
        T = ext[:3, 3]
        C_0 = (- R.T @ T).cpu()
        cut_in_phi, cut_in_theta, cut_in_radius = inverse_get_origin(
            C_0, look_at, self.up_vector)
        self.initial_yaw = cut_in_phi

        self.fov_rad = initial_view.FoVx

        # viewpoint
        self.lookat_point = look_at
        self.distance = torch.norm(C_0 - look_at).item() * 1.0
        self.yaw = None
        self.pitch = None

        self.background_color = background_color

        # Debug
        self.debug_folder = "./debug/drone/"
        os.makedirs(self.debug_folder, exist_ok=True)
        self.debug_verbose = True
        # self.debug_verbose = False

    def return_camera_pose(self, yaw, pitch, center=None, distance=None):
        if center is None:
            center = self.lookat_point
        else:
            self.lookat_point = center  # update the lookat point
        distance = self.distance if distance is None else distance
        yaw = self.initial_yaw + yaw
        cam = sphere2params(distance, yaw, pitch, center,
                            self.up_vector, self.fov_rad, device=self.device)
        return cam

    @torch.no_grad()
    def tracking_in_new_view(self, fg_set, cam, tracker,
                             memory_pkgs,
                             step_num=0,
                             prompt_aug=True):
        memory, ptr, prev_mems, prev_ptrs = memory_pkgs
        renders = render_simple(cam, fg_set, bg="white")
        view = render2img(renders["render"])
        h, w, _ = view.shape

        img_code = tracker.image_encoder(view)

        (obj_score, best_mask_idx, preds, new_mem, new_ptr) = \
            tracker.sam.step_video_masking(
                img_code, memory, ptr, prev_mems, prev_ptrs)

        # self.prev_mems.appendleft(new_mem)
        # self.prev_ptrs.appendleft(new_ptr)

        hr_mask = tracker.make_hires_mask_uint8(
            preds[0, best_mask_idx[0]], 0.0, height=h, width=w)

        if hr_mask is None:
            print("lost!")
            return None

        return hr_mask

    @torch.no_grad()
    def prompt_augmentation(self):
        pass


class VirtualDroneWithMemory(VirtualDrone):
    def __init__(self, up_vector, initial_view, look_at,
                 drone_id=0,
                 background_color=torch.tensor([0., 0., 0.]),
                 memory=None, ptr=None, p_mem=None, p_ptr=None,
                 device="cuda:0"):
        super().__init__(drone_id, initial_view, look_at, up_vector, device, background_color)

        # memory bank
        self.memory = memory
        self.ptr = ptr
        if p_mem is None or p_ptr is None:
            if memory is None or ptr is None:
                self.prev_mems, self.prev_ptrs = deque([]), deque([])
            else:
                self.prev_mems, self.prev_ptrs = deque([memory[0]]), deque([ptr[0]])
        else:
            self.prev_mems, self.prev_ptrs = p_mem, p_ptr
        self.seg_level = 0

    @torch.no_grad()
    def tracking_in_new_view(self, fg_set, cam, tracker,
                             step_num=0,
                             prompt_aug=True, memory_pkgs=None, preview_callback=None):
        renders = render_simple(cam, fg_set, bg_color=self.background_color)
        view = render2img(renders["render"])
        h, w, _ = view.shape

        img_code = tracker.image_encoder(view)

        if memory_pkgs is not None:
            mem, ptr, prev_mems, prev_ptrs = memory_pkgs
            prev_mems = prev_mems + self.prev_mems
            prev_ptrs = prev_ptrs + self.prev_ptrs
        else:
            mem, ptr, prev_mems, prev_ptrs = self.memory, self.ptr, self.prev_mems, self.prev_ptrs

        obj_score, best_mask_idx, preds, new_mem, new_ptr = \
            tracker.sam.step_video_masking(img_code, mem, ptr,
                                           prev_mems, prev_ptrs)

        self.prev_mems.appendleft(new_mem)
        self.prev_ptrs.appendleft(new_ptr)

        hr_mask = tracker.make_hires_mask_uint8(
            preds[0, best_mask_idx[0]], 0.0, height=h, width=w)

        if hr_mask is None:
            if preview_callback is not None:
                preview_callback(view, None)
            print("lost!")
            return None

        if prompt_aug:
            aug_mask = self.prompt_augmentation(img_code, hr_mask, h, w, tracker)
            hr_mask = aug_mask if aug_mask is not None else hr_mask

        if preview_callback is not None:
            preview_callback(view, hr_mask)

        if self.debug_verbose:
            mask_save = hr_mask.clone()
            mask_save = swell_mask(mask_save, padding=5)
            mask_save = mask_save.cpu().numpy() * 255
            image_save("view_step%03d_drone%03d.png" % (step_num, self.drone_id),
                       view, prefix=self.debug_folder)
            image_save("mask_step%03d_drone%03d.png" % (step_num, self.drone_id),
                       mask_save, prefix=self.debug_folder)
            viz_mask(view, mask_save,
                     save_name="contour_step%03d_drone%03d.png" % (step_num, self.drone_id),
                     prefix=self.debug_folder, contour=True)

        return hr_mask

    @torch.no_grad()
    def prompt_augmentation(self, img_code, mask, h, w, tracker):
        box = mask2box_torch(mask, expand=3)
        if box is None: return None
        b2 = [(p[0] / w, p[1] / h) for p in box]
        prompt_code = tracker.prompt_encoder(box=[b2])

        ref_preds, ref_ptrs, obj_score = tracker.decode_mask(img_code, prompt_code)
        mask_for_mem = ref_preds[:, [0], :, :]
        with torch.inference_mode():
            ref_mem = tracker.sam.memory_encoder(img_code[0],
                                                 mask_for_mem,
                                                 obj_score,
                                                 is_prompt_encoding=True)

        self.prev_mems.appendleft(ref_mem)
        self.prev_ptrs.appendleft(ref_ptrs[:, [0]])

        hr_mask = tracker.make_hires_mask_uint8(ref_preds[0, self.seg_level],
                                                0.0,
                                                height=h, width=w)
        if self.debug_verbose:
            aug_mask = mask.cpu().numpy()
            cv2.rectangle(aug_mask, box[0], box[1], color=[255, 0, 0])
            image_save("aug_mask.png", aug_mask, prefix=self.debug_folder)
            image_save("post_aug_mask.png", hr_mask.cpu().numpy(), prefix=self.debug_folder)

        return hr_mask


@torch.no_grad()
def segmentation_simple(tracker, image, box_prompt, seg_level=0):
    h, w, _ = image.shape
    image_code = tracker.image_encoder(image)
    box = [(p[0] / w, p[1] / h) for p in box_prompt]
    prompt_code = tracker.prompt_encoder(box=[box])

    ref_preds, ref_ptrs, obj_score = tracker.decode_mask(image_code, prompt_code)

    ref_mem = tracker.sam.memory_encoder(image_code[0], ref_preds[:, [0], :, :],
                                         obj_score, is_prompt_encoding=True)

    memory = tuple([ref_mem])
    ptr = tuple([ref_ptrs[:, [0]]])

    hr = tracker.make_hires_mask_uint8(ref_preds[0, seg_level],
                                       0.0,
                                       height=h, width=w)

    return hr, memory, ptr


class TemporalTrackerWrapper:
    def __init__(self, first_image, prompt, tracker, device="cuda:0"):
        mask, memory, ptr = segmentation_simple(tracker, first_image, prompt, seg_level=0)
        self.initial_mask = mask
        self.tracker = tracker
        self.memory = memory
        self.ptr = ptr
        self.device = device
        self.prev_mems, self.prev_ptrs = deque([memory[0]]), deque([ptr[0]])

    def get_memory_pkg(self):
        return self.memory, self.ptr, self.prev_mems, self.prev_ptrs

    def get_initial_mask(self):
        return self.initial_mask

    @torch.no_grad()
    def track(self, img):
        h, w, _ = img.shape
        img_code = self.tracker.image_encoder(img)
        (obj_score, best_mask_idx,
         preds, new_mem, new_ptr) = self.tracker.sam.step_video_masking(
            img_code, self.memory, self.ptr,
            self.prev_mems, self.prev_ptrs)

        self.prev_mems.appendleft(new_mem)
        self.prev_ptrs.appendleft(new_ptr)

        hr_mask = self.tracker.make_hires_mask_uint8(
            preds[0, best_mask_idx[0]], 0.0, height=h, width=w)

        if hr_mask is None:
            print("lost!")

        return hr_mask
