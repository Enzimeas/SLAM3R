import torch
import cv2
import numpy as np
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
import trimesh

from slam3r.utils.device import to_numpy
from slam3r.inference import collate_with_cat, to_cpu,loss_of_one_batch_multiview, \
                                inv, geotrf, get_multiview_scale
from slam3r.utils.geometry import xy_grid
from slam3r.utils.geometry import opencv_to_colmap_intrinsics

try:
    import poselib  # noqa
    HAS_POSELIB = True
except Exception as e:
    HAS_POSELIB = False


def save_traj(views, pred_frame_num, save_dir, scene_id, args, 
              intrinsics = None, traj_name = 'traj'): 
    save_name = f"{scene_id}_{traj_name}.txt"

    c2ws = []
    H, W, _ = views[0]['pts3d_world'][0].shape
    for i in tqdm(range(pred_frame_num)):
        pts = to_numpy(views[i]['pts3d_world'][0])
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        dist_coeffs = np.zeros(4).astype(np.float32)
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            pts.reshape(-1, 3).astype(np.float32), 
            points_2d.reshape(-1, 2).astype(np.float32), 
            intrinsics[i].astype(np.float32), 
            dist_coeffs)
    
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        # Extrinsic parameters (4x4 matrix)
        extrinsic_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
        c2w = inv(extrinsic_matrix)
        c2ws.append(c2w)
    c2ws = np.stack(c2ws, axis=0)
    translations = c2ws[:,:3,3]
    # draw the trajectory in horizontal plane
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_traj(ax, [i for i in range(len(translations))], translations,
                '-', "black", "estimate trajectory")
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(join(save_dir, save_name.replace('.txt', '.png')), dpi=90)
    np.savetxt(join(save_dir, save_name), c2ws.reshape(-1,16))

def plot_traj(ax, stamps, traj, style, color, label):
    """
    Plot a trajectory using matplotlib. 
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    """
    stamps.sort()
    interval = np.median([s-t for s, t in zip(stamps[1:], stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x) > 0:
            ax.plot(x, y, style, color=color, label=label)
            label = ""
            x = []
            y = []
        last = stamps[i]
    if len(x) > 0:
        ax.plot(x, y, style, color=color, label=label)

def estimate_intrinsics(pts3d_local):
    ##### estimate focal length
    _, H, W, _ = pts3d_local.shape
    pp = torch.tensor((W/2, H/2))
    focal = estimate_focal_knowing_depth(pts3d_local.cpu(), pp, focal_mode='weiszfeld')
    # print(f'Estimated focal of first camera: {focal.item()} (224x224)')
    intrinsic = np.eye(3)
    intrinsic[0, 0] = focal
    intrinsic[1, 1] = focal
    intrinsic[:2, 2] = pp
    return intrinsic

def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0., max_focal=np.inf):
    """ Reprojection method, for when the absolute depth is known:
        1) estimate the camera focal using a robust estimator
        2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'bad {focal_mode=}')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal


def unsqueeze_view(view):
    """Uunsqueeze view to batch size 1, 
    similar to collate_fn
    """
    if len(view['img'].shape) > 3:
        return view
    res = dict(img=view['img'][None], 
                 true_shape=view['true_shape'][None], 
                 idx=view['idx'], 
                 instance=view['instance'], 
                 pts3d_cam=torch.tensor(view['pts3d_cam'][None]),
                 valid_mask=torch.tensor(view['valid_mask'][None]),
                 camera_pose=torch.tensor(view['camera_pose']),
                 pts3d=torch.tensor(view['pts3d'][None])
                )
    if 'pointmap_img' in view:
        res['pointmap_img'] = view['pointmap_img'][None]
    
    return res

def transform_img(view):
    #transform to numpy, BGR, 0-255, HWC
    img = view['img'][0]
    # print(img.shape)
    img = img.permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img/2.+0.5)*255.
    return img

@torch.no_grad()
def my_inference_batch(batch_views:list, model, device='cuda', 
                       ref_id=0, 
                       tocpu=True, 
                       unsqueeze=True):
    """inference with batching
    batch_views: list of list, [[view1, view2, ...], [view1, view2, ...], ...]
                                     batch1                 batch2       ...
    """
    pairs = []
    for views in batch_views:
        if unsqueeze:
            pairs.append(tuple(unsqueeze_view(view) for view in views))
        else:
            pairs.append(tuple(views))

    input = collate_with_cat(pairs)
    res = loss_of_one_batch_multiview(input, model, None, device, ref_id=ref_id)
    result = [to_cpu(res)] if tocpu else [res]
    output = collate_with_cat(result)   #views,preds,loss,view1,..pred1...
    return output

def save_ply(points:np.array, save_path, colors:np.array=None, metadata:dict=None):
    #color:0-1
    if np.max(colors) > 1:
        colors = colors/255.
    pcd = trimesh.points.PointCloud(points, colors=colors)
    if metadata is not None:
        for key in metadata:
            pcd.metadata[key] = metadata[key]
    pcd.export(save_path)
    print(">> save_to ", save_path)

def save_vis(points, dis, vis_path):
    cmap = plt.get_cmap('Reds')
    color = cmap(dis/0.05)
    save_ply(points=points, save_path=vis_path, colors=color)

def uni_upsample(img,scale):
    img = np.array(img)
    upsampled_img = img[:,None,:,None].repeat(scale,1).repeat(scale,3).reshape(img.shape[0]*scale,-1)
    return upsampled_img

def get_free_gpu():
    # initialize PyCUDA
    try:
        import pycuda.driver as cuda
    except ImportError as e:
        print(f"{e} -- fail to import pycuda, choose GPU 0.")
        return 0
    
    cuda.init()
    device_count = cuda.Device.count()
    most_free_mem = 0
    most_free_id = 0
    for i in range(device_count):
        try:
            device = cuda.Device(i)
            context = device.make_context()
            # query the free memory on the device
            free_memory = cuda.mem_get_info()[0]
            if(free_memory > most_free_mem):
                most_free_mem = free_memory
                most_free_id = i
            
            context.pop()
            if(free_memory > 23e9):
                return i
        except:
            pass
    print("No totally free GPU found! Choose the most free one.")

    return most_free_id