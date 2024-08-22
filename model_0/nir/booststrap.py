import os
import numpy as np
from itertools import chain
import cv2
import torch
from torch.utils.data import DataLoader
import imageio
import argparse
from model import Siren
from util import get_mgrid, jacobian, VideoFitting

def train_fence(path, total_steps, lambda_interf=0.5, lambda_flow=0.5, verbose=True, steps_til_summary=100):
    g = Siren(in_features=3, out_features=2, hidden_features=256,
              hidden_layers=4, outermost_linear=True)
    g.cuda()
    f1 = Siren(in_features=2, out_features=3, hidden_features=256,
               hidden_layers=4, outermost_linear=True, first_omega_0=90.)
    f1.cuda()
    f2 = Siren(in_features=3, out_features=4, hidden_features=256, 
               hidden_layers=4, outermost_linear=True)
    f2.cuda()

    optim = torch.optim.Adam(lr=1e-4, params=chain(g.parameters(), f1.parameters(), f2.parameters()))

    v = VideoFitting(path)
    videoloader = DataLoader(v, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(videoloader))
    model_input, ground_truth = model_input[0].cuda(), ground_truth[0].cuda()

    batch_size = (v.H * v.W) // 8
    for step in range(total_steps):
        start = (step * batch_size) % len(model_input)
        end = min(start + batch_size, len(model_input))

        xyt = model_input[start:end].requires_grad_()
        xy, t = xyt[:, :-1], xyt[:, [-1]]
        h = g(xyt)
        xy_ = xy + h
        o_scene = torch.sigmoid(f1(xy_))
        o_obst = torch.sigmoid(f2(xyt))
        o_obst, alpha = o_obst[:, :-1], o_obst[:, [-1]]
        o = (1 - alpha) * o_scene + alpha * o_obst
        loss_recon = ((o - ground_truth[start:end]) ** 2).mean()
        loss_interf = alpha.abs().mean()
        loss_flow = jacobian(h, xyt).abs().mean()
        loss = loss_recon + lambda_interf * loss_interf + lambda_flow * loss_flow
        if not step % steps_til_summary:
            print("Step [%04d/%04d]: recon=%0.8f, interf=%0.4f, flow=%0.4f" % (step, total_steps, loss_recon, loss_interf, loss_flow))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return g, f1, f2, v.video
def channel_stack(data):
    parent_path = data[:9]
    print(parent_path)
    folder_path = os.path.join("xca_dataset",parent_path,'images',data)
    count = 0
    for filename in sorted(os.listdir(folder_path)):
        f = os.path.join(folder_path,filename)
        gray_image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        # print(sorted(os.listdir(folder_path)))
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        path = os.path.join("nir/nir_image",data,filename)
        cv2.imwrite(path,rgb_image)
        count +=1
        if count > 13:
            break
    return True
def main(args):
    data = args.data
    base_path = "nir/nir_image"
    path = os.path.join(base_path,data)
    print(path)
    os.makedirs(path,exist_ok=True)
    stack_flag = channel_stack(data)
    g, f1, f2, orig = train_fence(path, 3000)
    with torch.no_grad():
        N, _, H, W = orig.size()
        xyt = get_mgrid([H, W, N]).cuda()
        h = g(xyt)
        o_scene = torch.sigmoid(f1(xyt[:, :-1] + h))
        o_obst = torch.sigmoid(f2(xyt))
        o_obst = o_obst[:, :-1] * o_obst[:, [-1]]
        o_scene = o_scene.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_obst = o_obst.view(H, W, N, 3).permute(2, 0, 1, 3).cpu().detach().numpy()
        o_scene = (o_scene * 255).astype(np.uint8)
        o_obst = (o_obst * 255).astype(np.uint8)
        o_scene = [o_scene[i] for i in range(len(o_scene))]
        o_obst = [o_obst[i] for i in range(len(o_obst))]
        orig = orig.permute(0, 2, 3, 1).detach().numpy()
        orig = (orig * 255).astype(np.uint8)
        orig = [orig[i] for i in range(len(orig))]
    p = os.path.join("nirs",data)
    os.makedirs(p,exist_ok=True)
    name =os.path.join(p,"scene.png")
    cv2.imwrite(name,o_scene[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data")
    args = parser.parse_args()
    main(args)