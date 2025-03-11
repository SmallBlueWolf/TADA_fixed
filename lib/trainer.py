import glob
import random
import tqdm
import imageio
import tensorboardX
import numpy as np
import time
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from pytorch3d.ops.knn import knn_points
# from pytorch3d.loss import chamfer_distance
import torch.distributed as dist
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from lib.common.utils import *
from lib.common.visual import draw_landmarks, draw_mediapipe_landmarks
from lib.dpt import DepthNormalEstimation

# 在train_one_step中调用

class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 text, negative, dir_text,
                 opt,  # extra conf
                 model,  # network
                 guidance,  # guidance network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 max_keep_ckpt=2,  # max num of saved ckpts in disk
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.dpt = DepthNormalEstimation(use_depth=False) if opt.use_dpt else None
        self.default_view_data = None
        self.name = name
        self.text = text
        self.negative = negative
        self.dir_text = dir_text
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size

        self.workspace = os.path.join(opt.workspace, self.name, self.text)
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = opt.eval_interval
        self.use_checkpoint = opt.ckpt
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        self.model = model.to(self.device)
        
        '''自定义一些lora的超参在这里'''
        self.unet_lr = 0.0001
        self.unet_bs = 1
        self.warm_iters = 500
        self.K = 1
        self.K2 = 1
        
        
        
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1, 2, 3]).module
        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])

        # guide model
        self.guidance = guidance
        # self.guidance = torch.nn.DataParallel(self.guidance, device_ids=[0, 1, 2, 3]).module
        # text prompt
        self.text_embeds = None
        if self.guidance is not None:
            for p in self.guidance.parameters():
                p.requires_grad = False
            self.prepare_text_embeddings()

        # try out torch 2.0
        if torch.__version__[0] == '2':
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        '''初始化定义q_uet，使用lora'''
        if True: # 使用lora
            from .lora_unet import UNet2DConditionModel
            # from diffusers import UNet2DConditionModel
            from diffusers.loaders import AttnProcsLayers
            from diffusers.models.attention_processor import LoRAAttnProcessor
            import einops
            if True: # 使用v_pred
                _unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
                # 原版使用 stabilityai/stable-diffusion-2-1-base，也即不适用v_pred方法
                _unet.requires_grad_(False)
                lora_attn_procs = {}
                for name in _unet.attn_processors.keys():
                    cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                    if name.startswith("mid_block"):
                        hidden_size = _unet.config.block_out_channels[-1]
                    elif name.startswith("up_blocks"):
                        block_id = int(name[len("up_blocks.")])
                        hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                    elif name.startswith("down_blocks"):
                        block_id = int(name[len("down_blocks.")])
                        hidden_size = _unet.config.block_out_channels[block_id]
                    lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                _unet.set_attn_processor(lora_attn_procs)
                lora_layers = AttnProcsLayers(_unet.attn_processors)
            '''这里额外设置了_uet各个层注意力机制的模块'''
            '''这部分直接复制过来'''
            text_input = self.guidance.tokenizer(self.text, padding='max_length', max_length=self.guidance.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings = self.guidance.text_encoder(text_input.input_ids.to(self.guidance.device))[0]
            class LoraUnet(torch.nn.Module):
                def __init__(self, device):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = 64
                    self.in_channels = 4
                    self.device = device
                    self.dtype = torch.float32
                    self.text_embeddings = text_embeddings
                def forward(self,x,t,c=None,shading="albedo"):
                    textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
            self._unet = _unet
            self.lora_layers = lora_layers
            self.unet = LoraUnet(device=self.device).to(device)                     

        self.unet = self.unet.to(self.device)
        self.q_unet = self.unet
        if True: # use lora
            params = [
                {'params': self.lora_layers.parameters()},
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.lambertian_emb},
                {'params': self._unet.textureless_emb},
                {'params': self._unet.normal_emb},
            ] 
        self.unet_optimizer = optim.AdamW(params, lr=self.unet_lr) # naive adam
        warm_up_lr_unet = lambda iter: iter / (self.warm_iters*self.K+1) if iter <= (self.warm_iters*self.K+1) else 1
        self.unet_scheduler = optim.lr_scheduler.LambdaLR(self.unet_optimizer, warm_up_lr_unet)
        '''这部分直接复制过来'''

        print("[bwINFO] Finish Init q_unet(lora)")

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(self.workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

        # 初始化buffer
        self.buffer_imgs = None
        self.buffer_poses = None
        
        print("[bwINFO] Finish Trainer Init!")
        
    def add_buffer(self, latents, pose):
        """将 latents 和 pose 添加到缓冲区"""
        if not hasattr(self, 'buffer_imgs') or self.buffer_imgs is None:
            self.buffer_imgs = latents.detach()
            self.buffer_poses = pose.detach()
        else:
            self.buffer_imgs = torch.cat([self.buffer_imgs, latents.detach()], dim=0)[-self.opt.buffer_size:]
            self.buffer_poses = torch.cat([self.buffer_poses, pose.detach()], dim=0)[-self.opt.buffer_size:]

    def sample_buffer(self, batch_size):
        """从缓冲区采样 batch_size 个 latents 和 pose"""
        if self.buffer_imgs.shape[0] < batch_size:
            idx = torch.arange(self.buffer_imgs.shape[0], device=self.device)
        else:
            idx = torch.randperm(self.buffer_imgs.shape[0], device=self.device)[:batch_size]
        return self.buffer_imgs[idx], self.buffer_poses[idx]    
    # calculate the text embeddings.
    def prepare_text_embeddings(self):
        if self.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            return

        self.text_embeds = {
            'uncond': self.guidance.get_text_embeds([self.negative]),
            'default': self.guidance.get_text_embeds([f"a 3D rendering of {self.text}, full-body"]),
        }

        if self.opt.train_face_ratio < 1:
            self.text_embeds['body'] = {
                d: self.guidance.get_text_embeds([f"a {d} view 3D rendering of {self.text}, full-body"])
                for d in ['front', 'side', 'back', "overhead"]
            }

        if self.opt.train_face_ratio > 0:
            id_text = self.text.split("wearing")[0]
            self.text_embeds['face'] = {
                d: self.guidance.get_text_embeds([f"a {d} view 3D rendering of {id_text}, face"])
                for d in ['front', 'side', 'back']
            }

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train_step(self, data, is_full_body=True):
        """
        根据输入数据执行训练步骤，计算损失并返回必要的信息
        
        Args:
            data: 包含训练数据的字典
            is_full_body: 布尔值，指示是否使用全身数据
        
        Returns:
            pred: 渲染的RGB图像
            loss: 计算的损失值
            pseudo_loss: 伪损失值，用于监控
            latents: 生成的潜在表示
            shading: 使用的着色模式
        """
        do_rgbd_loss = self.default_view_data is not None and (self.global_step % self.opt.known_view_interval == 0)

        if do_rgbd_loss:
            data = self.default_view_data

        H, W = data['H'], data['W']
        mvp = data['mvp']  # [B, 4, 4]
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        B, N = rays_o.shape[:2]

        # 根据训练进度进行分辨率渐进式提升
        if self.opt.anneal_tex_reso:
            scale = min(1, self.global_step / (0.8 * self.opt.iters))

            def make_divisible(x, y): return x + (y - x % y)

            H = max(make_divisible(int(H * scale), 16), 32)
            W = max(make_divisible(int(W * scale), 16), 32)

        # 对已知视图添加噪声以增强泛化能力
        if do_rgbd_loss and self.opt.known_view_noise_scale > 0:
            noise_scale = self.opt.known_view_noise_scale
            rays_o = rays_o + torch.randn(3, device=self.device) * noise_scale
            rays_d = rays_d + torch.randn(3, device=self.device) * noise_scale

        # 准备文本嵌入
        dir_text_z = [self.text_embeds['uncond'], self.text_embeds[data['camera_type'][0]][data['dirkey'][0]]]
        dir_text_z = torch.cat(dir_text_z)

        # 执行第一次渲染，获取原始分辨率的输出
        out = self.model(rays_o, rays_d, mvp, data['H'], data['W'], shading='albedo')
        image = out['image'].permute(0, 3, 1, 2)
        normal = out['normal'].permute(0, 3, 1, 2)
        alpha = out['alpha'].permute(0, 3, 1, 2)

        # 执行第二次渲染，获取可能缩放后的输出
        out_annel = self.model(rays_o, rays_d, mvp, H, W, shading='albedo')
        image_annel = out_annel['image'].permute(0, 3, 1, 2)
        normal_annel = out_annel['normal'].permute(0, 3, 1, 2)
        alpha_annel = out_annel['alpha'].permute(0, 3, 1, 2)

        # 将图像和法线合并用于可视化
        pred = torch.cat([out['image'], out['normal']], dim=2)
        pred = (pred[0].detach().cpu().numpy() * 255).astype(np.uint8)

        # 计算当前训练进度
        p_iter = self.global_step / self.opt.iters

        # 选择着色模式
        # 随机决定使用哪种着色模式 - 主要是为了训练多个着色样式的能力
        if self.global_step < self.opt.albedo_iters+1:
            shading = 'albedo'
        else: 
            rand = random.random()
            if rand > 0.8: 
                shading = 'albedo'
            elif rand > 0.4 and (not self.opt.no_textureless): 
                shading = 'textureless'
            else: 
                if not self.opt.no_lambertian:
                    shading = 'lambertian'
                else:
                    shading = 'albedo'
                    
        if self.opt.normal:
            shading = 'normal'
            if self.opt.p_textureless > random.random():
                shading = 'textureless'
                    
        if self.global_step < self.opt.normal_iters+1:
            as_latent = True
            shading = 'normal'
        else:
            as_latent = False

        # 使用RGB损失或SDS损失
        if do_rgbd_loss:  # 使用RGB损失（当有真实图像参考时）
            gt_rgb = data['rgb']  # [B, 3, H, W]
            gt_normal = data['normal']  # [B, H, W, 3]
            gt_depth = data['depth']  # [B, H, W]
            
            # RGB损失
            loss = self.opt.lambda_rgb * F.mse_loss(image, gt_rgb)
            
            # 法线损失
            if self.opt.lambda_normal > 0:
                lambda_normal = self.opt.lambda_normal * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_normal * (1 - F.cosine_similarity(normal, gt_normal).mean())
                
            # 深度损失
            if self.opt.lambda_depth > 0:
                lambda_depth = self.opt.lambda_depth * min(1, self.global_step / self.opt.iters)
                loss = loss + lambda_depth * (1 - self.pearson(depth, gt_depth))
        else:
            # 使用VDS/SDS损失 (根据是否提供q_unet决定)
            # 将生成的图像送入guidance网络计算损失
            
            # VDS方法: 使用self.q_unet进行变分扩散得分估计
            # 在 trainer.py 的 train_step 或类似方法中
            pose = data['poses'].view(B, 16)
            
            loss, pseudo_loss, latents = self.guidance.train_step(
                text_embeddings=dir_text_z,  # 文本嵌入，通常由 guidance.get_text_embeds 生成
                pred_rgb=image_annel,  # 渲染图像或潜在表示
                guidance_scale=100,  # 引导尺度
                q_unet=self.q_unet,  # 分数估计网络
                pose=pose,  # 可选的姿势条件
                shading=shading if 'shading' in data else 'albedo'  # 可选的着色条件
            )
            print(f'[INFO] Finish calculating loss!')
            
            # # 额外的法线引导损失
            # if not self.dpt:  # 如果没有深度预测网络
            #     # 添加法线的guidance损失
            #     normal_loss, _, _ = self.guidance.train_step(
            #         dir_text_z, 
            #         normal, 
            #         guidance_scale=self.opt.scale, 
            #         q_unet=self.q_unet,
            #         pose=data['pose'].view(data['pose'].shape[0], 16) if 'pose' in data else None,
            #         shading=shading
            #     )
            #     loss += normal_loss.mean()
            # else:  # 如果有深度预测网络
            #     # 在训练早期或随机情况下使用法线引导
            #     if p_iter < 0.3 or random.random() < 0.5:
            #         normal_loss, _, _ = self.guidance.train_step(
            #             dir_text_z, 
            #             normal, 
            #             guidance_scale=self.opt.scale, 
            #             q_unet=self.q_unet,
            #             pose=data['pose'].view(data['pose'].shape[0], 16) if 'pose' in data else None,
            #             shading=shading
            #         )
            #         loss += normal_loss.mean()
            #     elif self.dpt is not None:
            #         # 使用深度估计网络生成法线并计算损失
            #         dpt_normal = self.dpt(image)
            #         dpt_normal = (1 - dpt_normal) * alpha + (1 - alpha)
            #         lambda_normal = self.opt.lambda_normal * min(1, self.global_step / self.opt.iters)
            #         loss += lambda_normal * (1 - F.cosine_similarity(normal, dpt_normal).mean())

        # # 添加额外的正则化项
        # if not self.opt.dmtet:
        #     # 不透明度正则化
        #     if self.opt.lambda_opacity > 0:
        #         loss_opacity = (out['weights_sum'] ** 2).mean()
        #         loss = loss + self.opt.lambda_opacity * loss_opacity

        #     # 熵正则化
        #     if self.opt.lambda_entropy > 0:
        #         alphas = out['weights'].clamp(1e-5, 1 - 1e-5)
        #         loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()
        #         loss = loss + self.opt.lambda_entropy * loss_entropy

        #     # 方向正则化
        #     if self.opt.lambda_orient > 0 and 'loss_orient' in out:
        #         loss_orient = out['loss_orient']
        #         loss = loss + self.opt.lambda_orient * loss_orient
        # else:
        #     # DMTet特有的正则化
        #     if self.opt.lambda_normal > 0:
        #         loss = loss + self.opt.lambda_normal * out['normal_loss']

        #     if self.opt.lambda_lap > 0:
        #         loss = loss + self.opt.lambda_lap * out['lap_loss']

        return pred, loss, pseudo_loss, latents, shading

    def eval_step(self, data):
        H, W = data['H'].item(), data['W'].item()
        mvp = data['mvp']
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading='albedo', is_train=False)
        w = out['normal'].shape[2]
        pred = torch.cat([out['normal'], out['image'],
                          torch.cat([out['normal'][:, :, :w // 2], out['image'][:, :, w // 2:]], dim=2)], dim=1)

        # dummy 
        loss = torch.zeros([1], device=pred.device, dtype=pred.dtype)

        return pred, loss

    def test_step(self, data):
        H, W = data['H'].item(), data['W'].item()
        mvp = data['mvp']
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        out = self.model(rays_o, rays_d, mvp, H, W, shading='albedo', is_train=False)
        w = out['normal'].shape[2]
        pred = torch.cat([out['normal'], out['image'],
                          torch.cat([out['normal'][:, :, :w // 2], out['image'][:, :, w // 2:]], dim=2)], dim=2)

        return pred, None

    def save_mesh(self, save_path=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, "mesh")

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path)

        self.log(f"==> Finished saving mesh.")

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            # 随机决定，使用面部数据或使用全身数据
            with torch.no_grad():
                # 使用面部数据
                if random.random() < self.opt.train_face_ratio:
                    train_loader.dataset.full_body = False
                    face_center, face_scale = self.model.get_mesh_center_scale("face")
                    train_loader.dataset.face_center = face_center
                    train_loader.dataset.face_scale = face_scale.item() * 10

                else:
                    # 使用全身数据
                    train_loader.dataset.full_body = True
                    # body_center, body_scale = self.model.get_mesh_center_scale("body")
                    # train_loader.dataset.body_center = body_center
                    # train_loader.dataset.body_scale = body_scale.item()

            self.train_one_epoch(train_loader)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                else:
                    os.makedirs(os.path.join(save_path, "image"), exist_ok=True)
                    cv2.imwrite(os.path.join(save_path, "image", f'{i:04d}.png'),
                                cv2.cvtColor(pred[..., :3], cv2.COLOR_RGB2BGRA))

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}.mp4'), all_preds, fps=25, quality=9,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        """
        执行一个训练周期
        
        Args:
            loader: 数据加载器，提供训练数据
        """
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        # 用于分布式训练中，确保主程序报告所有的指标
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # 分布式训练设置: 在每个epoch开始时设置采样器，确保数据shuffle
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        # 设置进度条 (仅在主进程)
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # 重置本地步数计数器
        self.local_step = 0

        for data in loader:
            # 随机选择当前变分分布的粒子
            self.model.set_idx()

            # # 如果启用了CUDA光线追踪，定期更新密度网格
            # if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
            #     with torch.cuda.amp.autocast(enabled=self.fp16):
            #         self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            # 清空优化器梯度
            self.optimizer.zero_grad()

            # 执行训练步骤
            with torch.amp.autocast('cuda', enabled=self.fp16):
                pred_rgbs, loss, pseudo_loss, latents, shading = self.train_step(data, loader.dataset.full_body)

            # 保存训练过程的可视化结果
            if self.global_step % 20 == 0:
                pred = cv2.cvtColor(pred_rgbs, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(self.workspace, 'train-vis', f'{self.name}/{self.global_step:04d}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, pred)

            # 反向传播
            self.scaler.scale(loss).backward()
            # 执行训练后的步骤（如TV正则化）
            # self.post_train_step()
            # 更新模型参数
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新学习率调度器
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            # VDS特有：将生成的latents添加到buffer中
            if hasattr(self, 'opt') and hasattr(self.opt, 'buffer_size') and self.opt.buffer_size != -1:
                self.add_buffer(latents, data['poses'])

            # VDS特有：训练q_unet模型
            if hasattr(self, 'q_unet') and hasattr(self, 'opt') and hasattr(self.opt, 'K2'):
                # 每K2步训练一次q_unet，且不使用SDS方法时
                if self.global_step % self.opt.K2 == 0 and not (hasattr(self.opt, 'sds') and self.opt.sds):
                    # 执行K次q_unet的训练
                    for _ in range(self.opt.K):
                        # 清空q_unet优化器梯度
                        self.unet_optimizer.zero_grad()
                        
                        # 随机选择时间步长
                        timesteps = torch.randint(0, 1000, (self.opt.unet_bs,), device=self.device).long()
                        
                        # 准备训练数据
                        with torch.no_grad():
                            # 如果buffer未满，使用当前batch的latents扩展
                            if not hasattr(self, 'buffer_imgs') or self.buffer_imgs is None or self.buffer_imgs.shape[0] < self.opt.buffer_size:
                                latents_clean = latents.expand(self.opt.unet_bs, latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                                if hasattr(self.opt, 'q_cond') and self.opt.q_cond:
                                    # 准备姿态条件
                                    pose = data['poses']
                                    pose = pose.view(pose.shape[0], 16)
                                    pose = pose.expand(self.opt.unet_bs, 16).contiguous()
                                    # 一定概率使用无条件训练
                                    if random.random() < self.opt.uncond_p:
                                        pose = torch.zeros_like(pose)
                            else:
                                # 从buffer中采样latents和对应的pose
                                latents_clean, pose = self.sample_buffer(self.opt.unet_bs)
                                # 一定概率使用无条件训练
                                if random.random() < self.opt.uncond_p:
                                    pose = torch.zeros_like(pose)
                        
                        # 添加噪声到latents
                        noise = torch.randn(latents_clean.shape, device=self.device)
                        latents_noisy = self.guidance.scheduler.add_noise(latents_clean, noise, timesteps)
                        
                        # 使用q_unet生成预测
                        if hasattr(self.opt, 'q_cond') and self.opt.q_cond:
                            model_output = self.q_unet(latents_noisy, timesteps, c=pose, shading=shading).sample
                        else:
                            model_output = self.q_unet(latents_noisy, timesteps).sample
                        
                        # 计算q_unet的损失
                        if hasattr(self.opt, 'v_pred') and self.opt.v_pred:
                            # v-prediction模式
                            loss_unet = F.mse_loss(model_output, self.guidance.scheduler.get_velocity(latents_clean, noise, timesteps))
                        else:
                            # ε-prediction模式
                            loss_unet = F.mse_loss(model_output, noise)
                        
                        # 反向传播和优化
                        loss_unet.backward()
                        self.unet_optimizer.step()
                        
                        # 更新q_unet的学习率调度器
                        if self.scheduler_update_every_step:
                            self.unet_scheduler.step()

            # 记录训练进度和指标（仅在主进程）
            if self.local_rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(
                        f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f}), "
                        f"lr={self.optimizer.param_groups[0]['lr']:.6f}, ")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        # 更新EMA模型
        if self.ema is not None:
            self.ema.update()

        # 计算平均损失并记录
        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        # 关闭进度条和报告训练指标（仅在主进程）
        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        # 更新学习率调度器（如果不是每步更新）
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        vis_frames = []
        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                # with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
                    pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
                    vis_frames.append(pred)
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        save_path = os.path.join(self.workspace, 'validation', f'{name}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, np.hstack(vis_frames))

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
