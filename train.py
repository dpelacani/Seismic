import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm

import wandb
from torchsummary import summary

from seismic import build_model

sys.path.append("/home/dp4018/scripts/InverseLDM/")
from invldm.datasets.brain2d_dataset import Brain2DDataset
from invldm.utils.visualisation import OSCAR_CMAP, visualise_samples
from invldm.utils.utils import namespace2dict, dict2namespace, scale2range
from invldm.utils.setup import set_seed

set_seed(42)

DEVICE="cuda:0"

if __name__ == "__main__":

    dataset_args = dict2namespace(dict(
        data_path="/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-models/",
        mode="vp",
        maxsamples=None,
        slowness=False,
        resize=[256, 256],
        scale=[0., 1.],
        clip_outliers="outer",
        to_tensor=False,
        # normalise=(0.10280502008738208, 0.22698244373387108),
        normalise=False,
        antialias=True,
        condition = dict(
            # mode="stack-stack-pca-single-256x256",
            # path="/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/stack_pca_single/data",
            mode="stack-512x512",
            path="/home/dp4018/data/ultrasound-data/Ultrasound-Vp-sagittal-data/acoustic/data/stack_resized/",
            scale=[0., 1.],
            clip_outliers="outer",
            to_tensor=False,
            normalise=False,
            antialias=True,

        ),
        sampling_only=False,
        batch_size=8,
    ))

    dataset = Brain2DDataset(dataset_args)
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_args.batch_size, sampler=sampler)
    img, data = next(iter(dataloader))

    embed_dim = 3
    embed_size = 32
    images_resolution = img.shape[2]
    images_channels = 1
    data_resolution = data.shape[2]
    data_channels = 1
    vision_layers = 2
    vision_width = 64
    vision_patch_size = 3
    mixed_precision = False

    model = build_model(
        embed_dim,
        embed_size,
        images_resolution,
        images_channels,
        data_resolution,
        data_channels,
        vision_layers,
        vision_width,
        vision_patch_size,
        mixed_precision
    ).to(DEVICE)

    _ = summary(model)

    img_f, data_f, img_logit, data_logit = model(img.to(DEVICE), data.to(DEVICE))

    print("\n================ LOGITS ==================")
    print(img_logit.shape, data_logit.shape)
    print(img_logit.min().item(), img_logit.max().item(), img_logit.mean().item())
    print(data_logit.min().item(), data_logit.max().item(), data_logit.mean().item())

    print("\n================ FEATURES ================")
    print(img_f.shape, data_f.shape)
    print(img_f.min().item(), img_f.max().item(), img_f.mean().item())
    print(data_f.min().item(), data_f.max().item(), data_f.mean().item())

    del img_f, data_f, img_logit, data_logit

    save_freq = 500


    n_epochs = 450
    warm_restart_steps = 2000
    bias_weight_decay = False
    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9,0.98), eps=1e-6)#, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, warm_restart_steps)
    scheduler = None
    losses = []

    # Initialise Weights and Biases and store hyperparameters
    wandb_run = wandb.init(
        project="SeismicClip",
        save_code=True,
        group="hyper_tune",
        dir="./exps",
        config={
            "data":namespace2dict(dataset_args),
            "training":{
                "epochs": n_epochs,
            },
            "model": {
                "embed_dim": embed_dim,
                "embed_size": embed_size,
                "images_resolution": images_resolution,
                "images_channels": images_channels,
                "data_resolution": data_resolution,
                "data_channels": data_channels,
                "vision_layers": vision_layers,
                "vision_width": vision_width,
                "vision_patch_size": vision_patch_size,
                "mixed_precision": mixed_precision,
                # "comment": "original clip parameters (some scaling by 4 given the data dimension)",
                "loss_fn": str(loss_fn)
            },
            "optim":{
                "optimiser": {"name":optimiser.__class__.__name__,
                            "betas": optimiser.defaults["betas"],
                            "weight_decay": optimiser.defaults["weight_decay"],
                            "eps": optimiser.defaults["eps"]},
                "bias_weight_decay": bias_weight_decay if optimiser.defaults["weight_decay"] > 0 else "N/A",
                "lr": optimiser.defaults["lr"],
                "lr_sched": scheduler,
                "warm_restart_steps": warm_restart_steps,
            },
        },
    )

    # Create exp folder and save config
    os.mkdir(f"./exps/{wandb_run.name}")
    os.mkdir(f"./exps/{wandb_run.name}/embeddings")
    with open(os.path.join(f"./exps/{wandb_run.name}", "config_training.yml"), "w") as f:
        yaml.dump(dict(wandb_run.config.items()), f, default_flow_style=False, sort_keys=False, indent=4)


    # Change weight decay of biases to zero if prompted
    if not bias_weight_decay and optimiser.defaults["weight_decay"] > 0:
        decay_params, no_decay_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue           
            elif len(param.shape) == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else: decay_params.append(param)
        params = [{'params': no_decay_params, 'weight_decay': 0.}, {'params': decay_params, 'weight_decay': optimiser.defaults["weight_decay"]}]        
    else:
        params=model.parameters()
    optimiser = torch.optim.Adam(params, **optimiser.defaults)

    # Step count
    step = 0

    # Train
    try:
        for epoch in tqdm(range(n_epochs)):
            for img, data in dataloader:
                with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu"):
                    # Get logits
                    _, _, logit_im, logit_dt = model(img.to(DEVICE), data.to(DEVICE))

                    # Compute  loss
                    targets = torch.arange(img.shape[0]).to(DEVICE)
                    loss = 0.5*(loss_fn(logit_im, targets) + loss_fn(logit_dt, targets))
                    
                # Store loss
                losses.append(loss.item())

                # Log wandb
                wandb.log({
                    "train/train_loss": loss,
                    "train/epoch": epoch,
                })
                if step % save_freq == 0 or step == n_epochs*len(dataloader)-1:
                    w_logit_im = wandb.Image(logit_im, caption=f"Logits of Images - step {step}")
                    w_logit_dt = wandb.Image(logit_dt, caption=f"Logits of Data - step {step}")
                    wandb.log({
                        "train/logit_images": w_logit_im,
                        "train/logit_data": w_logit_dt,
                    })

                    torch.save({
                        "model": model,
                        "epoch": epoch,
                        "step": step,
                        "optimiser": optimiser,
                        "dataset": dataset,
                        "lr_scheduler": scheduler,
                        "losses": losses,
                    }, f"./exps/{wandb_run.name}/{wandb_run.name}-ckpt.pt")

                # Zero grad and back propagation
                optimiser.zero_grad()
                loss.backward()


                # Update gradients and scheduler
                optimiser.step()
                # scheduler.step()

                # Clip logit scaler
                with torch.no_grad():
                    model.logit_scale.clamp_(max=np.log(100))

                # Increment step Count
                step+= 1
    except KeyboardInterrupt:
        wandb.finish()
    wandb.finish()

    # Evaluate dataset
    N = len(dataset)
    for i in range(N):
        d_path = dataset.data_paths[i]
        d_name = d_path.split("/")[-1].split(".")[0] 
        img, data = dataset[i]
        
        image_features = model.encode_image(img.unsqueeze(0).to(next(iter(model.parameters())).device)).detach().cpu().numpy()
        data_features = model.encode_data(data.unsqueeze(0).to(next(iter(model.parameters())).device)).detach().cpu().numpy()

        np.save(f"./exps/{wandb_run.name}/embeddings/{d_name}-sclip-data-features.npy", data_features[0])
        np.save(f"./exps/{wandb_run.name}/embeddings/{d_name}-sclip-image-features.npy", image_features[0])

