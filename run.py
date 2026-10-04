import os
from IRdiffusers.pipeline_IRdiffusion3 import IRDiffusion3
import argparse
import pandas as pd

import torch

def get_args():
    parser = argparse.ArgumentParser(
        description="IR-Guided Diffusion")
    parser.add_argument("--model", type=str, default="SD3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataroot", type=str, default="dataroot")
    parser.add_argument("--dataset", type=str, default="OAO_attackbench")
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--target_hidden_layer", type=int, default=1)
    parser.add_argument("--where_to_intervene", type=str, default="embed", 
                        choices=["embed", "pooler", "embed_and_pooler"])
    parser.add_argument("--process", type=str, default="injection_0.2")

    args = parser.parse_args()
    return args


@torch.inference_mode()
def get_embeddings(prompt, hidden_target, apply_ln, args):
    clip_prompt_embed, clip_pooled_prompt_embed, clip_hidden_prompt_embed, clip_pooled_hidden_prompt_embed = pipe._get_clip_prompt_embeds(
        prompt=prompt,
        device=args.device,
        num_images_per_prompt=1,
        clip_skip=None,  
        clip_model_index=0,
        return_hidden_states=True, hidden_layer=hidden_target, apply_final_layernorm=apply_ln)
    clip_prompt_embed2, clip_pooled_prompt_embed2, clip_hidden_prompt_embed2, clip_pooled_hidden_prompt_embed2 = pipe._get_clip_prompt_embeds(
        prompt=prompt,
        device=args.device,
        num_images_per_prompt=1,
        clip_skip=None,
        clip_model_index=1,
        return_hidden_states=True, hidden_layer=hidden_target, apply_final_layernorm=apply_ln)

    final_clip_prompt_embed = torch.cat([clip_prompt_embed, clip_prompt_embed2], dim=-1)
    final_clip_hidden_embed = torch.cat([clip_hidden_prompt_embed, clip_hidden_prompt_embed2], dim=-1)

    t5_prompt_embed, t5_hidden_embed = pipe._get_t5_prompt_embeds(
        prompt=prompt,
        num_images_per_prompt=1,
        device=args.device,
        return_hidden_states=True, hidden_layer=hidden_target, apply_final_layernorm=apply_ln)

    final_clip_prompt_embed = torch.nn.functional.pad(
        final_clip_prompt_embed, (0, t5_prompt_embed.shape[-1] - final_clip_prompt_embed.shape[-1])
    )
    final_clip_hidden_embed = torch.nn.functional.pad(
        final_clip_hidden_embed, (0, t5_hidden_embed.shape[-1] - final_clip_hidden_embed.shape[-1])
    )


    prompt_embed = torch.cat([final_clip_prompt_embed, t5_prompt_embed], dim=-2)
    hidden_embed = torch.cat([final_clip_hidden_embed, t5_hidden_embed], dim=-2)

    pooled_prompt_embed = torch.cat([clip_pooled_prompt_embed, clip_pooled_prompt_embed2], dim=-1)
    pooled_hidden_embed = torch.cat([clip_pooled_hidden_prompt_embed, clip_pooled_hidden_prompt_embed2], dim=-1)

    lambd = float(args.process.split("_")[-1])

    delta = hidden_embed
    delta_pooled = pooled_hidden_embed

    return prompt_embed, pooled_prompt_embed, prompt_embed + lambd*delta, pooled_prompt_embed + lambd*delta_pooled

def main():
    df = pd.read_csv(f"{args.dataroot}/{args.dataset}.csv")
    all_text = list(df["caption"])

    hidden_target = args.target_hidden_layer

    negative_prompt_embed, negative_pooled_prompt_embed, \
    negative_hidden_embed, negative_pooled_hidden_embed = get_embeddings("", hidden_target, True, args)

    negative_embed_set = {"orig": negative_prompt_embed, "hidden": negative_hidden_embed}
    negative_pooled_set = {"orig": negative_pooled_prompt_embed, "hidden": negative_pooled_hidden_embed}

    save_path = f"save_path"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    for i, prompt in enumerate(all_text):
        if type(prompt) is not str: continue


        with torch.inference_mode():
            prompt_embed, pooled_prompt_embed, hidden_embed, pooled_hidden_embed = get_embeddings(prompt, hidden_target, True, args)
            prompt_embeds_set = {"orig": prompt_embed, "hidden": hidden_embed}
            pooled_embeds_set = {"orig": pooled_prompt_embed, "hidden": pooled_hidden_embed}

            gen = torch.Generator(device=args.device)
            gen.manual_seed(args.seed)

            outputs = pipe(prompt_embeds_set=prompt_embeds_set,
                pooled_embeds_set=pooled_embeds_set,
                negative_embeds_set=negative_embed_set,
                negative_pooled_set=negative_pooled_set, 
                num_images_per_prompt=args.n_images, 
                generator=gen, 
                where_to_intervene = args.where_to_intervene,  
                )
            batch_images = outputs.images
            # print(os.path.join(save_path, f"{i}.png"))

            batch_images[0].save(os.path.join(save_path, f"{i}.png"))


if __name__ == "__main__":
    args = get_args()
    gen = torch.Generator(device=args.device)
    gen.manual_seed(args.seed)

    pipe = IRDiffusion3.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    main()