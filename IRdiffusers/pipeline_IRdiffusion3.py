# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5TokenizerFast,
)


from diffusers.image_processor import VaeImageProcessor, PipelineImageInput
from diffusers.loaders import (
    FromSingleFileMixin,
    SD3LoraLoaderMixin, SD3IPAdapterMixin
)
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor


from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from IRdiffusers.transition_controller import TransitionController


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusion3Pipeline

        >>> pipe = StableDiffusion3Pipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("sd3.png")
        ```
"""


def compute_pred_x0(latents, model_output, step_index, scheduler):
    sigma = scheduler.sigmas[step_index]
    if torch.is_tensor(sigma):
        sigma = sigma.to(latents.device)
    else:
        sigma = torch.tensor(sigma, device=latents.device)
    
    while sigma.dim() < latents.dim():
        sigma = sigma.unsqueeze(-1)

    pred_original_sample = latents - (sigma * model_output)
    
    return pred_original_sample

# diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class IRDiffusion3(DiffusionPipeline, SD3LoraLoaderMixin, FromSingleFileMixin, SD3IPAdapterMixin):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        image_encoder (`SiglipVisionModel`, *optional*):
            Pre-trained Vision Model for IP Adapter.
        feature_extractor (`SiglipImageProcessor`, *optional*):
            Image processor for IP Adapter.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "negative_pooled_prompt_embeds"]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        image_encoder: SiglipVisionModel = None,
        feature_extractor: SiglipImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        self.patch_size = (
            self.transformer.config.patch_size if hasattr(self, "transformer") and self.transformer is not None else 2
        )

    def to_device_repeat_view(self, x, num_images_per_prompt, batch_size, device):
        # x: [batch, seq, dim]
        x = x.to(dtype=self.text_encoder.dtype, device=device)

        # [batch, seq, dim] -> [batch, num_images, seq, dim] -> [batch*num_images, seq, dim]
        x = x[:, None, :, :].expand(batch_size, num_images_per_prompt, x.shape[1], x.shape[2])
        x = x.reshape(batch_size * num_images_per_prompt, x.shape[2], x.shape[3])
        return x

    def pooled_to_device_repeat_view(self, x, num_images_per_prompt, batch_size, device):
        # x: [batch, dim]
        x = x.to(dtype=self.text_encoder.dtype, device=device)

        x = x[:, None, :].expand(batch_size, num_images_per_prompt, x.shape[1])
        x = x.reshape(batch_size * num_images_per_prompt, x.shape[2])
        return x



    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        return_hidden_states=False,
        hidden_layer=None,
        apply_final_layernorm = False, 

    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.transformer.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_3.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device), output_hidden_states=True)
        final_layernorm = self.text_encoder_3.encoder.final_layer_norm

        if return_hidden_states is True:
            if type(hidden_layer) is int: 
                hidden_prompt_embeds = prompt_embeds.hidden_states[hidden_layer]
                if apply_final_layernorm:
                    hidden_prompt_embeds = final_layernorm(hidden_prompt_embeds)
                hidden_prompt_embeds = self.to_device_repeat_view(hidden_prompt_embeds, num_images_per_prompt, batch_size, device)
            elif type(hidden_layer) is list:
                hidden_prompt_embeds = []
                for i in hidden_layer:
                    hidden_embed = prompt_embeds.hidden_states[i]
                    if apply_final_layernorm:
                        hidden_embed = final_layernorm(hidden_embed)
                    hidden_embed = self.to_device_repeat_view(hidden_embed, num_images_per_prompt, batch_size, device)
                    hidden_prompt_embeds.append(hidden_embed)

        prompt_embeds = prompt_embeds[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if return_hidden_states is True:
            return prompt_embeds, hidden_prompt_embeds
        else:
            return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
        return_hidden_states = False,
        hidden_layer = None, 
        apply_final_layernorm = False, 
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

        if return_hidden_states:
            ## manual projection ----------------------------------------------------
            eot_idx = text_input_ids.argmax(dim=-1)
            last_hidden = prompt_embeds['last_hidden_state']
            proj = text_encoder.text_projection

            last_hidden = last_hidden[torch.arange(last_hidden.size(0)), eot_idx]
            last_hidden_pooled = proj(last_hidden)
            assert torch.all(prompt_embeds[0] == last_hidden_pooled)
            ## manual projection ----------------------------------------------------

            final_layernorm = text_encoder.text_model.final_layer_norm


            if type(hidden_layer) is int: 
                hidden_prompt_embeds = prompt_embeds.hidden_states[hidden_layer]
                hidden_prompt_embeds_eot = hidden_prompt_embeds[torch.arange(last_hidden.size(0)), eot_idx]

                if apply_final_layernorm:
                    hidden_prompt_embeds = final_layernorm(hidden_prompt_embeds)
                    hidden_prompt_embeds_eot = final_layernorm(hidden_prompt_embeds_eot)
                pooled_hidden_prompt_embeds = proj(hidden_prompt_embeds_eot)

                hidden_prompt_embeds = self.to_device_repeat_view(hidden_prompt_embeds, num_images_per_prompt, batch_size, device)
                pooled_hidden_prompt_embeds = self.pooled_to_device_repeat_view(pooled_hidden_prompt_embeds, num_images_per_prompt, batch_size, device)

            elif type(hidden_layer) is list:
                pooled_hidden_prompt_embeds = []
                hidden_prompt_embeds = []
                for i in hidden_layer:
                    hidden_prompt_embed = prompt_embeds.hidden_states[i]
                    hidden_prompt_embed_eot = hidden_prompt_embed[torch.arange(last_hidden.size(0)), eot_idx]
                    
                    if apply_final_layernorm:
                        hidden_prompt_embed = final_layernorm(hidden_prompt_embed)
                        hidden_prompt_embed_eot = final_layernorm(hidden_prompt_embed_eot)
                    pooled_hidden_prompt_embed = proj(hidden_prompt_embed_eot)

                    hidden_prompt_embed = self.to_device_repeat_view(hidden_prompt_embed, num_images_per_prompt, batch_size, device)
                    hidden_prompt_embeds.append(hidden_prompt_embed)
                    
                    pooled_hidden_prompt_embed = self.pooled_to_device_repeat_view(pooled_hidden_prompt_embed, num_images_per_prompt, batch_size, device)
                    pooled_hidden_prompt_embeds.append(pooled_hidden_prompt_embed)

        pooled_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        
        if return_hidden_states:
            return prompt_embeds, pooled_prompt_embeds, hidden_prompt_embeds, pooled_hidden_prompt_embeds
        else:
            return prompt_embeds, pooled_prompt_embeds

    

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        prompt_embeds_set,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        max_sequence_length=None,
        pooled_embeds_set=None,
        negative_embeds_set=None,
        negative_pooled_set=None,
        num_images_per_prompt=1,
        guidance_scale=7.0,
        where_to_intervene=None,
    ):
        unsupported_inputs = {
            "prompt": prompt,
            "prompt_2": prompt_2,
            "prompt_3": prompt_3,
            "negative_prompt": negative_prompt,
            "negative_prompt_2": negative_prompt_2,
            "negative_prompt_3": negative_prompt_3,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
        provided_unsupported_inputs = [
            name for name, value in unsupported_inputs.items() if value is not None
        ]
        if provided_unsupported_inputs:
            raise ValueError(
                "`IRDiffusion3` only accepts the custom embedding-set API. "
                f"Unsupported inputs were provided: {provided_unsupported_inputs}."
            )

        embedding_sets = {
            "prompt_embeds_set": (prompt_embeds_set, 3),
            "pooled_embeds_set": (pooled_embeds_set, 2),
            "negative_embeds_set": (negative_embeds_set, 3),
            "negative_pooled_set": (negative_pooled_set, 2),
        }
        required_keys = {"orig", "hidden"}
        all_tensors = []

        for set_name, (tensor_set, expected_ndim) in embedding_sets.items():
            if not isinstance(tensor_set, dict):
                raise TypeError(f"`{set_name}` must be a dict with keys {sorted(required_keys)}.")

            if set(tensor_set) != required_keys:
                raise ValueError(
                    f"`{set_name}` must contain exactly the keys {sorted(required_keys)}, "
                    f"but got {sorted(tensor_set)}."
                )

            for key in ("orig", "hidden"):
                tensor = tensor_set[key]
                if not torch.is_tensor(tensor):
                    raise TypeError(f"`{set_name}['{key}']` must be a torch.Tensor.")
                if tensor.ndim != expected_ndim:
                    raise ValueError(
                        f"`{set_name}['{key}']` must be {expected_ndim}D, "
                        f"but got shape {tuple(tensor.shape)}."
                    )
                if any(size <= 0 for size in tensor.shape):
                    raise ValueError(f"`{set_name}['{key}']` cannot have an empty dimension.")
                if not tensor.is_floating_point():
                    raise TypeError(f"`{set_name}['{key}']` must use a floating-point dtype.")
                all_tensors.append((f"{set_name}['{key}']", tensor))

            if tensor_set["orig"].shape != tensor_set["hidden"].shape:
                raise ValueError(
                    f"`{set_name}['orig']` and `{set_name}['hidden']` must have the same shape, "
                    f"but got {tuple(tensor_set['orig'].shape)} and {tuple(tensor_set['hidden'].shape)}."
                )

        prompt_shape = prompt_embeds_set["orig"].shape
        pooled_shape = pooled_embeds_set["orig"].shape
        negative_shape = negative_embeds_set["orig"].shape
        negative_pooled_shape = negative_pooled_set["orig"].shape
        batch_size = prompt_shape[0]
        negative_batch_size = negative_shape[0]

        if pooled_shape[0] != batch_size:
            raise ValueError(
                "`pooled_embeds_set` and `prompt_embeds_set` must have the same batch size, "
                f"but got {pooled_shape[0]} and {batch_size}."
            )
        if negative_batch_size not in (1, batch_size):
            raise ValueError(
                "The negative embedding batch size must be 1 or match the positive batch size, "
                f"but got {negative_batch_size} and {batch_size}."
            )
        if negative_pooled_shape[0] != negative_batch_size:
            raise ValueError(
                "`negative_pooled_set` and `negative_embeds_set` must have the same batch size, "
                f"but got {negative_pooled_shape[0]} and {negative_batch_size}."
            )
        if negative_shape[1:] != prompt_shape[1:]:
            raise ValueError(
                "Positive and negative token embeddings must have the same sequence and feature dimensions, "
                f"but got {tuple(prompt_shape[1:])} and {tuple(negative_shape[1:])}."
            )
        if negative_pooled_shape[1:] != pooled_shape[1:]:
            raise ValueError(
                "Positive and negative pooled embeddings must have the same feature dimension, "
                f"but got {tuple(pooled_shape[1:])} and {tuple(negative_pooled_shape[1:])}."
            )

        dtypes = {tensor.dtype for _, tensor in all_tensors}
        if len(dtypes) != 1:
            dtype_details = ", ".join(f"{name}={tensor.dtype}" for name, tensor in all_tensors)
            raise ValueError(f"All custom embedding tensors must use the same dtype; got {dtype_details}.")

        if where_to_intervene not in {"embed", "pooler", "embed_and_pooler"}:
            raise ValueError(
                "`where_to_intervene` must be one of "
                "{'embed', 'pooler', 'embed_and_pooler'}."
            )

        if (
            not isinstance(num_images_per_prompt, int)
            or isinstance(num_images_per_prompt, bool)
            or num_images_per_prompt <= 0
        ):
            raise ValueError("`num_images_per_prompt` must be a positive integer.")

        if guidance_scale <= 1:
            raise ValueError(
                "The current custom embedding-set path requires `guidance_scale > 1` because it uses "
                "classifier-free guidance with positive and negative embedding sets."
            )

        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or isinstance(width, bool)
            or height <= 0
            or width <= 0
        ):
            raise ValueError("`height` and `width` must be positive integers.")

        required_multiple = self.vae_scale_factor * self.patch_size
        if height % required_multiple != 0 or width % required_multiple != 0:
            raise ValueError(
                f"`height` and `width` must be divisible by {required_multiple}, but got "
                f"{height} and {width}. You can use {height - height % required_multiple} and "
                f"{width - width % required_multiple}."
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def skip_guidance_layers(self):
        return self._skip_guidance_layers

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_image
    def encode_image(self, image: PipelineImageInput, device: torch.device) -> torch.Tensor:
        """Encodes the given image into a feature representation using a pre-trained image encoder.

        Args:
            image (`PipelineImageInput`):
                Input image to be encoded.
            device: (`torch.device`):
                Torch device.

        Returns:
            `torch.Tensor`: The encoded image feature representation.
        """
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=self.dtype)

        return self.image_encoder(image, output_hidden_states=True).hidden_states[-2]

    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
    ) -> torch.Tensor:
        """Prepares image embeddings for use in the IP-Adapter.

        Either `ip_adapter_image` or `ip_adapter_image_embeds` must be passed.

        Args:
            ip_adapter_image (`PipelineImageInput`, *optional*):
                The input image to extract features from for IP-Adapter.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Precomputed image embeddings.
            device: (`torch.device`, *optional*):
                Torch device.
            num_images_per_prompt (`int`, defaults to 1):
                Number of images that should be generated per prompt.
            do_classifier_free_guidance (`bool`, defaults to True):
                Whether to use classifier free guidance or not.
        """
        device = device or self._execution_device

        if ip_adapter_image_embeds is not None:
            if do_classifier_free_guidance:
                single_negative_image_embeds, single_image_embeds = ip_adapter_image_embeds.chunk(2)
            else:
                single_image_embeds = ip_adapter_image_embeds
        elif ip_adapter_image is not None:
            single_image_embeds = self.encode_image(ip_adapter_image, device)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.zeros_like(single_image_embeds)
        else:
            raise ValueError("Neither `ip_adapter_image_embeds` or `ip_adapter_image_embeds` were provided.")

        image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            negative_image_embeds = torch.cat([single_negative_image_embeds] * num_images_per_prompt, dim=0)
            image_embeds = torch.cat([negative_image_embeds, image_embeds], dim=0)

        return image_embeds.to(device=device)

    def enable_sequential_cpu_offload(self, *args, **kwargs):
        if self.image_encoder is not None and "image_encoder" not in self._exclude_from_cpu_offload:
            logger.warning(
                "`pipe.enable_sequential_cpu_offload()` might fail for `image_encoder` if it uses "
                "`torch.nn.MultiheadAttention`. You can exclude `image_encoder` from CPU offloading by calling "
                "`pipe._exclude_from_cpu_offload.append('image_encoder')` before `pipe.enable_sequential_cpu_offload()`."
            )

        super().enable_sequential_cpu_offload(*args, **kwargs)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds_set: dict[str, torch.Tensor] = None,
        pooled_embeds_set: dict[str, torch.Tensor] = None,
        negative_embeds_set: dict[str, torch.Tensor] = None,
        negative_pooled_set: dict[str, torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
        return_hidden_states: Optional[bool] = False,
        hidden_layer: Optional[Union[int, List[int]]] = None, 
        where_to_intervene: str = None
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image (`PipelineImageInput`, *optional*):
                Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`torch.Tensor`, *optional*):
                Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
                emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
                `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
            skip_guidance_layers (`List[int]`, *optional*):
                A list of integers that specify layers to skip during guidance. If not provided, all layers will be
                used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
                Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
            skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
                `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
                with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
                with a scale of `1`.
            skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
                `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
            skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
                `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
                `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
                StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            prompt_embeds_set, 
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            max_sequence_length=max_sequence_length,
            pooled_embeds_set=pooled_embeds_set,
            negative_embeds_set=negative_embeds_set,
            negative_pooled_set=negative_pooled_set,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
            where_to_intervene=where_to_intervene,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt_embeds_set is not None:
            batch_size = len(next(iter(prompt_embeds_set.values())))
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        dtype = list(prompt_embeds_set.values())[0].dtype

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        latents = self.switch_on_satupoint_loop(
                    latents, 
                    num_inference_steps, 
                    timesteps, 
                    prompt_embeds_set, 
                    pooled_embeds_set,
                    negative_embeds_set,
                    negative_pooled_set,
                    num_images_per_prompt, 
                    batch_size, 
                    device,
                    skip_layer_guidance_start,
                    skip_layer_guidance_stop,
                    skip_guidance_layers,
                    num_warmup_steps,
                    where_to_intervene,
                    )

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


    def switch_on_satupoint_loop(
                self, 
                latents, 
                num_inference_steps, 
                timesteps, 
                prompt_embeds_set, 
                pooled_embeds_set,
                negative_embeds_set,
                negative_pooled_set,
                num_images_per_prompt, 
                batch_size, 
                device,
                skip_layer_guidance_start,
                skip_layer_guidance_stop,
                skip_guidance_layers,
                num_warmup_steps,
                where_to_intervene,
                ):

        img_batch_size = latents.shape[0]
        detector = TransitionController(img_batch_size, device=latents.device, total_steps=num_inference_steps, warmup=0)

        prev_x0 = None

        prompt_embeds_orig = self.to_device_repeat_view(prompt_embeds_set['orig'], num_images_per_prompt, batch_size, device) 
        prompt_embeds_hidden = self.to_device_repeat_view(prompt_embeds_set['hidden'], num_images_per_prompt, batch_size, device) 
        pooled_embeds_orig = self.pooled_to_device_repeat_view(pooled_embeds_set['orig'], num_images_per_prompt, batch_size, device) 
        pooled_embeds_hidden = self.pooled_to_device_repeat_view(pooled_embeds_set['hidden'], num_images_per_prompt, batch_size, device) 

        negative_embeds_orig = self.to_device_repeat_view(negative_embeds_set['orig'], num_images_per_prompt, batch_size, device)  
        negative_embeds_hidden = self.to_device_repeat_view(negative_embeds_set['hidden'], num_images_per_prompt, batch_size, device) 
        negative_pooled_orig = self.pooled_to_device_repeat_view(negative_pooled_set['orig'], num_images_per_prompt, batch_size, device) 
        negative_pooled_hidden = self.pooled_to_device_repeat_view(negative_pooled_set['hidden'], num_images_per_prompt, batch_size, device) 

        final_embeds_orig = torch.cat([negative_embeds_orig, prompt_embeds_orig], dim=0) 
        final_pooled_orig = torch.cat([negative_pooled_orig, pooled_embeds_orig], dim=0) 
        final_embeds_hidden = torch.cat([negative_embeds_hidden, prompt_embeds_hidden], dim=0) 
        final_pooled_hidden = torch.cat([negative_pooled_hidden, pooled_embeds_hidden], dim=0) 

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                mask_stopped = detector.is_stopped.view(img_batch_size, 1, 1) 
                full_mask = torch.cat([mask_stopped, mask_stopped], dim=0) 

                mask_stopped_pooled = detector.is_stopped.view(img_batch_size, 1)  
                full_mask_pooled = torch.cat([mask_stopped_pooled, mask_stopped_pooled], dim=0)  

                if where_to_intervene == "embed_and_pooler":
                    prompt_embeds = torch.where(full_mask, final_embeds_orig, final_embeds_hidden)
                    pooled_prompt_embeds = torch.where(full_mask_pooled, final_pooled_orig, final_pooled_hidden) 
                elif where_to_intervene == "pooler":
                    prompt_embeds = final_embeds_orig
                    pooled_prompt_embeds = torch.where(full_mask_pooled, final_pooled_orig, final_pooled_hidden) 
                elif where_to_intervene == "embed":
                    prompt_embeds = torch.where(full_mask, final_embeds_orig, final_embeds_hidden) 
                    pooled_prompt_embeds = final_pooled_orig

                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])


                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance: 
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep = t.expand(latents.shape[0])
                        latent_model_input = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=original_prompt_embeds,
                            pooled_projections=original_pooled_prompt_embeds,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                current_x0 = compute_pred_x0(latents, noise_pred, i, self.scheduler)                
                if prev_x0 is None:
                    diff_vals = 0
                else:
                    diff_vals = (current_x0 - prev_x0).pow(2).mean(dim=[1, 2, 3])
                    is_stopped_flags = detector.update(diff_vals, i)
                prev_x0 = current_x0

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()


        return latents
