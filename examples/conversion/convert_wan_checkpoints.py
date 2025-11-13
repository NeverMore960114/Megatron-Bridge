# from megatron.bridge.models.hf_pretrained.wan import PreTrainedWAN
# from megatron.bridge.models.wan.wan_bridge import WanBridge
# from megatron.bridge.training.model_load_save import save_megatron_model
# import os, random
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = str(29500 + random.randint(0, 1000))
# os.environ["RANK"] = "0"
# os.environ["WORLD_SIZE"] = "1"
# os.environ["LOCAL_RANK"] = "0"
# #
# hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
# # hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-14B-Diffusers")
# bridge = WanBridge()
# #
# provider = bridge.provider_bridge(hf)
# provider.perform_initialization = False
# megatron_models = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
# #
# bridge.load_weights_hf_to_megatron(hf, megatron_models)
# save_megatron_model(megatron_models, "/opt/megatron_checkpoint", hf_tokenizer_path=None)


# convert_wan_checkpoints.py

import os, random, multiprocessing as mp

def main():
    from megatron.bridge.models.hf_pretrained.wan import PreTrainedWAN
    from megatron.bridge.models.wan.wan_bridge import WanBridge
    from megatron.bridge.training.model_load_save import save_megatron_model

    # --- minimal torch.distributed single-rank env ---
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + random.randint(0, 1000)))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    # --- build & load ---
    hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    # hf = PreTrainedWAN("Wan-AI/Wan2.1-T2V-14B-Diffusers")

    bridge = WanBridge()
    provider = bridge.provider_bridge(hf)
    provider.perform_initialization = False

    # If you're on GPU but want CPU init to reduce peak mem:
    megatron_models = provider.provide_distributed_model(
        wrap_with_ddp=False, use_cpu_initialization=True
    )
    print(megatron_models[0])
    bridge.load_weights_hf_to_megatron(hf, megatron_models)


    # Save Megatron-format checkpoint (this triggers async writer internally)
    save_megatron_model(
        megatron_models,
        "/opt/megatron_checkpoint_WAN",
        hf_tokenizer_path=None
    )

if __name__ == "__main__":
    # On Linux, prefer 'fork' to avoid re-importing the module on spawn.
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        # already set (fine on re-entry or non-Linux)
        pass

    # If you’re on macOS/Windows and still want to be extra safe:
    # mp.freeze_support()

    main()

