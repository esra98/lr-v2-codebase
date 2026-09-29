#!/usr/bin/env python
"""Check that a machine is ready to train, before starting a long run.

Run it on the training server with the same options as the training command:

    python scripts/preflight.py -c configs/model/base.yaml model.modality=video

It checks the software, the GPU, the dataset, the tokenizer and the free disk
space, and then trains for a few steps on real data. Every check prints PASS
or FAIL, and the exit code is non-zero if any failed.
"""

import argparse
import os
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Batches differ in length, which fragments the GPU memory pool of PyTorch
# until allocations fail although enough memory is free. Expandable segments
# avoid that. Must be set before PyTorch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

FAILED = []


def report(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + str(detail) + ')' if detail != '' else ''}", flush=True)
    if not ok:
        FAILED.append(name)
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", action="append", default=[])
    parser.add_argument("--sample", type=int, default=200, help="Clips per split that are decoded")
    parser.add_argument("--steps", type=int, default=20, help="Training steps of the trial run")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    # ---- software ---------------------------------------------------------
    print(f"python {sys.version.split()[0]} on {sys.platform}")
    modules = {}
    for name in ("torch", "pytorch_lightning", "omegaconf", "sentencepiece", "av", "soundfile", "numpy"):
        try:
            modules[name] = __import__(name)
            print(f"      {name} {getattr(modules[name], '__version__', '?')}")
        except Exception as e:
            report(f"import {name}", False, f"{type(e).__name__}: {e}")
    if FAILED:
        sys.exit("\nInstall the missing packages: pip install -r requirements.txt")
    import torch

    from avsr.config import load_config, resolve_path
    from avsr.data.manifest import read_manifest
    from avsr.models.e2e import count_parameters
    from avsr.preprocess import video_io
    from avsr.text.tokenizer import UNK, build_tokenizer

    cfg = load_config(args.config, args.overrides)
    report("configuration is valid", True, f"{cfg.model.modality}, precision {cfg.train.precision}")

    # ---- GPU --------------------------------------------------------------
    cuda = torch.cuda.is_available()
    try:
        name = torch.cuda.get_device_name(0) if cuda else "training would run on the CPU"
        if cuda:
            torch.zeros(1, device="cuda")  # fails here if PyTorch and the driver do not match
        report("a CUDA GPU is available and usable", cuda, name)
    except Exception as e:
        cuda = False
        report("a CUDA GPU is available and usable", False, f"{type(e).__name__}: {str(e).splitlines()[0][:150]}")
        print(
            "      PyTorch cannot use the GPU. Its CUDA build ("
            + str(torch.version.cuda)
            + ") probably needs a newer NVIDIA driver.\n"
            "      Install a build that matches the driver shown by nvidia-smi, for example:\n"
            "      pip install torch --index-url https://download.pytorch.org/whl/cu126"
        )
        sys.exit("\n1 CHECK(S) FAILED; nothing else was checked.")
    if cuda:
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"      GPU memory {total:.1f} GB, {torch.cuda.device_count()} device(s); this code trains on one")
        if cfg.train.precision == "bf16-mixed":
            report("the GPU supports bfloat16", torch.cuda.is_bf16_supported())

    # ---- dataset ----------------------------------------------------------
    root, manifest_dir = resolve_path(cfg.data.root), resolve_path(cfg.data.manifests)
    report("dataset folder exists", os.path.isdir(root), root)
    manifests = {}
    for split in ("train", "val", "test"):
        path = os.path.join(manifest_dir, f"{split}.tsv")
        if report(f"{split} manifest exists", os.path.isfile(path), path):
            manifests[split] = read_manifest(path)
    if FAILED:
        sys.exit("\nThe dataset or its manifests are missing; nothing else can be checked.")
    for split, rows in manifests.items():
        missing = [r.utt_id for r in rows if not os.path.isfile(os.path.join(root, r.session, r.clip_id + ".mp4"))]
        hours = sum(r.num_frames for r in rows) / 25 / 3600
        report(
            f"every {split} clip is on disk",
            not missing,
            f"{len(rows)} clips, {hours:.1f} h" if not missing else f"{len(missing)} of {len(rows)} missing, e.g. {missing[0]}",
        )

    rng = random.Random(0)
    need_video = cfg.model.modality in ("video", "audiovisual")
    need_audio = cfg.model.modality in ("audio", "audiovisual")
    bad, sizes, checked = [], set(), 0
    start = time.time()
    for rows in manifests.values():
        for row in rng.sample(rows, min(args.sample, len(rows))):
            path = os.path.join(root, row.session, row.clip_id + ".mp4")
            try:
                if need_video:
                    video = video_io.read_video_gray(path)
                    sizes.add(video.shape[1:3])
                    if len(video) != row.num_frames:
                        bad.append(f"{row.utt_id}: {len(video)} frames, manifest says {row.num_frames}")
                if need_audio:
                    audio = video_io.read_audio(path)
                    if len(audio) < row.num_frames * video_io.SAMPLES_PER_FRAME - 1600:
                        bad.append(f"{row.utt_id}: audio shorter than the video")
                checked += 1
            except Exception as e:
                bad.append(f"{row.utt_id}: {type(e).__name__}: {e}")
    rate = checked / max(time.time() - start, 1e-6)
    report("sampled clips decode and match the manifest", not bad, f"{checked} clips, {rate:.0f} clips/s" if not bad else bad[0])
    # One loader process should manage well over 50 clips a second.
    report("clips decode fast enough", rate > 30, f"{rate:.0f} clips/s in one process")
    if need_video:
        crop = cfg.augment.video.crop_size
        fits = all(h >= crop and w >= crop for h, w in sizes)
        report("frames are large enough for the training crop", fits, f"stored {sorted(sizes)}, crop {crop or 'none'}")

    # ---- tokenizer --------------------------------------------------------
    try:
        tokenizer = build_tokenizer(cfg.tokenizer)
        report("tokenizer loads", True, f"{tokenizer.vocab_size} units")
        unk = tokenizer.token_list.index(UNK) if UNK in tokenizer.token_list else -1
        rows = rng.sample(manifests["train"], min(2000, len(manifests["train"])))
        with_unk = sum(unk in tokenizer.encode(r.text).tolist() for r in rows)
        report("training text is covered by the vocabulary", with_unk == 0, f"{with_unk} of {len(rows)} sentences contain unknown units")
        lossy = sum(tokenizer.decode(tokenizer.encode(r.text)) != r.text for r in rows)
        report("text survives encoding and decoding", lossy == 0, f"{lossy} of {len(rows)} changed")
    except Exception as e:
        report("tokenizer loads", False, f"{type(e).__name__}: {e}")
    if cfg.model.pretrained.path:
        path = resolve_path(cfg.model.pretrained.path)
        report("pretrained weights exist", os.path.isfile(path), path)
    if FAILED:
        sys.exit(f"\n{len(FAILED)} CHECK(S) FAILED; the trial run was not started.")

    # ---- trial run --------------------------------------------------------
    from omegaconf import OmegaConf
    from pytorch_lightning import Trainer, seed_everything

    from avsr.training.module import AVSRDataModule, AVSRModule

    trial = OmegaConf.merge(cfg, OmegaConf.from_dotlist(["data.limit_train=600", "data.limit_val=60"]))
    seed_everything(cfg.seed, workers=True)
    try:
        module = AVSRModule(trial)
        parameters = count_parameters(module.model)
        trainer = Trainer(
            accelerator="gpu" if cuda else "cpu",
            devices=1,
            precision=cfg.train.precision if cuda else "32-true",
            max_steps=args.steps,
            limit_val_batches=3,
            num_sanity_val_steps=2,
            gradient_clip_val=cfg.train.grad_clip,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            use_distributed_sampler=False,
            reload_dataloaders_every_n_epochs=1,
        )
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        trainer.fit(module, datamodule=AVSRDataModule(trial, module.tokenizer))
        seconds = time.time() - start
        loss = float(trainer.callback_metrics.get("train/loss", float("nan")))
        ok = trainer.global_step >= args.steps and loss == loss and module.oom_batches == 0
        detail = f"{trainer.global_step} steps in {seconds:.0f} s, loss {loss:.1f}, {module.oom_batches} batches out of memory"
        report("trial training run", ok, detail)
        if cuda:
            peak = torch.cuda.max_memory_allocated() / 2**30
            report(
                "the batch size fits the GPU with room to spare",
                peak < 0.9 * total,
                f"peak {peak:.1f} of {total:.1f} GB at data.max_frames_per_batch={cfg.data.max_frames_per_batch}",
            )
    except Exception as e:
        report("trial training run", False, f"{type(e).__name__}: {e}")
        parameters = 0

    # ---- disk -------------------------------------------------------------
    if parameters:
        copies = 2 if cfg.train.ema_decay > 0 else 1
        weights = parameters * 4 * copies / 2**30
        needed = weights * cfg.train.save_top_k + weights * 3  # kept epochs, and the full latest state
        target = resolve_path(cfg.exp_dir)
        os.makedirs(target, exist_ok=True)
        free = shutil.disk_usage(target).free / 2**30
        report("enough free disk for the checkpoints", free > 1.2 * needed, f"about {needed:.0f} GB needed, {free:.0f} GB free in {target}")

    print(f"\n{'READY TO TRAIN' if not FAILED else str(len(FAILED)) + ' CHECK(S) FAILED'}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
