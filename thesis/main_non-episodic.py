"""Non-episodic main script, which does not make use of the learn2learn lib."""

import argparse
from collections import defaultdict
from functools import partial
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from thesis.lit_models import LitBaseNonEpisodic, LitAdaptiveBatchnormModel
from thesis.writer_code.lit_models import (
    LitWriterCodeAdaptiveModelNonEpisodic,
)
from thesis.util import (
    get_label_encoder,
    save_label_encoder,
    get_pl_tb_logger,
    copy_hyperparameters_to_logging_dir,
    prepare_iam_splits,
    get_parameter_names,
    EOS_TOKEN,
    SOS_TOKEN,
    PAD_TOKEN,
    load_best_pl_checkpoint,
    get_bn_statistics,
    collect_bn_layers,
)

from htr.data import IAMDataset
from htr.util import LitProgressBar

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.plugins import DDPPlugin

from thesis.writer_code.util import load_hinge_codes


def main(args):

    print(f"Main model used: {str(args.main_model_arch)}")
    print(f"Base model used: {str(args.base_model_arch).upper()}")
    print(f"Percentage of old statistics used: {args.old_stats_prcnt * 100}%")

    seed_everything(args.seed)

    # Initalize logging/cache directories.
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = Path(__file__).parent.resolve()
    tb_logger = get_pl_tb_logger(log_dir, args.experiment_name)
    log_dir = tb_logger.log_dir

    label_enc = get_label_encoder(args.trained_model_path)
    save_label_encoder(label_enc, log_dir)
    model_hparams_file, hparams = copy_hyperparameters_to_logging_dir(
        args.trained_model_path, log_dir
    )
    base_model_params = get_parameter_names(args.trained_model_path)

    only_lowercase = hparams["only_lowercase"]
    augmentations = "train" if args.use_image_augmentations else "val"

    ds = IAMDataset(
        args.data_dir,
        "word",
        "train",
        label_enc=label_enc,
        return_writer_id=True,
        return_writer_id_as_idx=True,
        only_lowercase=only_lowercase,
    )

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = label_enc.transform(
        [EOS_TOKEN, SOS_TOKEN, PAD_TOKEN]
    )
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )

    ds_train, ds_val, ds_test = prepare_iam_splits(
        ds, Path(__file__).resolve().parent.parent / "aachen_splits"
    )

    # Set image transforms.
    ds.set_transforms_for_split("test")
    ds_train.set_transforms_for_split(augmentations)
    ds_val.set_transforms_for_split("val")
    ds_test.set_transforms_for_split("test")

    # Initialize dataloaders.
    dl = DataLoader(
        ds,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=2 * args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Dataset split sizes:")
    print(f"train:\t{len(ds_train)}")
    print(f"val:\t{len(ds_val)}")
    print(f"test:\t{len(ds_test)}")

    writer_codes, code_size = load_hinge_codes(
        Path(__file__).resolve().parent.parent, code_name=args.code_name
    )
    writer_codes = {
        ds.writer_id_to_idx[wid]: code
        for wid, code in writer_codes.items()
        # ds.writer_id_to_idx[wid]: np.zeros_like(code) for wid, code in writer_codes.items()  # fill with zero code
        # ds.writer_id_to_idx[wid]: np.random.normal(0, 1, code.size).astype(code.dtype) for wid, code in writer_codes.items()  # fill with random numbers
    }

    # Define model arguments.
    args_ = dict(
        checkpoint_path=args.trained_model_path,
        model_hparams_file=model_hparams_file,
        label_encoder=ds_train.label_enc,
        load_meta_weights=True,
        model_params_to_log={"only_lowercase": only_lowercase},
        writer_codes=writer_codes,
        code_size=code_size,
        prms_to_log={
            "main_model_arch": args.main_model_arch,
            "base_model_arch": args.base_model_arch,
            "seed": args.seed,
            "model_path": args.trained_model_path,
            "early_stopping_patience": args.early_stopping_patience,
            "use_image_augmentations": args.use_image_augmentations,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
        },
        **vars(args),
    )
    # Initialize with a trained base model.
    learner = (
        LitWriterCodeAdaptiveModelNonEpisodic.init_with_base_model_from_checkpoint(
            **args_
        )
    )

    num_bn_layers = 19
    bn_stats_dir = Path("../bn_stats")
    # if bn_stats_dir.is_dir():
    #     print("Loading batchnorm statistics from disk.")
    #     layer_stats_per_writer = dict()
    #     for layer_id in range(num_bn_layers + 1):
    #         stats = np.load(
    #             str(bn_stats_dir / f"layer_{layer_id}_stats_per_writer.npy")
    #         )  # (nwriters, C, 2)
    #         layer_stats_per_writer[layer_id] = stats
    # else:
    device = "cpu" if args.use_cpu else "cuda:0"
    dataset = ds_test if args.test else ds_val

    base_model = learner.model.model
    writer_stats_pth = Path(f"writer_stats_{'test' if args.test else 'val'}.pkl")
    if writer_stats_pth.is_file():
        print("Loading stats from disk.")
        with open(writer_stats_pth, "rb") as f:
            writer_stats = pickle.load(f)
        bn_layers = collect_bn_layers(base_model)
        for i, m in enumerate(bn_layers):
            # Do not use a exponentially moving average but a cumulative
            # average.
            m.momentum = None
            # Save old statistics.
            m.running_mean_old = m.running_mean.clone()
            m.running_var_old = m.running_var.clone()
            # Set a global identifier for each layer.
            m.layer_idx = i
    else:
        print(
            f"Calculating writer-specific activation statistics for {len(dataset.writer_ids)} writers."
        )
        writer_stats = get_bn_statistics(
            base_model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        print("Done.")
        with open(writer_stats_pth, "wb") as f:
            pickle.dump(writer_stats, f)
        print("Stats saved.")

    # Make the primary key for the statistics the layer index.
    layer_stats_per_writer = dict()
    for i in range(num_bn_layers + 1):
        layer_stats_per_writer[i] = torch.stack(
            [wstats[i] for wstats in writer_stats], 0
        )

    base_model = learner.model.model
    d_model = (
        base_model.lstm_encoder.rnn_encoder.input_size
        if args.base_model_arch == "sar"
        else base_model.encoder.resnet_out_features
    )
    learner = LitAdaptiveBatchnormModel(
        base_model=base_model,
        layer_stats_per_writer=layer_stats_per_writer,
        d_model=d_model,
        cer_metric=base_model.cer_metric,
        wer_metric=base_model.wer_metric,
        **args_,
    )

    # Save batchnorm statistics.
    # bn_stats_dir.mkdir(exist_ok=True)
    # for layer_id in layer_stats_per_writer.keys():
    #     wrtr_stats = []
    #     for wid in layer_stats_per_writer[layer_id].keys():
    #         channel_stats = []
    #         for chan, stats in layer_stats_per_writer[layer_id][wid].items():
    #             channel_stats.append([stats["mean"], stats["var"]])
    #         wrtr_stats.append(channel_stats)
    #     wrtr_stats_np = np.array(wrtr_stats)
    #     with open(bn_stats_dir / f"layer_{layer_id}_stats_per_writer.npy", "wb") as f:
    #         np.save(f, wrtr_stats_np)
    #
    callbacks = [
        ModelSummary(max_depth=3),
        LitProgressBar(),
        ModelCheckpoint(
            save_top_k=(-1 if args.save_all_checkpoints else 3),
            mode="min",
            monitor="word_error_rate",
            filename=args.main_model_arch
            + "-{epoch}-{char_error_rate:.4f}-{word_error_rate:.4f}",
            save_weights_only=True,
        ),
    ]
    callbacks = learner.add_model_specific_callbacks(
        callbacks,
        label_encoder=ds_train.label_enc,
        is_train=not (args.validate or args.test),
    )
    if args.early_stopping_patience != -1:
        callbacks.append(
            EarlyStopping(
                monitor="word_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
            )
        )

    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        gpus=(0 if args.use_cpu else 1),
        log_every_n_steps=10,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    if args.test:
        trainer.test(learner, dl_test)
    else:
        trainer.validate(learner, dl_val)

    if args.test_on_fit_end:
        trainer.test(learner, dl_test)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--main_model_arch", type=str, required=True,
                        choices=["WriterCodeAdaptiveModelNonEpisodic"])
    parser.add_argument("--base_model_arch", type=str, required=True,
                        choices=["fphtr", "sar"], default="fphtr")
    parser.add_argument("--trained_model_path", type=str, required=True,
                        help=("Path to a base model checkpoint"))
    parser.add_argument("--data_dir", type=str, required=True, help="IAM dataset root folder.")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory where the lighning logs will be stored.")
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of checks with no improvement after which "
                             "training will be stopped. Setting this to -1 will disable "
                             "early stopping.")
    parser.add_argument("--use_image_augmentations", action="store_true", default=False,
                        help="Whether to use image augmentations during training. For "
                             "MAML this does not seem to be too beneficial so far.")
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--test_on_fit_end", action="store_true", default=False,
                        help="Run test on the best model checkpoint after training.")
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name, used as the name of the folder in "
                             "which logs are stored.")
    # fmt: on

    parser = LitBaseNonEpisodic.add_model_specific_args(parser)
    parser = LitWriterCodeAdaptiveModelNonEpisodic.add_model_specific_args(parser)
    parser = LitAdaptiveBatchnormModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)  # adds Pytorch Lightning arguments

    args = parser.parse_args()
    main(args)
