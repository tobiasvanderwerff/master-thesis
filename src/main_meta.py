import argparse
import shutil
from copy import copy
from pathlib import Path
from functools import partial

from models import *
from lit_models import MetaHTR
from data import IAMDataset
from lit_util import MAMLCheckpointIO, LitProgressBar, PtTaskDataset
from lit_callbacks import LogModelPredictionsMAML
from util import filter_df_by_freq, pickle_load, pickle_save

import pytorch_lightning as pl
import pandas as pd
import learn2learn as l2l
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


LOGGING_DIR = "lightning_logs/"
NUM_QUERY_PREDICTIONS_TO_LOG = 10


def main(args):

    seed_everything(args.seed)

    log_dir_root = Path(__file__).parent.parent.resolve()
    tb_logger = pl_loggers.TensorBoardLogger(log_dir_root / LOGGING_DIR, name="")

    assert Path(
        args.trained_model_path
    ).is_file(), f"{args.trained_model_path} does not point to a file."

    # Load the label encoder for the trained model.
    le_path = Path(args.trained_model_path).parent.parent / "label_encoder.pkl"
    assert le_path.is_file(), (
        f"Label encoder file not found at {le_path}. "
        f"Make sure 'label_encoder.pkl' exists in the lightning_logs directory."
    )
    label_enc = pd.read_pickle(le_path)

    # Save the label encoder in the logging directory.
    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    pickle_save(label_enc, log_dir / "label_encoder.pkl")

    ds = IAMDataset(
        args.data_dir,
        "word",
        "train",
        use_cache=False,
        label_enc=label_enc,
        skip_bad_segmentation=False,
        return_writer_id=True,
    )

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = ds.label_enc.transform(
        [ds._eos_token, ds._sos_token, ds._pad_token]
    ).tolist()
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )

    # Split the dataset into train/val/(test).
    if args.use_aachen_splits:
        # Use the Aachen splits for the IAM dataset. It should be noted that these
        # splits do not encompass the complete IAM dataset.
        aachen_path = Path(__file__).parent.parent / "aachen_splits"
        train_splits = (aachen_path / "train.uttlist").read_text().splitlines()
        validation_splits = (
            (aachen_path / "validation.uttlist").read_text().splitlines()
        )
        # test_splits = (aachen_path / "test.uttlist").read_text().splitlines()

        data_train = ds.data[ds.data["img_id"].isin(train_splits)]
        data_val = ds.data[ds.data["img_id"].isin(validation_splits)]
        # data_test = ds.data[ds.data["img_id"].isin(test_splits)]

        ds_train = copy(ds)
        ds_train.data = data_train

        ds_val = copy(ds)
        ds_val.data = data_val

        # ds_test = copy(ds)
        # ds_test.data = data_test
    else:
        ds_train, ds_val = torch.utils.data.random_split(
            ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))]
        )
        ds_val.dataset = copy(ds)

    # Exclude writers from the dataset that do not have sufficiently many samples.
    ds_train.data = filter_df_by_freq(ds_train.data, "writer_id", args.shots * 2)
    ds_val.data = filter_df_by_freq(ds_val.data, "writer_id", args.shots * 2)
    # ds_test.data = filter_df_by_freq(ds_test.data, "writer_id", args.shots * 2)

    # Set image transforms.
    if args.use_aachen_splits:
        ds_train.set_transforms_for_split("train")
        ds_val.set_transforms_for_split("val")
        # ds_test.set_transforms_for_split("test")
    else:
        ds_train.dataset.set_transforms_for_split("train")
        ds_val.dataset.set_transforms_for_split("val")

    assert (ds_train.data["writer_id"].value_counts() >= args.shots * 2).all()
    assert (ds_val.data["writer_id"].value_counts() >= args.shots * 2).all()

    # Initalize meta dataset and cache the result.
    # TODO: skip previous data initializaiton steps when cache is present.
    cache_dir = Path(args.cache_dir) if args.cache_dir else log_dir / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)
    ds_meta_train_path = cache_dir / "ds_meta_train.pkl"
    ds_meta_val_path = cache_dir / "ds_meta_val.pkl"
    if ds_meta_train_path.is_file():
        print("Loading cached train meta dataset.")
        ds_meta_train = pickle_load(ds_meta_train_path)
    else:
        print("Creating train meta dataset...")
        ds_meta_train = l2l.data.MetaDataset(ds_train)
        pickle_save(ds_meta_train, ds_meta_train_path)
    if ds_meta_val_path.is_file():
        print("Loading cached val meta dataset.")
        ds_meta_val = pickle_load(ds_meta_val_path)
    else:
        print("Creating val meta dataset...")
        ds_meta_val = l2l.data.MetaDataset(ds_val)
        pickle_save(ds_meta_val, ds_meta_val_path)

    # Define learn2learn task transforms.
    train_tsk_trnsf = [
        # Nways picks N random labels (writers in this case)
        l2l.data.transforms.NWays(ds_meta_train, n=args.ways),
        # Keeps K samples for each present writer.
        l2l.data.transforms.KShots(ds_meta_train, k=args.shots * 2),
        # Load the data.
        l2l.data.transforms.LoadData(ds_meta_train),
    ]
    val_tsk_trnsf = [
        l2l.data.transforms.NWays(ds_meta_val, n=args.ways),
        l2l.data.transforms.LoadData(ds_meta_val),
    ]
    taskset_train = l2l.data.TaskDataset(
        ds_meta_train,
        train_tsk_trnsf,
        num_tasks=-1,
        task_collate=collate_fn,
    )
    taskset_val = l2l.data.TaskDataset(
        ds_meta_val,
        val_tsk_trnsf,
        num_tasks=-1,
        task_collate=collate_fn,
    )

    # Wrap the task datasets into a simple class that sets a length for the dataset (
    # other than 1, which is the default if setting num_tasks=-1).
    # This is necessary because the dataset length is used by Pytorch dataloaders to
    # determine how many batches are in the dataset per epoch.
    taskset_train = PtTaskDataset(
        taskset_train, epoch_length=int(len(ds_train.writer_ids) / args.ways)
    )
    taskset_val = PtTaskDataset(
        taskset_val, epoch_length=int(len(ds_val.writer_ids) / args.ways)
    )

    # Copy hyper parameters. The loaded model has an associated `hparams.yaml` file,
    # which we copy to the current logging directory so that we can load the model
    # later using the saved hyper parameters.
    model_hparams_file = Path(args.trained_model_path).parent.parent / "hparams.yaml"
    shutil.copy(model_hparams_file, log_dir / "model_hparams.yaml")

    # Initialize MAML with a trained FPHTR model.
    hparams_file = (
        str(model_hparams_file)
        if model_hparams_file.is_file()
        else model_hparams_file.parent / "model_hparams.yaml"
    )
    learner = MetaHTR.init_with_fphtr_from_checkpoint(
        args.trained_model_path,
        hparams_file,
        ds.label_enc,
        taskset_train,
        taskset_val=taskset_val,
        ways=args.ways,
        shots=args.shots,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        num_workers=args.num_workers,
        params_to_log={
            "seed": args.seed,
            "splits": ("Aachen" if args.use_aachen_splits else "random"),
            "max_epochs": args.max_epochs,
            "model_path": args.trained_model_path,
        },
    )

    # This checkpoint plugin is necessary to save the weights obtained using MAML in
    # the proper way. The weights should be stored in the same format as they would
    # be saved without using MAML, to make it straightforward to load the model
    # weights later on.
    checkpoint_io = MAMLCheckpointIO()

    im, t, _ = next(iter(learner.val_dataloader()))
    val_batch = (
        im[: args.shots],
        t[: args.shots],
        im[args.shots : args.shots + NUM_QUERY_PREDICTIONS_TO_LOG],
        t[args.shots : args.shots + NUM_QUERY_PREDICTIONS_TO_LOG],
    )
    im, t, _ = next(iter(learner.train_dataloader()))
    train_batch = (
        im[: args.shots],
        t[: args.shots],
        im[args.shots : args.shots + NUM_QUERY_PREDICTIONS_TO_LOG],
        t[args.shots : args.shots + NUM_QUERY_PREDICTIONS_TO_LOG],
    )

    callbacks = [
        LitProgressBar(),
        ModelCheckpoint(
            save_top_k=(-1 if args.save_all_checkpoints else 3),
            mode="min",
            monitor="word_error_rate",
            filename="MAML-{step}-{char_error_rate:.4f}-{word_error_rate:.4f}",
            save_weights_only=True,
        ),
        # EarlyStopping(
        #     monitor="word_error_rate",
        #     patience=args.early_stopping_patience,
        #     verbose=True,
        #     mode="min",
        #     # check_on_train_epoch_end=False,  # check at the end of validation
        # ),
        LogModelPredictionsMAML(
            ds.label_enc,
            val_batch=val_batch,
            train_batch=train_batch,
            use_gpu=(False if args.use_cpu else True),
            enable_grad=True,
            predict_on_train_start=True,
        ),
    ]

    trainer = pl.Trainer(
        logger=tb_logger,
        plugins=[checkpoint_io],
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        precision=args.precision,  # default is 32 bit
        num_nodes=args.num_nodes,
        gpus=(0 if args.use_cpu else 1),
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=10,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    if args.validate:  # validate a trained model
        trainer.validate(learner)  # TODO: check if this works properly
    else:  # train a model
        trainer.fit(learner)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    # Trainer arguments.
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to train on.")
    parser.add_argument("--precision", type=int, default=32, help="How many bits of floating point precision to use.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.0 indicates no smoothing)")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)

    # Program arguments.
    parser.add_argument("--trained_model_path", type=str,
                        help=("Path to a model checkpoint, which will be used as a "
                              "starting point for MAML/MetaHTR."))
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--use_aachen_splits", action="store_true", default=False)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)

    parser = MetaHTR.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
    # fmt: on
