import argparse
import shutil
from copy import copy
from pathlib import Path
from functools import partial

from models import *
from lit_models import LitFullPageHTREncoderDecoder, MetaHTR
from data import IAMDataset
from lit_util import MAMLCheckpointIO, LitProgressBar

import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.plugins import DDPPlugin

import learn2learn as l2l

from util import filter_df_by_freq

LOGGING_DIR = "lightning_logs/"
LOGMODELPREDICTIONS_TO_SAMPLE = 8


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

    # Sanity check. TODO: remove this later.
    assert (ds_train.data["writer_id"].value_counts() >= args.shots * 2).all()
    assert (ds_val.data["writer_id"].value_counts() >= args.shots * 2).all()

    ds_meta_train = l2l.data.MetaDataset(ds_train)
    ds_meta_val = l2l.data.MetaDataset(ds_val)

    # Copy hyper parameters. The loaded model has an associated `hparams.yaml` file,
    # which we copy to the current logging directory so that we can load the model
    # later using the saved hyper parameters.
    model_hparams_file = Path(args.trained_model_path).parent.parent / "hparams.yaml"
    save_dir = Path(tb_logger.log_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    shutil.copy(model_hparams_file, save_dir / "hparams.yaml")

    # Load the model. Note that the vocab length and special tokens given below
    # are derived from the saved label encoder associated with the checkpoint.
    model = LitFullPageHTREncoderDecoder.load_from_checkpoint(
        args.trained_model_path,
        hparams_file=str(model_hparams_file),
        label_encoder=ds.label_enc,
    )
    # The `LitFullPageHTREncoderDecoder` class is a simple wrapper around a Pytorch
    # module, which we pass to the MAML learner.
    model = model.model

    # Define learn2learn task transforms.
    train_tsk_trnsf = [
        # Nways picks N random labels (writers in this case)
        l2l.data.transforms.NWays(ds_meta_train, n=args.ways),
        # Keeps K samples for each present label.
        l2l.data.transforms.KShots(ds_meta_train, k=args.shots * 2),
        # Load the data.
        l2l.data.transforms.LoadData(ds_meta_train),
        # Given samples from K classes, maps the labels to 0, ..., K.
        # l2l.data.transforms.RemapLabels(ds_meta_train),
        # Re-orders the samples in the task description such that they are sorted in
        # consecutive order.
        # l2l.data.transforms.ConsecutiveLabels(ds_meta_train),
    ]
    val_tsk_trnsf = [  # learn2learn transforms
        l2l.data.transforms.NWays(ds_meta_val, n=args.ways),
        l2l.data.transforms.KShots(ds_meta_val, k=args.shots * 2),
        l2l.data.transforms.LoadData(ds_meta_val),
    ]
    taskset_train = l2l.data.TaskDataset(
        ds_meta_train,
        train_tsk_trnsf,
        num_tasks=args.max_iterations,
        task_collate=collate_fn,
    )
    taskset_val = l2l.data.TaskDataset(
        ds_meta_val,
        val_tsk_trnsf,
        num_tasks=len(ds_val.writer_ids) / args.ways,
        task_collate=collate_fn,
    )

    learner = MetaHTR(
        model,
        taskset_train,
        taskset_val=taskset_val,
        ways=args.ways,
        shots=args.shots,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        num_workers=args.num_workers,
    )

    # This checkpoint plugin is necessary to save the weights obtained using MAML in
    # the proper way. The weights should be stored in the same format as they would
    # be saved without using MAML, to make it straightforward to load the model
    # weights later on.
    checkpoint_io = MAMLCheckpointIO()

    callbacks = (
        [
            LitProgressBar(),
            ModelCheckpoint(
                save_top_k=(-1 if args.save_all_checkpoints else 3),
                mode="min",
                monitor="char_error_rate",
                filename="MAML-{step}-{char_error_rate:.4f}-{word_error_rate:.4f}",
                every_n_train_steps=args.val_check_interval + 1,
                save_weights_only=True,
            ),
            EarlyStopping(
                monitor="char_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
                check_on_train_epoch_end=False,  # check at the end of validation
            ),
            # LogModelPredictions(
            #     ds.label_enc,
            #     val_batch=next(iter(learner.val_dataloader())),
            #     use_gpu=(False if args.use_cpu else True),
            # )
        ],
    )

    trainer = pl.Trainer(
        logger=tb_logger,
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        precision=args.precision,  # default is 32 bit
        num_nodes=args.num_nodes,
        gpus=(1 if args.use_gpu else 0),
        max_epochs=1,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        plugins=[checkpoint_io],
        enable_model_summary=False,
        val_check_interval=args.val_check_interval,
        callbacks=callbacks,
    )
    trainer.logger._default_hp_metric = None

    if args.validate:  # validate a trained model
        trainer.validate(learner)  # TODO: check if this works properly
    else:  # train a model
        trainer.fit(learner)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    # Trainer arguments.
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to train on.")
    parser.add_argument("--precision", type=int, default=32, help="How many bits of floating point precision to use.")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing epsilon (0.0 indicates no smoothing)")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--num_sanity_val_steps", type=int, default=2)
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--val_check_interval", type=int, default=10,
                        help="After how many train batches to run validation")

    # Program arguments.
    parser.add_argument("--trained_model_path", type=str,
                        help=("Path to a model checkpoint, which will be used as a "
                              "starting point for MAML/MetaHTR."))
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--use_aachen_splits", action="store_true", default=False)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--debug_mode", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)

    parser = LitFullPageHTREncoderDecoder.add_model_specific_args(parser)
    parser = MetaHTR.add_model_specific_args(parser)

    args = parser.parse_args()
    main(args)
    # fmt: on
