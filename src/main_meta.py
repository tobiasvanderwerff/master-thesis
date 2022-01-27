import argparse
import shutil
from copy import copy
from pathlib import Path
from functools import partial

from models import *
from lit_models import MetaHTR
from data import IAMDataset
from lit_util import MetaHTRCheckpointIO, LitProgressBar
from lit_callbacks import (
    LogModelPredictionsMetaHTR,
    LogLayerWiseLearningRates,
    LogInstanceSpecificWeights,
    LogWorstPredictionsMetaHTR,
)
from util import (
    filter_df_by_freq,
    LabelEncoder,
    PtTaskDataset,
    freeze_batchnorm_weights,
)

import learn2learn as l2l
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ModelSummary
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.core.saving import load_hparams_from_yaml


LOGGING_DIR = "lightning_logs/"
PREDICTIONS_TO_LOG = {
    "word": 8,
    "line": 6,
    "form": 1,
}


def main(args):

    seed_everything(args.seed)

    log_dir_root = Path(__file__).parent.parent.resolve()
    tb_logger = pl_loggers.TensorBoardLogger(
        str(log_dir_root / LOGGING_DIR), name="", version=args.experiment_name
    )

    assert Path(
        args.trained_model_path
    ).is_file(), f"{args.trained_model_path} does not point to a file."

    # Load the label encoder for the trained model.
    model_path = Path(args.trained_model_path)
    le_path_1 = model_path.parent.parent / "label_encoding.txt"
    le_path_2 = model_path.parent.parent / "label_encoder.pkl"
    assert le_path_1.is_file() or le_path_2.is_file(), (
        f"Label encoder file not found at {le_path_1} or {le_path_2}. "
        f"Make sure 'label_encoding.txt' exists in the lightning_logs directory."
    )
    le_path = le_path_2 if le_path_2.is_file() else le_path_1
    label_enc = LabelEncoder().read_encoding(le_path)

    # Save the label encoder in the logging directory.
    log_dir = Path(tb_logger.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    label_enc.dump(log_dir)

    # Copy hyper-parameters. The loaded model has an associated `hparams.yaml` file,
    # which we copy to the current logging directory so that we can load the model
    # later using the saved hyper parameters.
    _model_path = Path(args.trained_model_path)
    if (_model_path.parent.parent / "model_hparams.yaml").is_file():
        # MetaHTR checkpoint.
        fphtr_hparams_file = str(_model_path.parent.parent / "model_hparams.yaml")
    else:  # FPHTR checkpoint.
        fphtr_hparams_file = str(_model_path.parent.parent / "hparams.yaml")
    shutil.copy(fphtr_hparams_file, log_dir / "model_hparams.yaml")
    hparams = load_hparams_from_yaml(fphtr_hparams_file)
    only_lowercase = hparams["only_lowercase"]
    augmentations = "train" if args.use_image_augmentations else "val"

    ds = IAMDataset(
        args.data_dir,
        "word",
        "train",
        label_enc=label_enc,
        return_writer_id=True,
        only_lowercase=only_lowercase,
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
        test_splits = (aachen_path / "test.uttlist").read_text().splitlines()

        data_train = ds.data[ds.data["img_id"].isin(train_splits)]
        data_val = ds.data[ds.data["img_id"].isin(validation_splits)]
        data_test = ds.data[ds.data["img_id"].isin(test_splits)]

        ds_train = copy(ds)
        ds_train.data = data_train

        ds_val = copy(ds)
        ds_val.data = data_val

        ds_test = copy(ds)
        ds_test.data = data_test
    else:
        ds_train, ds_val = torch.utils.data.random_split(
            ds, [math.ceil(0.8 * len(ds)), math.floor(0.2 * len(ds))]
        )
        ds_val.dataset = copy(ds)

    # Exclude writers from the dataset that do not have sufficiently many samples.
    ds_train.data = filter_df_by_freq(ds_train.data, "writer_id", args.shots * 2)
    ds_val.data = filter_df_by_freq(ds_val.data, "writer_id", args.shots * 2)
    ds_test.data = filter_df_by_freq(ds_test.data, "writer_id", args.shots * 2)

    # Set image transforms.
    if args.use_aachen_splits:
        ds_train.set_transforms_for_split(augmentations)
        ds_val.set_transforms_for_split("val")
        ds_test.set_transforms_for_split("test")
    else:
        ds_train.dataset.set_transforms_for_split(augmentations)
        ds_val.dataset.set_transforms_for_split("val")

    # Intersection of writer sets should be the empty set, thus length 0.
    assert (
        len(set(ds_train.writer_ids) & set(ds_val.writer_ids) & set(ds_test.writer_ids))
        == 0
    )

    # Initalize cache directory.
    cache_dir = Path(args.cache_dir) if args.cache_dir else log_dir / "cache"
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Setting the _bookkeeping_path attribute will make the MetaDataset instance
    # load its label-index mapping from a file, rather than creating it (which takes a
    # long time). If the path does not exists, the bookkeeping will be created and
    # stored on disk afterwards.
    ds_train._bookkeeping_path = cache_dir / "train_l2l_bookkeeping.pkl"
    ds_val._bookkeeping_path = cache_dir / "val_l2l_bookkeeping.pkl"
    ds_test._bookkeeping_path = cache_dir / "test_l2l_bookkeeping.pkl"

    ds_meta_train = l2l.data.MetaDataset(ds_train)
    ds_meta_val = l2l.data.MetaDataset(ds_val)
    ds_meta_test = l2l.data.MetaDataset(ds_test)

    ds_train = ds_meta_train.dataset
    ds_val = ds_meta_val.dataset
    ds_test = ds_meta_test.dataset

    eos_tkn_idx, sos_tkn_idx, pad_tkn_idx = ds_train.label_enc.transform(
        [ds_train._eos_token, ds_train._sos_token, ds_train._pad_token]
    )
    collate_fn = partial(
        IAMDataset.collate_fn,
        pad_val=pad_tkn_idx,
        eos_tkn_idx=eos_tkn_idx,
        dataset_returns_writer_id=True,
    )

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
    test_tsk_trnsf = [
        l2l.data.transforms.NWays(ds_meta_test, n=args.ways),
        l2l.data.transforms.LoadData(ds_meta_test),
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
    taskset_test = l2l.data.TaskDataset(
        ds_meta_test,
        test_tsk_trnsf,
        num_tasks=-1,
        task_collate=collate_fn,
    )

    # Wrap the task datasets into a simple class that sets a length for the dataset
    # (other than 1, which is the default if setting num_tasks=-1).
    # This is necessary because the dataset length is used by Pytorch dataloaders to
    # determine how many batches are in the dataset per epoch.
    taskset_train = PtTaskDataset(
        taskset_train, epoch_length=int(len(ds_train.writer_ids) / args.ways)
    )
    taskset_val = PtTaskDataset(
        taskset_val, epoch_length=int(len(ds_val.writer_ids) / args.ways)
    )
    taskset_test = PtTaskDataset(
        taskset_test, epoch_length=int(len(ds_test.writer_ids) / args.ways)
    )

    # Initialize MAML with a trained FPHTR model.
    learner = MetaHTR.init_with_fphtr_from_checkpoint(
        args.trained_model_path,
        fphtr_hparams_file,
        ds_train.label_enc,
        fphtr_params_to_log={"only_lowercase": only_lowercase},
        load_meta_weights=True,
        taskset_train=taskset_train,
        taskset_val=taskset_val,
        taskset_test=taskset_test,
        ways=args.ways,
        shots=args.shots,
        outer_lr=args.outer_lr,
        num_workers=args.num_workers,
        use_cosine_lr_scheduler=args.use_cosine_lr_scheduler,
        use_instance_weights=(not args.no_instance_weights),
        allow_nograd=args.freeze_batchnorm_gamma,
        num_epochs=args.max_epochs,  # note this can be wrong when using early stopping
        prms_to_log={
            "seed": args.seed,
            "splits": ("Aachen" if args.use_aachen_splits else "random"),
            "max_epochs": args.max_epochs,
            "model_path": args.trained_model_path,
            "gradient_clip_val": args.gradient_clip_val,
            "early_stopping_patience": args.early_stopping_patience,
            "use_cosine_lr_scheduler": args.use_cosine_lr_scheduler,
            "use_image_augmentations": args.use_image_augmentations,
            "freeze_batchnorm_gamma": args.freeze_batchnorm_gamma,
        },
    )
    # learner.freeze_all_layers_except_classifier()
    if args.freeze_batchnorm_gamma:
        freeze_batchnorm_weights(learner, freeze_bias=False)

    # This checkpoint plugin is necessary to save the weights obtained using MAML in
    # the proper way. The weights should be stored in the same format as they would
    # be saved without using MAML, to make it straightforward to load the model
    # weights later on.
    checkpoint_io = MetaHTRCheckpointIO()

    # Prepare fixed batches used for monitoring model predictions during training.
    im, t, wrtrs = next(iter(learner.val_dataloader()))
    # Select the first writer in the batch.
    val_batch = (im[: args.shots * 2], t[: args.shots * 2], wrtrs[: args.shots * 2])
    im, t, wrtrs = next(iter(learner.train_dataloader()))
    train_batch = (im[: args.shots * 2], t[: args.shots * 2], wrtrs[: args.shots * 2])
    assert (
        val_batch[-1].unique().numel() == 1 and val_batch[-1].unique().numel() == 1
    ), "Only one writer should be in the batch for logging."
    val_batch, train_batch = [
        (
            im[: args.shots],
            t[: args.shots],
            im[args.shots : args.shots + PREDICTIONS_TO_LOG["word"]],
            t[args.shots : args.shots + PREDICTIONS_TO_LOG["word"]],
            wrtrs[0],
        )
        for (im, t, wrtrs) in [val_batch, train_batch]
    ]

    callbacks = [
        ModelSummary(max_depth=3),
        LitProgressBar(),
        ModelCheckpoint(
            save_top_k=(-1 if args.save_all_checkpoints else 3),
            mode="min",
            monitor="word_error_rate",
            filename="MAML-{epoch}-{char_error_rate:.4f}-{word_error_rate:.4f}",
            save_weights_only=True,
        ),
        LogModelPredictionsMetaHTR(
            label_encoder=ds_train.label_enc,
            val_batch=val_batch,
            train_batch=train_batch,
            enable_grad=True,
            predict_on_train_start=True,
        ),
        LogLayerWiseLearningRates(),
        LogInstanceSpecificWeights(ds_train.label_enc),
        LogWorstPredictionsMetaHTR(
            train_dataloader=learner.train_dataloader(),
            val_dataloader=learner.val_dataloader(),
            test_dataloader=learner.test_dataloader(),
            training_skipped=(args.validate or args.test),
        ),
    ]
    if args.early_stopping_patience != -1:
        callbacks.append(
            EarlyStopping(
                monitor="word_error_rate",
                patience=args.early_stopping_patience,
                verbose=True,
                mode="min",
                # check_on_train_epoch_end=False,  # check at the end of validation
            )
        )

    trainer = Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        plugins=[checkpoint_io],
        strategy=(
            DDPPlugin(find_unused_parameters=False) if args.num_nodes != 1 else None
        ),  # ddp = Distributed Data Parallel
        gpus=(0 if args.use_cpu else 1),
        log_every_n_steps=10,
        enable_model_summary=False,
        callbacks=callbacks,
    )

    if args.validate:  # validate a trained model
        trainer.validate(learner)
    elif args.test:  # test a trained model
        trainer.test(learner)
    else:  # train a model
        trainer.fit(learner)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--trained_model_path", type=str, required=True,
                        help=("Path to a model checkpoint, which will be used as a "
                              "starting point for MAML/MetaHTR."))
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Number of checks with no improvement after which "
                             "training will be stopped. Setting this to -1 will disable "
                             "early stopping.")
    parser.add_argument("--use_aachen_splits", action="store_true", default=False)
    parser.add_argument("--use_image_augmentations", action="store_true", default=False,
                        help="Whether to use image augmentations during training. For "
                             "MAML this does not seem to be too beneficial so far.")
    parser.add_argument("--save_all_checkpoints", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name, used as the name of the folder in "
                             "which logs are stored.")
    # fmt: on

    parser = MetaHTR.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)  # adds Pytorch Lightning arguments

    args = parser.parse_args()
    main(args)
