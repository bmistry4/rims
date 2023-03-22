from pathlib import Path

from pytorch_lightning.loggers import WandbLogger


def create_experiment_name(args, addon: str = "", joiner="_") -> str:
    """
    Creates the experiment name based on the model config
    Experiment name will be of form:
        <task type>_<ID>_<model type>_<seed><_postfix>

    :param args: parser args
    :param addon: addition info to add to name at the end
    :param joiner: connector symbol to join the substrings together
    :return: experiment name
    """
    if addon != "":
        addon = joiner + addon

    # wandb doesn't like / for their run id's so it's got to be replaced
    prefix = args.name_prefix.replace('/', '-')
    return f"{prefix}{joiner}" \
           f"ID{args.id}{joiner}" \
           f"{args.cell}{joiner}" \
           f"S{args.seed}{addon}"


def setup_wandb_logger(run_log_path: str, args):
    """
    Create a weights and biases logger and the folder where results are stored locally.
    :param run_log_path: directory path where the local results will be stored (up to the task type subfolder)
    :param args: parser args
    :return: logger
    """
    if not args.logger_off:
        exp_name = create_experiment_name(args, addon=args.name_postfix)
        Path(run_log_path).mkdir(parents=True, exist_ok=True)
        print(f"\nLOGGING DIRECTORY AT: {run_log_path}\n")

        # setting id will set the local save folder name. Save folder name: run-<yyyymmdd_hhmmss>-<exp_name>
        logger = WandbLogger(project=args.wandb_project, name=exp_name, save_dir=run_log_path, id=exp_name,
                             notes=args.wandb_notes)

        # Extra logging options
        # logger.watch(model)                   # log gradients and model topology
        # logger.watch(model, log="all")        # log gradients, parameter histogram and model topology
        # logger.watch(model, log_freq=500)     # change log frequency of gradients and parameters (default = 100 steps)
        # logger.watch(model, log_graph=False)  # do not log graph (in case of errors)
    else:
        logger = False

    return logger
