from kaggle_secrets import UserSecretsClient
import wandb


def wandb_init(lr0, epochs, batch_size):
    wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    logger = wandb.init(
        project="mp242",
        config={"lr0": lr0, "epochs": epochs, "batch_size": batch_size},
    )
    logger.define_metric("eval/precision", summary="max")
    logger.define_metric("eval/recall", summary="max")
    logger.define_metric("eval/mAP50", summary="max")
    logger.define_metric("eval/mAP5095", summary="max")
    return logger
