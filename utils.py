import os
import wandb


def wandb_init(lr0, epochs, batch_size, project="mp242"):
    api_key = os.getenv("WANDB_API_KEY", None)

    if api_key is None:
        try:
            from kaggle_secrets import UserSecretsClient

            api_key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            api_key = None

    # Skip if no api_key
    if not api_key:
        print("WANDB_API_KEY not found; skipping wandb.init()")
        return None

    # os.environ["WANDB_API_KEY"] = api_key
    wandb.login(key=api_key)

    logger = wandb.init(
        project=project,
        config={"lr0": lr0, "epochs": epochs, "batch_size": batch_size},
    )
    logger.define_metric("eval/precision", summary="max")
    logger.define_metric("eval/recall", summary="max")
    logger.define_metric("eval/mAP50", summary="max")
    logger.define_metric("eval/mAP5095", summary="max")

    return logger
