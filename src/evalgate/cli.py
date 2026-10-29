from pathlib import Path
from typing import Annotated, NoReturn

import openai
import typer

from evalgate.errors import EvalgateConfigError
from evalgate.providers.openai import OpenAIClient
from evalgate.run import run_feature

app = typer.Typer(help="A regression gate for LLM-powered features.", no_args_is_help=True)

CONFIG_ERROR_EXIT_CODE = 2


# Without a callback, Typer turns a lone command into the whole app and `evalgate run` breaks.
@app.callback()
def main() -> None:
    pass


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="The feature's feature.yaml.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write the run record.")] = Path("runs"),
) -> None:
    """Make a run of a feature under test over its golden dataset."""
    try:
        openai_client = OpenAIClient()
    except openai.OpenAIError as error:
        _exit_with_config_error(f"cannot create the OpenAI client: {error}")

    try:
        record_path = run_feature(config, llm=openai_client, out_dir=out_dir)
    except EvalgateConfigError as error:
        _exit_with_config_error(str(error))

    typer.echo(f"Run record written to {record_path}")


def _exit_with_config_error(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(CONFIG_ERROR_EXIT_CODE)
