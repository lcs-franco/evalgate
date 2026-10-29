from pathlib import Path
from typing import Annotated, NoReturn

import openai
import typer

from evalgate.errors import EvalgateConfigError
from evalgate.providers.openai import OpenAIClient
from evalgate.run import run_feature

app = typer.Typer(help="A regression gate for LLM-powered features.", no_args_is_help=True)

CONFIG_ERROR = 2


@app.callback()
def main() -> None:
    """Keeps `run` a subcommand while it is the only one."""


@app.command()
def run(
    config: Annotated[Path, typer.Argument(help="The feature's feature.yaml.")],
    out_dir: Annotated[Path, typer.Option(help="Where to write the run record.")] = Path("runs"),
) -> None:
    """Make a run of a feature under test over its golden dataset."""
    try:
        llm = OpenAIClient()
    except openai.OpenAIError as error:
        _fail(f"cannot create the OpenAI client: {error}")
    try:
        record_path = run_feature(config, llm=llm, out_dir=out_dir)
    except EvalgateConfigError as error:
        _fail(str(error))
    typer.echo(f"Run record written to {record_path}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(CONFIG_ERROR)
