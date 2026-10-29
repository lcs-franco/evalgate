from pydantic import ValidationError


class EvalgateConfigError(Exception):
    """A feature configuration, prompt version or golden dataset is invalid.

    The message is meant for the person who wrote the file, so it names the file and the problem.
    """


class InvalidOutputError(Exception):
    """The model output could not be parsed or validated against the feature's output type."""

    def __init__(self, reason: str, raw: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw


def describe_validation_error(error: ValidationError, prefix: str = "") -> list[str]:
    """One line per problem, such as `expected.category: Input should be 'billing' or 'general'`."""
    lines = []
    for detail in error.errors():
        location = ".".join(str(part) for part in (prefix, *detail["loc"]) if part != "")
        lines.append(f"{location}: {detail['msg']}" if location else detail["msg"])
    return lines
