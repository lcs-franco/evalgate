from pydantic import ValidationError


class EvalgateConfigError(Exception):
    pass


class InvalidOutputError(Exception):
    def __init__(self, reason: str, raw_output: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_output = raw_output


def validation_error_lines(error: ValidationError, location_prefix: str = "") -> list[str]:
    lines = []

    for detail in error.errors():
        location_parts = (location_prefix, *detail["loc"])
        location = ".".join(str(part) for part in location_parts if part != "")

        lines.append(f"{location}: {detail['msg']}" if location else detail["msg"])

    return lines
