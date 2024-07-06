import sys
import os
import platform
import glob

import logging

from types import SimpleNamespace

import re
import yaml

import urllib

from pathlib import Path

LOGGING_NAME = "dodcicd"
RANK = int(os.getenv("RANK", "-1"))
MACOS, LINUX, WINDOWS = (
    platform.system() == x for x in ["Darwin", "Linux", "Windows"]
)  # environment booleans
VERBOSE = True  # global verbose mode


def set_logging(name="LOGGING_NAME", verbose=True):
    """Sets up logging for the given name with UTF-8 encoding support, ensuring compatibility across different
    environments.
    """
    level = (
        logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR
    )  # rank in world for Multi-GPU trainings

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":

        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))

        try:
            # Attempt to reconfigure stdout to use UTF-8 encoding if possible
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            # For environments where reconfigure is not available, wrap stdout in a TextIOWrapper
            elif hasattr(sys.stdout, "buffer"):
                import io

                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(f"{name}.log")
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    f_handler.setLevel(level)
    f_handler.setFormatter(formatter)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(f_handler)
    logger.propagate = False
    return logger


# Set logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)

for logg in "sentry_sdk", "urllib3.connectionpool":
    logging.getLogger(logg).setLevel(logging.CRITICAL + 1)


def emojis(string=""):
    """Return platform-dependent emoji-safe version of string."""
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    LOGGER.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
ASSETS = ROOT / "assets"  # default images
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"


def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {
        ".yaml",
        ".yml",
    }, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(
                r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+",
                "",
                s,
            )

        # Add YAML filename to dict and return
        data = (
            yaml.safe_load(s) or {}
        )  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data


class IterableSimpleNamespace(SimpleNamespace):
    """IterableSimpleNamespace is an extension class of SimpleNamespace that adds iterable functionality and
    enables usage with dict() and for loops.
    """

    def __iter__(self):
        """Return an iterator of key-value pairs from the namespace's attributes."""
        return iter(vars(self).items())

    def __str__(self):
        """Return a human-readable string representation of the object."""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'.
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = (
        Path(url).as_posix().replace(":/", "://")
    )  # Pathlib turns :// -> :/, as_posix() for Windows
    return urllib.parse.unquote(url).split("?")[
        0
    ]  # '%2F' to '/', split https://url.com/file.txt?auth


def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def check_suffix(file="yolov8n.pt", suffix=".pt", msg=""):
    """Check file(s) for acceptable suffix."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix,)
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}, not {s}"


def check_file(file, suffix="", hard=True):
    """Search/download file (if necessary) and return path."""
    check_suffix(file, suffix)  # optional
    file = str(file).strip()  # convert to string and strip spaces
    if (
        not file
        or (
            "://" not in file and Path(file).exists()
        )  # '://' check required in Windows Python<3.10
        or file.lower().startswith("grpc://")
    ):  # file exists or gRPC Triton images
        return file
    else:  # search
        files = glob.glob(str(ROOT / "**" / file), recursive=True) or glob.glob(
            str(ROOT.parent / file)
        )  # find file
        if not files and hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        elif len(files) > 1 and hard:
            raise FileNotFoundError(
                f"Multiple files match '{file}', specify exact path: {files}"
            )
        return files[0] if len(files) else []  # return file


def check_yaml(file, suffix=(".yaml", ".yml"), hard=True):
    """Search/download YAML file (if necessary) and return path, checking suffix."""
    return check_file(file, suffix, hard=hard)
