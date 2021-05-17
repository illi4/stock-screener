import logging
import sys
import os
import zipfile
import logging.handlers


def rotation_namer(name):
    """
    Namer to return the filename plus extension
    """
    return name + ".zip"


def rotator(source, dest):
    """
    Logs rotator to compress the file and remove the compressed one
    """
    print(f"compressing {source} -> {dest}")
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source)
    os.remove(source)


def get_logger(name="main", debug_level=logging.DEBUG, filename="log.log"):
    """
    Create a logger

    :param app: service name (e.g. 'algo')
    :param debug_level: logging debug level
    :param filename: created file name
    :return: logger object
    """

    # Trying a reworked logger

    formatter = logging.Formatter("%(asctime)s: %(message)s")

    # Default is to compress every 3 days
    main_handler = logging.handlers.TimedRotatingFileHandler(
        filename, when="D", interval=7, encoding="utf-8", backupCount=6, utc=False
    )
    main_handler.rotator = rotator
    main_handler.namer = rotation_namer
    main_handler.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(debug_level)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(debug_level)

    logger.addHandler(main_handler)
    logger.addHandler(ch)

    return logger


# Define logger here to be reused across modules
log_object = get_logger()
