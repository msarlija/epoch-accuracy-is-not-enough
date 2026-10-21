# -*- coding: utf-8 -*-
"""Stream one recording at a time from S3 into a single reused temp path.

Two cohorts (WSC, STAGES) are not on local disk; they live in NSRR / Stanford S3
buckets. Downloading them wholesale is not an option — WSC alone is thousands of
multi-hundred-MB PSGs — so each recording is fetched to a FIXED path, read,
preprocessed, and then overwritten by the next one. Disk use stays at one
recording regardless of corpus size, which is how the old pipeline did it
(``C:\\Data\\WSC\\temp\\temp.edf``) and why a stray ``temp.edf`` is sitting in the
STAGES directory.

That makes ``prepare_data.py``'s resume behaviour load-bearing rather than
merely convenient: an interrupted run costs one download, and re-running skips
every recording already written to the corpus.

Credentials are never handled here. boto3 picks them up from the environment
(``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``, or a profile), so no key
appears in this repo, in a config, or in a log line.
"""

import glob
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# boto3 at DEBUG emits dozens of lines per transfer, which buries the
# per-recording progress that prepare_data -v exists to show.
for _noisy in ("boto3", "botocore", "s3transfer", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_CLIENTS: dict = {}


class S3ObjectMissing(RuntimeError):
    """The bucket has no such key — a normal outcome for a partial cohort."""


def s3_client(region: Optional[str] = None):
    """Cached boto3 S3 client. Credentials come from the environment."""
    import boto3
    key = region or "_default"
    if key not in _CLIENTS:
        missing = [v for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
                   if not os.environ.get(v)]
        if missing:
            # Not fatal: a profile or instance role may still supply them, and
            # boto3 will say so if not. Worth a warning because the usual cause
            # is simply forgetting to export them.
            log.warning("S3: %s not in the environment; relying on a boto3 "
                        "profile or instance role", ", ".join(missing))
        _CLIENTS[key] = boto3.client("s3", **({"region_name": region}
                                              if region else {}))
    return _CLIENTS[key]


def download_to(bucket: str, key: str, dest: str,
                region: Optional[str] = None) -> str:
    """Fetch ``key`` to ``dest``, overwriting it. Returns ``dest``.

    Downloads to a sibling ``.part`` file first and renames on success, so an
    interrupted transfer cannot leave a truncated EDF at ``dest`` for the next
    run to read as if it were complete.
    """
    from botocore.exceptions import ClientError

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    # Sweep stale intermediates before starting. Two layers create them: this
    # function's own `.part`, and s3transfer's `<dest>.part.<random>` scratch
    # file. A killed transfer leaves either behind, and since the random suffix
    # is unpredictable a same-name cleanup would never find it -- so match the
    # prefix instead. Only ever touches temp slots, never a corpus file.
    for stale in glob.glob(part + "*"):
        try:
            os.remove(stale)
            log.info("removed stale transfer file %s", os.path.basename(stale))
        except OSError:
            pass
    client = s3_client(region)
    try:
        client.download_file(bucket, key, part)
    except ClientError as err:
        code = err.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "403", "AccessDenied"):
            if os.path.exists(part):
                os.remove(part)
            raise S3ObjectMissing(f"s3://{bucket}/{key}: {code}") from None
        raise
    os.replace(part, dest)
    size_mb = os.path.getsize(dest) / 1e6
    log.info("s3://%s/%s -> %s (%.1f MB)", bucket, key,
             os.path.basename(dest), size_mb)
    return dest


def temp_paths(root: str, *suffixes: str) -> dict:
    """``{suffix: <root>/temp<suffix>}`` — the fixed slots a cohort reuses.

    e.g. ``temp_paths(root, ".edf", ".txt")`` ->
    ``{'.edf': '<root>/temp.edf', '.txt': '<root>/temp.txt'}``
    """
    return {s: os.path.join(root, "temp" + s) for s in suffixes}
