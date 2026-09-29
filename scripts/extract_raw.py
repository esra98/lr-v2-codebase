#!/usr/bin/env python
"""Extract dataset tarballs into the project. The source folder is only read."""

import argparse
import glob
import json
import os
import sys
import tarfile


def is_safe_member(member, dest):
    """Reject absolute paths, parent traversal and links."""
    if member.issym() or member.islnk():
        return False
    target = os.path.realpath(os.path.join(dest, member.name))
    return target.startswith(os.path.realpath(dest) + os.sep)


def _load_owners(dest):
    path = os.path.join(dest, ".extracted", "owners.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_owners(dest, owners):
    path = os.path.join(dest, ".extracted", "owners.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(owners, f, indent=2, sort_keys=True)
    os.replace(path + ".tmp", path)


def extract(tar_path, dest):
    """Extract one tarball. Each top-level folder becomes one session.

    Two tarballs that contain a folder of the same name would silently mix
    two recordings under one session, so that is refused.
    """
    name = os.path.basename(tar_path)
    done_marker = os.path.join(dest, ".extracted", name + ".done")
    if os.path.isfile(done_marker):
        print(f"skip  {name} (already extracted)")
        return
    owners = _load_owners(dest)
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not is_safe_member(member, dest):
                print(f"  refusing unsafe member {member.name!r}", file=sys.stderr)
                continue
            session = member.name.split("/", 1)[0]
            owner = owners.get(session)
            if owner is None:
                if os.path.exists(os.path.join(dest, session)):
                    sys.exit(f"{name}: folder {session!r} already exists in {dest} and did not come from a tarball")
                owners[session] = name
                _save_owners(dest, owners)
            elif owner != name:
                sys.exit(f"{name}: folder {session!r} was already extracted from {owner}")
            tar.extract(member, dest, filter="data")
            count += member.isfile()
    os.makedirs(os.path.dirname(done_marker), exist_ok=True)
    open(done_marker, "w").close()
    print(f"done  {name} ({count} files)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, help="Folder holding video-*.tar.gz (read only)")
    parser.add_argument("--dest", default="data/raw")
    parser.add_argument("--only", nargs="*", help="Tarball names to extract, e.g. video-2.tar.gz")
    args = parser.parse_args()

    src, dest = os.path.realpath(args.src), os.path.realpath(args.dest)
    if dest == src or dest.startswith(src + os.sep):
        sys.exit("Refusing to extract into the source dataset folder.")
    os.makedirs(dest, exist_ok=True)

    tarballs = sorted(glob.glob(os.path.join(src, "*.tar.gz")))
    if args.only:
        tarballs = [t for t in tarballs if os.path.basename(t) in set(args.only)]
    if not tarballs:
        sys.exit(f"No tarballs found in {src}")
    for tar_path in tarballs:
        extract(tar_path, dest)


if __name__ == "__main__":
    main()
