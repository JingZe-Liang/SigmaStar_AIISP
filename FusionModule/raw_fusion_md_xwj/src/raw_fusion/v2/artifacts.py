"""Content-addressed, owner-aware artifact I/O for V2."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from typing import Protocol

from .schemas.common import ContractError, validate_artifact_mapping, require_exact_keys


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class ArrayRef:
    path: Path
    sha256: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OwnedArtifactRef:
    ref: ArtifactRef
    owner_json: Path
    allowed_root: Path
    allowed_root_identity: tuple[int, int] | None = None

    @property
    def resolved_path(self) -> Path:
        return resolve_owned_path(self.owner_json, self.allowed_root, self.ref.path)


@dataclass(slots=True)
class ArtifactFileSnapshotV2:
    ref: OwnedArtifactRef
    descriptor: int
    size: int

    def __enter__(self) -> "ArtifactFileSnapshotV2":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def pread_exact(self, offset: int, size: int) -> bytes:
        if self.descriptor < 0:
            raise ContractError("snapshot is closed")
        if offset < 0 or size < 0 or offset + size > self.size:
            raise ContractError("snapshot read is outside the captured payload")
        try:
            payload = os.pread(self.descriptor, size, offset)
        except OSError as error:
            raise ContractError("cannot read snapshot") from error
        if len(payload) != size:
            raise ContractError("snapshot read is short")
        return payload

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


class ArtifactSnapshotReaderV2(Protocol):
    """Read artifacts from verified private snapshots."""

    def read_bytes(self, ref: OwnedArtifactRef, context: str) -> bytes: ...

    def open_snapshot(
        self,
        ref: OwnedArtifactRef,
        context: str,
        *,
        expected_size: int | None = None,
    ) -> ArtifactFileSnapshotV2: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ContractError(f"cannot read artifact: {path}") from error
    return digest.hexdigest()


def load_json_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {path}")
    return value


def parse_artifact_ref(value: Mapping[str, object]) -> ArtifactRef:
    validate_artifact_mapping(value)
    path = Path(str(value["path"]))
    if path.is_absolute() or any(part == "" for part in path.parts):
        raise ContractError("ArtifactRef.path must be a relative path")
    return ArtifactRef(path=path, sha256=str(value["sha256"]))


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_owned_path(owner_json: Path, allowed_root: Path, ref_path: Path) -> Path:
    root = Path(allowed_root).resolve()
    owner = Path(owner_json).resolve()
    candidate = (owner.parent / ref_path).resolve()
    if not _within(owner, root) or not _within(candidate, root):
        raise ContractError("outside artifact root")
    return candidate


def bind_artifact_ref(owner_json: Path, allowed_root: Path, ref: ArtifactRef) -> OwnedArtifactRef:
    root = Path(allowed_root).resolve()
    root_identity = _capture_allowed_root_identity(root)
    resolved = resolve_owned_path(owner_json, root, ref.path)
    if not resolved.is_file():
        raise ContractError(f"artifact is not a regular file: {resolved}")
    actual = sha256_file(resolved)
    if actual != ref.sha256:
        raise ContractError(f"artifact SHA-256 mismatch: {resolved}")
    return OwnedArtifactRef(
        ref=ref,
        owner_json=Path(owner_json).resolve(),
        allowed_root=root,
        allowed_root_identity=root_identity,
    )


def bind_top_level_artifact(path: Path, allowed_root: Path) -> OwnedArtifactRef:
    resolved = Path(path).resolve()
    if not _within(resolved, Path(allowed_root).resolve()):
        raise ContractError("outside artifact root")
    synthetic_owner = resolved.parent / ".top_level_cli_owner.json"
    return bind_artifact_ref(synthetic_owner, allowed_root, ArtifactRef(Path(resolved.name), sha256_file(resolved)))


def _is_canonical_absolute_path(path: Path) -> bool:
    return path.is_absolute() and path.anchor == os.sep and all(component not in (".", "..") for component in path.parts)


def _owned_artifact_components(ref: OwnedArtifactRef) -> tuple[Path, tuple[str, ...]]:
    root = Path(ref.allowed_root)
    owner = Path(ref.owner_json)
    if ref.allowed_root_identity is None:
        raise ContractError("verified artifact snapshots require a bound allowed-root identity")
    if not _is_canonical_absolute_path(root):
        raise ContractError("allowed root path must be canonical")
    if not _is_canonical_absolute_path(owner):
        raise ContractError("outside artifact root: owner JSON path must be canonical")
    if ref.ref.path.is_absolute():
        raise ContractError("outside artifact root")
    try:
        owner_parent = owner.parent.relative_to(root)
    except ValueError as error:
        raise ContractError("outside artifact root") from error

    components = list(owner_parent.parts)
    for component in ref.ref.path.parts:
        if component in ("", "."):
            continue
        if component == "..":
            if not components:
                raise ContractError("outside artifact root")
            components.pop()
        else:
            components.append(component)
    if not components:
        raise ContractError("artifact path must name a regular file")
    return root, tuple(components)


def _directory_identity(descriptor: int, context: str) -> tuple[int, int]:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ContractError(f"cannot inspect artifact root for {context}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ContractError(f"artifact root is not a directory: {context}")
    return metadata.st_dev, metadata.st_ino


def _open_nofollow_directory(path: Path, context: str) -> int:
    try:
        nofollow = os.O_NOFOLLOW
    except AttributeError as error:
        raise ContractError("O_NOFOLLOW is required for verified artifact snapshots") from error
    cloexec = getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, os.O_RDONLY | os.O_DIRECTORY | nofollow | cloexec)
    except OSError as error:
        raise ContractError(f"cannot open artifact root for {context}") from error


def _capture_allowed_root_identity(root: Path) -> tuple[int, int]:
    descriptor = _open_nofollow_directory(root, "binding")
    try:
        return _directory_identity(descriptor, "binding")
    finally:
        os.close(descriptor)


def _open_owned_nofollow_descriptor(ref: OwnedArtifactRef, context: str) -> int:
    root, components = _owned_artifact_components(ref)
    nofollow = os.O_NOFOLLOW
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory = _open_nofollow_directory(root, context)

    descriptor = -1
    try:
        if _directory_identity(directory, context) != ref.allowed_root_identity:
            raise ContractError(f"allowed root changed during capture: {context}")
        for index, component in enumerate(components):
            is_final = index == len(components) - 1
            flags = os.O_RDONLY | nofollow | cloexec
            if not is_final:
                flags |= os.O_DIRECTORY
            else:
                flags |= os.O_NONBLOCK
            descriptor = os.open(component, flags, dir_fd=directory)
            if is_final:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ContractError(f"artifact is not a regular file: {context}")
                result = descriptor
                descriptor = -1
                return result
            os.close(directory)
            directory = descriptor
            descriptor = -1
    except ContractError:
        raise
    except OSError as error:
        raise ContractError(f"cannot open artifact for {context}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory)
    raise AssertionError("artifact path must have a final component")


def _identity_and_size(descriptor: int, context: str) -> tuple[int, int, int]:
    try:
        metadata = os.fstat(descriptor)
    except OSError as error:
        raise ContractError(f"cannot inspect artifact for {context}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"artifact is not a regular file: {context}")
    return metadata.st_dev, metadata.st_ino, metadata.st_size


def _create_private_snapshot_descriptor(context: str) -> int:
    try:
        flags = os.MFD_ALLOW_SEALING | getattr(os, "MFD_CLOEXEC", 0)
        return os.memfd_create("raw-fusion-v2-snapshot", flags)
    except (AttributeError, OSError) as error:
        raise ContractError(f"cannot create immutable private snapshot for {context}") from error


def _seal_private_snapshot(descriptor: int, context: str) -> None:
    try:
        os.fsync(descriptor)
        seals = fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        os.lseek(descriptor, 0, os.SEEK_SET)
    except (AttributeError, OSError) as error:
        raise ContractError(f"cannot finalize immutable private snapshot for {context}") from error


def _copy_verified_descriptor_to_private_snapshot(
    source_descriptor: int,
    ref: OwnedArtifactRef,
    context: str,
    expected_size: int | None,
) -> ArtifactFileSnapshotV2:
    identity, inode, source_size = _identity_and_size(source_descriptor, context)
    snapshot_descriptor = -1
    digest = hashlib.sha256()
    remaining = source_size
    copied_size = 0
    try:
        try:
            snapshot_descriptor = _create_private_snapshot_descriptor(context)
        except ContractError:
            raise

        while remaining:
            try:
                payload = os.read(source_descriptor, min(1024 * 1024, remaining))
            except OSError as error:
                raise ContractError(f"cannot read artifact for {context}") from error
            if not payload:
                raise ContractError(f"artifact read is short during capture: {context}")
            digest.update(payload)
            copied_size += len(payload)
            remaining -= len(payload)
            write_offset = 0
            while write_offset < len(payload):
                try:
                    written = os.write(snapshot_descriptor, payload[write_offset:])
                except OSError as error:
                    raise ContractError(f"cannot write private snapshot for {context}") from error
                if written <= 0:
                    raise ContractError(f"private snapshot write is short: {context}")
                write_offset += written

        try:
            if os.read(source_descriptor, 1):
                raise ContractError(f"artifact size changed during capture: {context}")
        except OSError as error:
            raise ContractError(f"cannot read artifact for {context}") from error

        if _identity_and_size(source_descriptor, context) != (identity, inode, source_size):
            raise ContractError(f"artifact changed during capture: {context}")
        recheck_descriptor = _open_owned_nofollow_descriptor(ref, context)
        try:
            if _identity_and_size(recheck_descriptor, context) != (identity, inode, source_size):
                raise ContractError(f"artifact changed during capture: {context}")
        finally:
            os.close(recheck_descriptor)

        if copied_size != source_size or digest.hexdigest() != ref.ref.sha256:
            raise ContractError(f"artifact SHA-256 mismatch during capture: {context}")
        if expected_size is not None and copied_size != expected_size:
            raise ContractError(f"artifact size mismatch during capture: {context}")
        _seal_private_snapshot(snapshot_descriptor, context)
        return ArtifactFileSnapshotV2(ref=ref, descriptor=snapshot_descriptor, size=copied_size)
    except BaseException:
        if snapshot_descriptor >= 0:
            os.close(snapshot_descriptor)
        raise


def open_verified_artifact_snapshot(
    ref: OwnedArtifactRef,
    context: str,
    *,
    expected_size: int | None = None,
) -> ArtifactFileSnapshotV2:
    """Capture the declared child payload under a bound root identity.

    ``owner_json`` supplies only the logical parent for ``ref.path``. This
    primitive does not capture or verify owner JSON bytes; callers must derive
    child refs from a separately captured owner snapshot.
    """
    source_descriptor = _open_owned_nofollow_descriptor(ref, context)
    try:
        return _copy_verified_descriptor_to_private_snapshot(source_descriptor, ref, context, expected_size)
    finally:
        os.close(source_descriptor)


def read_verified_artifact_bytes(ref: OwnedArtifactRef, context: str) -> bytes:
    with open_verified_artifact_snapshot(ref, context) as snapshot:
        return snapshot.pread_exact(0, snapshot.size)


def load_verified_json_object(ref: OwnedArtifactRef, context: str) -> Mapping[str, object]:
    try:
        value = json.loads(read_verified_artifact_bytes(ref, context))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot load verified JSON object: {context}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON object required: {context}")
    return value


def bind_child_artifact_ref(parent: OwnedArtifactRef, value: Mapping[str, object]) -> OwnedArtifactRef:
    return bind_artifact_ref(parent.resolved_path, parent.allowed_root, parse_artifact_ref(value))


def rebase_artifact_ref(target: OwnedArtifactRef, new_owner_json: Path) -> ArtifactRef:
    target_path = resolve_owned_path(target.owner_json, target.allowed_root, target.ref.path)
    new_owner = Path(new_owner_json).resolve()
    if not _within(new_owner, target.allowed_root):
        raise ContractError("outside artifact root")
    relative = Path(os.path.relpath(target_path, start=new_owner.parent)).as_posix()
    return ArtifactRef(path=Path(relative), sha256=target.ref.sha256)


def read_artifact_json(ref: OwnedArtifactRef) -> Mapping[str, object]:
    return load_json_object(ref.resolved_path)


def publish_directory_atomic(destination: Path, validator: Callable[[Path], object] | Callable[[Path], object], *args, build: Callable[[Path], object] | None = None) -> Path:
    # Accept both publish_directory_atomic(destination, validator, build=...) and
    # the natural writer/validator positional form used by producers.
    if args:
        if len(args) != 1 or build is not None:
            raise TypeError("publish_directory_atomic accepts one validator and one writer")
        writer, validator = validator, args[0]
        build = writer
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ContractError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        if build is not None:
            build(temporary)
        result = validator(temporary)
        if result is False:
            raise ContractError("temporary directory validation failed")
        for path in sorted(temporary.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        os.replace(temporary, destination)
        fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_training_context_v2(mapping: Mapping[str, object], *, model_factory, optimizer_factory):
    from .schemas.runtime import validate_checkpoint_v2

    validate_checkpoint_v2(mapping)
    model = model_factory(mapping)
    optimizer = optimizer_factory(model)
    return model, optimizer
