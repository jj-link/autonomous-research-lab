"""Cache integrity and ownership regressions; no models or temporary directories."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from autonomous_research_lab.models import LabError, read_json
from autonomous_research_lab.store import Store

MODULE = Path(__file__).resolve().parents[1] / "research/qwen38-nvfp4/model_cache.py"
SPEC = importlib.util.spec_from_file_location("qwen38_model_cache", MODULE)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class CheckpointCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / ".arl-cache/tests" / uuid.uuid4().hex
        self.source = self.folder / "hdd-model"
        (self.source / "nested/deeper").mkdir(parents=True)
        (self.source / "weights.bin").write_bytes(b"approved weights")
        (self.source / "nested/deeper/config.json").write_bytes(b'{"model": "pinned"}')
        (self.source / "empty").write_bytes(b"")
        self.store = Store(self.folder / "state")
        self.original = self.store.register_input(self.source, "qwen38-bf16")
        self.root = self.folder / "native-cache"
        self.checkpoint = {"label": "bf16", "repo_id": "pinned/model", "revision": "a" * 40}
        # Storage probing is covered separately; filesystem fixtures remain portable.
        self.native = patch.object(cache, "_native")
        self.native.start()
        self.capacity = patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=128 * 1024**3))
        self.capacity.start()

    def tearDown(self):
        self.capacity.stop()
        self.native.stop()
        shutil.rmtree(self.folder)

    def stage(self):
        return cache.cache_checkpoint(self.store, self.checkpoint, self.original["id"], self.root)

    def marker(self):
        return read_json(self.root / cache.STATE_NAME)

    def change_marker(self, **changes):
        state = self.marker()
        state.update(changes)
        cache.atomic_json(self.root / cache.STATE_NAME, state)

    def test_copy_checks_pinned_bytes_without_verifying_hdd_source(self):
        (self.source / "unapproved.bin").write_bytes(b"never copy this")
        verify = self.store.verify_input
        with patch.object(self.store, "verify_input", wraps=verify) as checked:
            cached = self.stage()
            checked.assert_not_called()
        self.assertNotEqual(cached["id"], self.original["id"])
        for key in ("files", "bytes", "sha256"):
            self.assertEqual(cached[key], self.original[key])
        self.assertEqual(cached["source"], str(self.root / cache.DATA_NAME))
        self.assertFalse((self.root / cache.DATA_NAME / "unapproved.bin").exists())
        self.assertEqual(self.store.verify_input(cached["id"]), cached)
        self.assertEqual(self.store.get_input(self.original["id"]), self.original)
        with patch.object(self.store, "verify_input", wraps=verify) as checked:
            self.assertEqual(self.stage(), cached)
            checked.assert_called_once_with(cached["id"])

    def test_same_size_hdd_corruption_fails_pinned_hash_and_keeps_owned_partial(self):
        (self.source / "weights.bin").write_bytes(b"modified weights")
        with self.assertRaisesRegex(LabError, "expected SHA-256"):
            self.stage()
        self.assertEqual(self.marker()["status"], "copying")
        self.assertEqual(self.store.list_inputs(), [self.original])
        with self.assertRaisesRegex(LabError, "partial"):
            self.stage()
        cache.release_checkpoint_cache(self.store, self.original["id"], self.root)
        self.assertEqual((self.source / "weights.bin").read_bytes(), b"modified weights")
        self.assertEqual(self.marker()["status"], "released")

    def test_same_size_cached_corruption_is_not_reused_but_can_be_released(self):
        cached = self.stage()
        weights = self.root / cache.DATA_NAME / "weights.bin"
        before = weights.stat()
        weights.write_bytes(b"modified weights")
        os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(LabError, "Approved input changed"):
            self.stage()
        cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertEqual((self.source / "weights.bin").read_bytes(), b"approved weights")

    def test_insufficient_capacity_fails_before_root_or_large_writes(self):
        required = cache._required_space(self.root.parent, self.original)
        with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=required + cache.RESERVE_BYTES - 1)):
            with self.assertRaisesRegex(LabError, "8 GiB"):
                self.stage()
        self.assertFalse(self.root.exists())
        self.assertEqual(self.store.list_inputs(), [self.original])

    def test_capacity_boundary_includes_allocation_and_accepts_exact_reserve(self):
        required = cache._required_space(self.root.parent, self.original)
        self.assertGreaterEqual(required, self.original["bytes"])
        with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=required + cache.RESERVE_BYTES)):
            cached = self.stage()
        self.assertEqual(cached["sha256"], self.original["sha256"])
        with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=cache.RESERVE_BYTES - 1)):
            with self.assertRaisesRegex(LabError, "8 GiB"):
                self.stage()
        with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=cache.RESERVE_BYTES)):
            self.assertEqual(self.stage(), cached)

    def test_release_retains_history_and_is_idempotent_for_same_input(self):
        cached = self.stage()
        history = self.store.list_inputs()
        cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertEqual(set(path.name for path in self.root.iterdir()), {cache.STATE_NAME})
        released = self.marker()
        cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        cache.release_checkpoint_cache(self.store, self.original["id"], self.root)
        self.assertEqual(self.marker(), released)
        self.assertEqual(self.store.list_inputs(), history)
        self.assertEqual(self.store.verify_input(self.original["id"]), self.original)

    def test_cache_identity_getter_tracks_active_and_released_without_rehashing(self):
        self.assertIsNone(cache.cached_input_id(self.root))
        cached = self.stage()
        with patch.object(Store, "verify_input", side_effect=AssertionError("inspection must not admit bytes")):
            self.assertEqual(cache.cached_input_id(self.root), cached["id"])
        cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertIsNone(cache.cached_input_id(self.root))

    def test_cache_identity_getter_rejects_unowned_inconsistent_and_missing_history(self):
        cached = self.stage()
        unknown = self.root / "unknown"
        unknown.write_bytes(b"keep")
        with self.assertRaisesRegex(LabError, "unowned"):
            cache.cached_input_id(self.root)
        unknown.unlink()
        self.change_marker(database=str(self.folder / "missing.sqlite3"))
        with self.assertRaisesRegex(LabError, "history is unavailable"):
            cache.cached_input_id(self.root)
        self.assertFalse((self.folder / "missing.sqlite3").exists())
        self.assertTrue((self.root / cache.DATA_NAME).exists())
        self.assertEqual(self.store.get_input(cached["id"]), cached)

    def test_partial_cache_has_no_active_registered_identity(self):
        with patch.object(self.store, "register_input", side_effect=LabError("interrupted approval")):
            with self.assertRaisesRegex(LabError, "interrupted approval"):
                self.stage()
        self.assertIsNone(cache.cached_input_id(self.root))
        cache.release_checkpoint_cache(self.store, self.original["id"], self.root)
        self.assertIsNone(cache.cached_input_id(self.root))

    def test_new_approval_at_same_source_path_does_not_reuse_old_cache(self):
        self.stage()
        newly_pinned = self.store.register_input(self.source, "same-path-new-approval")
        with self.assertRaisesRegex(LabError, "explicit release"):
            cache.cache_checkpoint(self.store, self.checkpoint, newly_pinned["id"], self.root)

    def test_different_checkpoint_requires_release_and_old_release_cannot_delete_next(self):
        cached = self.stage()
        next_source = self.folder / "next-model"
        next_source.mkdir()
        (next_source / "weights.bin").write_bytes(b"next weights")
        next_input = self.store.register_input(next_source, "qwen38-fp4")
        next_checkpoint = dict(self.checkpoint, label="fp4", repo_id="pinned/fp4")
        with self.assertRaisesRegex(LabError, "explicit release"):
            cache.cache_checkpoint(self.store, next_checkpoint, next_input["id"], self.root)
        cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        following = cache.cache_checkpoint(self.store, next_checkpoint, next_input["id"], self.root)
        with self.assertRaisesRegex(LabError, "does not own"):
            cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertEqual(self.store.verify_input(following["id"]), following)

    def test_unowned_existing_directory_is_never_reused_or_deleted(self):
        self.root.mkdir()
        unknown = self.root / "keep.txt"
        unknown.write_bytes(b"unrelated")
        with self.assertRaisesRegex(LabError, "ownership state"):
            self.stage()
        with self.assertRaisesRegex(LabError, "ownership state"):
            cache.release_checkpoint_cache(self.store, self.original["id"], self.root)
        self.assertEqual(unknown.read_bytes(), b"unrelated")

    def test_unknown_file_in_owned_root_blocks_both_admission_and_release(self):
        cached = self.stage()
        unknown = self.root / cache.DATA_NAME / "keep.txt"
        unknown.write_bytes(b"unrelated")
        for operation in (self.stage, lambda: cache.release_checkpoint_cache(self.store, cached["id"], self.root)):
            with self.assertRaisesRegex(LabError, "unowned"):
                operation()
        self.assertEqual(unknown.read_bytes(), b"unrelated")
        self.assertTrue((self.root / cache.DATA_NAME / "weights.bin").exists())

    def test_replaced_inode_and_missing_file_block_release_before_any_deletion(self):
        cached = self.stage()
        weights = self.root / cache.DATA_NAME / "weights.bin"
        replacement = self.folder / "replacement"
        replacement.write_bytes(weights.read_bytes())
        replacement.replace(weights)
        with self.assertRaisesRegex(LabError, "replaced"):
            cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertTrue((self.root / cache.DATA_NAME / "empty").exists())
        weights.unlink()
        with self.assertRaisesRegex(LabError, "incomplete"):
            self.stage()
        with self.assertRaisesRegex(LabError, "incomplete"):
            cache.release_checkpoint_cache(self.store, cached["id"], self.root)

    def test_stale_pinned_digest_and_wrong_source_are_rejected(self):
        cached = self.stage()
        original_state = self.marker()
        for changes in ({"inventory_sha256": "0" * 64}, {"source": str(self.folder)}, {"root_identity": [0, 0]}):
            with self.subTest(changes=changes):
                self.change_marker(**changes)
                with self.assertRaises(LabError):
                    self.stage()
                with self.assertRaises(LabError):
                    cache.release_checkpoint_cache(self.store, cached["id"], self.root)
                cache.atomic_json(self.root / cache.STATE_NAME, original_state)

    def test_registration_must_match_total_inventory_not_only_file_hashes(self):
        register = self.store.register_input
        def incorrect(*args, **kwargs):
            approved = register(*args, **kwargs)
            return dict(approved, sha256="0" * 64)
        with patch.object(self.store, "register_input", side_effect=incorrect):
            with self.assertRaisesRegex(LabError, "inventory differs"):
                self.stage()
        self.assertEqual(self.marker()["status"], "copying")

    def test_overlapping_and_relative_cache_roots_are_rejected(self):
        for root in (self.source, self.source / "cache", self.store.root / "cache", self.folder, Path("relative-cache")):
            with self.subTest(root=root), self.assertRaises(LabError):
                cache.cache_checkpoint(self.store, self.checkpoint, self.original["id"], root)
        self.assertEqual((self.source / "weights.bin").read_bytes(), b"approved weights")

    @unittest.skipUnless(os.name == "posix", "POSIX links fixture")
    def test_symlink_cache_root_parent_and_owned_file_are_rejected(self):
        link = self.folder / "linked"
        link.symlink_to(self.source, target_is_directory=True)
        for root in (link, link / "cache"):
            with self.subTest(root=root), self.assertRaises(LabError):
                cache.cache_checkpoint(self.store, self.checkpoint, self.original["id"], root)
        cached = self.stage()
        target = self.root / cache.DATA_NAME / "weights.bin"
        target.unlink()
        target.symlink_to(self.source / "weights.bin")
        with self.assertRaisesRegex(LabError, "link"):
            self.stage()
        with self.assertRaisesRegex(LabError, "link"):
            cache.release_checkpoint_cache(self.store, cached["id"], self.root)
        self.assertEqual((self.source / "weights.bin").read_bytes(), b"approved weights")

    @unittest.skipUnless(os.name == "posix", "POSIX links fixture")
    def test_hdd_nested_symlink_is_not_followed(self):
        original = self.source / "nested"
        moved = self.folder / "moved-original"
        original.rename(moved)
        original.symlink_to(moved, target_is_directory=True)
        with self.assertRaisesRegex(LabError, "without links"):
            self.stage()
        cache.release_checkpoint_cache(self.store, self.original["id"], self.root)
        self.assertEqual((moved / "deeper/config.json").read_bytes(), b'{"model": "pinned"}')


class NativeStorageTests(unittest.TestCase):
    def test_windows_and_unknown_mounts_are_rejected(self):
        with patch.object(cache.sys, "platform", "win32"), self.assertRaisesRegex(LabError, "native Linux"):
            cache._native(Path("/home/workbench/cache"))
        for kind in ("9p", "drvfs", "ntfs", "fuseblk", "tmpfs", "overlay", "unknown"):
            mount = {"filesystems": [{"fstype": kind, "target": "/", "source": "fixture"}]}
            with self.subTest(kind=kind), patch.object(cache.sys, "platform", "linux"), patch.object(cache.subprocess, "check_output", return_value=json.dumps(mount)):
                with self.assertRaisesRegex(LabError, "native Linux filesystem"):
                    cache._native(Path("/home/workbench/cache"))

    def test_native_mount_probe_uses_target_and_checks_mount_relationship(self):
        path = Path(__file__).resolve().parent
        mounted = {"filesystems": [{"fstype": "ext4", "target": path.anchor, "source": "/dev/sdc"}]}
        with patch.object(cache.sys, "platform", "linux"), patch.object(cache.subprocess, "check_output", return_value=json.dumps(mounted)) as probe:
            cache._native(path)
            self.assertEqual(probe.call_args.args[0], ["findmnt", "--target", str(path), "--json", "--output", "FSTYPE,TARGET,SOURCE"])
        mounted["filesystems"][0]["target"] = "/unrelated"
        with patch.object(cache.sys, "platform", "linux"), patch.object(cache.subprocess, "check_output", return_value=json.dumps(mounted)):
            with self.assertRaisesRegex(LabError, "mount identity"):
                cache._native(Path("/home/workbench/cache"))

    def test_unavailable_mount_probe_fails_closed(self):
        with patch.object(cache.sys, "platform", "linux"), patch.object(cache.subprocess, "check_output", side_effect=FileNotFoundError("findmnt")):
            with self.assertRaisesRegex(LabError, "Cannot establish"):
                cache._native(Path("/home/workbench/cache"))
