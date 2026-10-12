from pathlib import Path
import unittest

from autonomous_research_lab.models import ExperimentSpec, LabError, Limits


class ManifestTests(unittest.TestCase):
    def manifest(self, **fields):
        return ExperimentSpec.from_dict({"name": "study", "source": "code", "entrypoint": "main.py", **fields}, Path.cwd())

    def test_paths_cannot_escape_the_bundle_or_output_directory(self):
        for path in ("../main.py", "/main.py", "a/../../main.py", "a\\main.py", "C:/main.py", "a/./main.py"):
            with self.subTest(path=path), self.assertRaises(LabError):
                self.manifest(entrypoint=path)
            with self.subTest(output=path), self.assertRaises(LabError):
                self.manifest(outputs=[path])

    def test_limits_reject_nonpositive_boolean_and_nonfinite_allocations(self):
        for limits in ({"wall_seconds": 0}, {"pids": -1}, {"memory_bytes": True}, {"memory_bytes": 1024}, {"output_bytes": 0}, {"cpus": float("inf")}, {"cpus": True}, {"cpus": float("nan")}):
            with self.subTest(limits=limits), self.assertRaises(LabError):
                Limits.from_dict(limits)

    def test_misspelled_resource_fields_are_not_silently_ignored(self):
        with self.assertRaises(LabError):
            self.manifest(limits={"wall_secs": 5})
        with self.assertRaises(LabError):
            self.manifest(command="python main.py")

    def test_nonfinite_settings_cannot_enter_the_research_record(self):
        with self.assertRaises(LabError):
            self.manifest(settings={"threshold": float("nan")})

    def test_gpu_permission_requires_a_boolean(self):
        with self.assertRaises(LabError):
            self.manifest(gpu="false")

    def test_input_bindings_reject_host_paths_and_mount_path_injection(self):
        input_id = "input_" + "a" * 32
        for bindings in (
            {"/host": input_id}, {"../secret": input_id}, {"nested/model": input_id},
            {"model": "/mnt/d/weights"}, {"model": {"path": "/secrets"}}, [],
            {"x" * 65: input_id}, {f"input{i}": input_id for i in range(17)},
        ):
            with self.subTest(bindings=bindings), self.assertRaises(LabError):
                self.manifest(inputs=bindings)

    def test_unsupported_versions_fail_instead_of_being_reinterpreted(self):
        with self.assertRaises(LabError):
            self.manifest(schema_version=2)


if __name__ == "__main__":
    unittest.main()
