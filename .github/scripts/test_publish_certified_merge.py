import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("publisher", Path(__file__).with_name("publish_certified_merge.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def report(side="2", score=100):
    return dict(accepted=1000, missing=0, rejected=0, score=score,
                results=[dict(n=n, L=side, reason="ok") for n in range(1, 1001)])


class GateTests(unittest.TestCase):
    def test_noop(self):
        a = report()
        self.assertEqual(p.validate_transition([a, a], [a, a], False), [])

    def test_strict_dual_gain(self):
        old, new = report(), report(score=99)
        new["results"][6]["L"] = "1.99999999999999999999999999999999999999"
        self.assertEqual(p.validate_transition([old, old], [new, new], True), [7])

    def test_changed_tie_rejected(self):
        a = report()
        with self.assertRaises(ValueError):
            p.validate_transition([a, a], [a, a], True)

    def test_l_gain_without_score_gain_rejected(self):
        old, new = report(), report()
        new["results"][0]["L"] = "1.99"
        with self.assertRaises(ValueError):
            p.validate_transition([old, old], [new, new], True)

    def test_one_mode_regression_rejected(self):
        old, new = report(), report(score=99)
        new["results"][0]["L"] = "2.00000000000000000000000000000001"
        with self.assertRaises(ValueError):
            p.validate_transition([old, old], [old, new], True)

    def test_different_wins_between_modes_rejected(self):
        old, a, b = report(), report(score=99), report(score=99)
        a["results"][0]["L"] = "1.9"
        b["results"][1]["L"] = "1.9"
        with self.assertRaises(ValueError):
            p.validate_transition([old, old], [a, b], True)

    def test_missing_or_duplicate_groups_rejected(self):
        for edit in ("missing", "duplicate", "reject", "nan"):
            a = report()
            if edit == "missing":
                a["results"].pop()
            elif edit == "duplicate":
                a["results"][0]["n"] = 2
            elif edit == "reject":
                a["rejected"] = 1
            else:
                a["results"][0]["L"] = "NaN"
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                p.validate_report(a)

    def test_noop_certification_mismatch_rejected(self):
        a, b = report(), report(score=99)
        with self.assertRaises(ValueError):
            p.validate_transition([a, a], [b, b], False)

    def test_both_modes_required(self):
        with self.assertRaises(ValueError):
            p.validate_transition([report()], [report()], False)


class LoopGuardTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after = "a" * 40, "b" * 40
        self.event = dict(before=self.before, after=self.after, size=1, commits=[dict(id=self.after)])

    def read(self, *args):
        return {
            ("rev-list", "--count", self.before + ".." + self.after): "1",
            ("show", "-s", "--format=%P", self.after): self.before,
            ("show", "-s", "--format=%B", self.after): p.COMMIT_MESSAGE,
            ("show", "-s", "--format=%an%n%ae%n%cn%n%ce", self.after): "\n".join([p.NAME, p.EMAIL, p.NAME, p.EMAIL]),
            ("diff", "--name-only", self.before, self.after, "--"): p.CANONICAL,
        }[args]

    def test_exact_self_commit_skips(self):
        self.assertTrue(p.own_commit_push(self.event, "push", self.read))

    def test_manual_never_skips(self):
        self.assertFalse(p.own_commit_push(self.event, "workflow_dispatch", self.read))

    def test_mixed_push_never_skips(self):
        self.event["size"] = 2
        self.assertFalse(p.own_commit_push(self.event, "push", self.read))

    def test_extra_changed_file_never_skips(self):
        def read(*args):
            return p.CANONICAL + "\npoc/new/final.csv" if args[0] == "diff" else self.read(*args)
        self.assertFalse(p.own_commit_push(self.event, "push", read))

    def test_wrong_message_never_skips(self):
        def read(*args):
            return "ordinary change" if "--format=%B" in args else self.read(*args)
        self.assertFalse(p.own_commit_push(self.event, "push", read))

    def test_other_identity_never_skips(self):
        def read(*args):
            return "other" if "--format=%an%n%ae%n%cn%n%ce" in args else self.read(*args)
        self.assertFalse(p.own_commit_push(self.event, "push", read))

    def test_merged_or_missing_history_never_skips(self):
        def read(*args):
            return self.before + " " + "c" * 40 if "--format=%P" in args else self.read(*args)
        self.assertFalse(p.own_commit_push(self.event, "push", read))


class MirrorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.remote, self.root = base / "origin.git", base / "clone"
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        subprocess.run(["git", "clone", str(self.remote), str(self.root)], check=True, capture_output=True)
        self.patch = patch.object(p, "ROOT", self.root)
        self.patch.start()
        # Function default cwd is bound on definition; route it to the test clone.
        self.original_git = p.git
        self.git_patch = patch.object(p, "git", side_effect=self.local_git)
        self.git_patch.start()
        self.value_patch = patch.object(p, "value", side_effect=self.local_value)
        self.value_patch.start()
        self.local_git("checkout", "-b", "main")
        (self.root / "README").write_text("fixture only")
        self.local_git("add", ".")
        self.local_git(*p.identity_args(), "commit", "-m", "fixture")
        self.local_git("push", "-u", "origin", "main")
        self.sha = self.local_value("rev-parse", "HEAD")

    def local_git(self, *args, cwd=None, check=True):
        return self.original_git(*args, cwd=cwd or self.root, check=check)

    def local_value(self, *args, cwd=None):
        return self.local_git(*args, cwd=cwd).stdout.strip()

    def tearDown(self):
        self.value_patch.stop()
        self.git_patch.stop()
        self.patch.stop()
        self.temp.cleanup()

    def test_create_update_and_noop_exact_bytes(self):
        data = b"fixture packing bytes\r\n"
        self.assertTrue(p.publish_snapshot(data, 22.1, self.sha))
        first = self.local_value("ls-remote", "origin", "refs/heads/merge").split()[0]
        self.assertEqual(self.local_value("ls-tree", "-r", "--name-only", first), "merged-22.10000000000.csv")
        actual = subprocess.check_output(["git", "show", first + ":merged-22.10000000000.csv"], cwd=self.root)
        self.assertEqual(actual, data)
        self.assertTrue(p.publish_snapshot(data, 22.1, self.sha))
        self.assertEqual(self.local_value("ls-remote", "origin", "refs/heads/merge").split()[0], first)
        self.assertTrue(p.publish_snapshot(b"new fixture", 22.0, self.sha))
        second = self.local_value("ls-remote", "origin", "refs/heads/merge").split()[0]
        self.assertEqual(self.local_value("rev-parse", second + "^"), first)
        self.assertEqual(self.local_value("ls-tree", "-r", "--name-only", second), "merged-22.00000000000.csv")
        self.assertEqual(self.local_value("show", "-s", "--format=%an%n%ae%n%cn%n%ce", second),
                         "\n".join([p.NAME, p.EMAIL, p.NAME, p.EMAIL]))

    def test_stale_source_does_not_publish(self):
        self.assertFalse(p.publish_snapshot(b"fixture", 1.0, "a" * 40))
        self.assertEqual(self.local_git("ls-remote", "--heads", "origin", "merge").stdout, "")

    def test_source_advances_before_push_does_not_publish(self):
        with patch.object(p, "fetch_main", side_effect=[self.sha, "b" * 40]):
            self.assertFalse(p.publish_snapshot(b"fixture", 1.0, self.sha))
        self.assertEqual(self.local_git("ls-remote", "--heads", "origin", "merge").stdout, "")



class PublisherIntegrationTests(MirrorIntegrationTests):
    def prepare_union(self, improve=False, bad=False):
        canonical = self.root / p.CANONICAL
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"initial fixture bytes")
        helper = self.root / "poc/auto_union/poc/union.py"
        helper.parent.mkdir(parents=True, exist_ok=True)
        fixture = {"before": report(), "after": report(score=99) if improve else report()}
        if improve:
            fixture["after"]["results"][0]["L"] = "1.9"
        if bad:
            fixture["after"]["results"][1]["L"] = "2.1"
        code = (
            "import pathlib,json,hashlib\n"
            "class pd: __version__='3.1.0'\n"
            "fixture=json.loads(" + repr(json.dumps(fixture)) + ")\n"
            "def full_gate(path): return [fixture['before'],fixture['before']]\n"
            "def union(path,workers,audit):\n"
            " initial=path.read_bytes()\n"
            + (" path.write_bytes(b'improved fixture bytes')\n" if improve else "")
            + " proof=dict(before_sha256=hashlib.sha256(initial).hexdigest(),after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),improved="
            + ("[1]" if improve else "[]")
            + ",reports=[fixture['after'],fixture['after']])\n"
            " (audit/'union.json').write_text(json.dumps(proof))\n"
        )
        helper.write_text(code)
        self.local_git("add", ".")
        self.local_git(*p.identity_args(), "commit", "-m", "local unit fixtures")
        self.local_git("push", "origin", "main")
        self.sha = self.local_value("rev-parse", "HEAD")
        event = self.root.parent / "event.json"
        event.write_text("{}")
        self.env = dict(GITHUB_REF="refs/heads/main", GITHUB_EVENT_NAME="workflow_dispatch", GITHUB_EVENT_PATH=str(event))

    def test_main_noop_does_not_commit(self):
        self.prepare_union()
        with patch.dict(p.os.environ, self.env):
            p.main()
        self.assertEqual(self.local_value("ls-remote", "origin", "refs/heads/main").split()[0], self.sha)
        self.assertEqual(self.local_value("rev-parse", "HEAD"), self.sha)
        self.assertEqual((self.root / p.CANONICAL).read_bytes(), b"initial fixture bytes")

    def test_main_improvement_commits_only_canonical_and_publishes_union_sha(self):
        self.prepare_union(improve=True)
        with patch.dict(p.os.environ, self.env):
            p.main()
        sha = self.local_value("rev-parse", "HEAD")
        self.assertNotEqual(sha, self.sha)
        self.assertEqual(self.local_value("show", "--format=", "--name-only", sha), p.CANONICAL)
        self.assertEqual(self.local_value("show", "-s", "--format=%an%n%ae%n%cn%n%ce", sha),
                         "\n".join([p.NAME, p.EMAIL, p.NAME, p.EMAIL]))
        mirror = self.local_value("ls-remote", "origin", "refs/heads/merge").split()[0]
        self.assertIn(sha, self.local_value("show", "-s", "--format=%B", mirror))
        self.assertEqual(subprocess.check_output(["git", "show", mirror + ":merged-99.00000000000.csv"], cwd=self.root), b"improved fixture bytes")

    def test_main_bad_gate_does_not_push_either_branch(self):
        self.prepare_union(improve=True, bad=True)
        with patch.dict(p.os.environ, self.env), self.assertRaises(ValueError):
            p.main()
        self.assertEqual(self.local_value("ls-remote", "origin", "refs/heads/main").split()[0], self.sha)
        self.assertEqual(self.local_git("ls-remote", "--heads", "origin", "merge").stdout, "")

    def test_main_concurrent_change_prevents_both_pushes(self):
        self.prepare_union(improve=True)
        with patch.dict(p.os.environ, self.env), patch.object(p, "fetch_main", side_effect=[self.sha, "b" * 40]):
            p.main()
        self.assertEqual(self.local_value("ls-remote", "origin", "refs/heads/main").split()[0], self.sha)
        self.assertEqual(self.local_git("ls-remote", "--heads", "origin", "merge").stdout, "")

    def test_atomic_push_rejects_main_advance_in_final_race(self):
        data = b"fixture bytes"
        original_git = self.local_git
        def race(*args, **kwargs):
            if args[:2] == ("push", "--atomic"):
                # Move remote main after the publisher's last fetch, without touching it locally.
                tree = self.local_value("rev-parse", self.sha + "^{tree}")
                result = subprocess.run(
                    ["git", *p.identity_args(), "commit-tree", tree, "-p", self.sha, "-m", "concurrent fixture"],
                    cwd=self.root, check=True, text=True, capture_output=True)
                subprocess.run(["git", "--git-dir", str(self.remote), "update-ref", "refs/heads/main", result.stdout.strip()],
                               check=True, capture_output=True)
            return original_git(*args, **kwargs)
        with patch.object(p, "git", side_effect=race), self.assertRaises(subprocess.CalledProcessError):
            p.publish_snapshot(data, 1.0, self.sha)
        self.assertEqual(self.local_git("ls-remote", "--heads", "origin", "merge").stdout, "")


if __name__ == "__main__":
    unittest.main()
