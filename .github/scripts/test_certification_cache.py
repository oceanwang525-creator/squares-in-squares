import json
import unittest
from unittest.mock import patch
import certification_cache as c

class CacheTests(unittest.TestCase):
    def test_tampering_rejected(self):
        d=c.envelope(dict(context={"v":1},entries={"a"*64:[1,0.0,"1","ok"]}))
        d["payload"]["entries"]["a"*64][2]="0.9"
        with self.assertRaises(ValueError):c.decode(json.dumps(d),{"v":1})
    def test_context_revision_invalidates(self):
        d=c.envelope(dict(context={"v":1},entries={"a"*64:[1,0.0,"1","ok"]}))
        self.assertEqual(c.decode(json.dumps(d),{"v":2}),{})
    def test_valid_evidence_reused_exactly(self):
        d=c.envelope(dict(context={"v":1},entries={"a"*64:[1,0.0,"1","ok"]}))
        self.assertEqual(c.decode(json.dumps(d),{"v":1})["a"*64],(1,0.0,"1","ok"))
    def test_bad_entry_rejected_even_with_consistent_envelope(self):
        d=c.envelope(dict(context={},entries={"a"*64:[1,0.0,None,"invalid"]}))
        with self.assertRaises(ValueError):c.decode(json.dumps(d),{})
    def test_fork_event_and_failed_runs_untrusted(self):
        r=dict(head_branch="main",event="push",conclusion="success",head_repository={"id":1},
            repository={"id":1},path=".github/workflows/publish-certified-merge.yml")
        self.assertTrue(c.trusted_run(r,"x/y",1))
        for update in [dict(event="pull_request"),dict(head_repository={"id":2}),
                       dict(conclusion="failure"),dict(head_branch="branch")]:
            self.assertFalse(c.trusted_run(dict(r,**update),"x/y",1))
    def test_archive_digest_is_not_just_manifest_checksum(self):
        self.assertNotEqual(c.digest(b"good"),c.digest(b"tampered"))
if __name__=="__main__":unittest.main()
