import importlib.util
from pathlib import Path
import sys
import tempfile
import json
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import train_projector as trainer

class DatasetTests(unittest.TestCase):
    def test_duplicate_image_bytes_across_different_paths_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for name in ['a','b']:
                (p/f'{name}.jpg').write_bytes(b'identical image bytes')
                (p/f'{name}.jsonl').write_text(json.dumps({'id':name,'image':f'{name}.jpg','question':'Q','answer':'A'}))
            with self.assertRaisesRegex(ValueError,'duplicate image content'):
                trainer.check_split(trainer.read_data(p/'a.jsonl'),trainer.read_data(p/'b.jsonl'))

    def test_duplicate_ids_rejected(self):
        rows=[{'id':'x','image':'a','_image_sha256':'a'}]
        with self.assertRaisesRegex(ValueError,'id overlap'):
            trainer.check_split(rows,rows)

    def test_blank_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'empty.jsonl';p.write_text('')
            with self.assertRaisesRegex(ValueError,'empty'):
                trainer.read_data(p)

@unittest.skipUnless(importlib.util.find_spec('torch'),'torch needed for loss-label regression')
class LabelTests(unittest.TestCase):
    def test_answer_only_labels_are_shifted_once(self):
        import torch
        x,y=trainer.answer_labels(torch,torch.tensor([[1,2,3]]),torch.tensor([[4,5]]))
        self.assertEqual(x.tolist(),[[1,2,3,4]])
        self.assertEqual(y.tolist(),[[-100,-100,4,5]])

if __name__=='__main__':unittest.main()
