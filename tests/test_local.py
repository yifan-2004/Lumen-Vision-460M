import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lumen_cli', ROOT / 'scripts/ask_image.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def source_method(relative, class_name, method, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    fn.decorator_list = []
    mod = ast.Module(body=[fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(mod), relative, 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
    return namespace[method]


class ProcessorRegression(unittest.TestCase):
    def call_processor(self, **kwargs):
        seen = {}
        class Tokenizer:
            def apply_chat_template(self, *args, **kw): return 'prompt'
            def __call__(self, *args, **kw):
                seen.update(kw); return {'input_ids': [[1]], 'attention_mask': [[1]]}
        obj = types.SimpleNamespace(tokenizer=Tokenizer(), lm_max_position_embeddings=8192,
                                    _build_image_processor=lambda: None)
        fn = source_method('hf-inference/processing_visionpsynano.py', 'VisionPsyNanoProcessor', '__call__',
            {'_extract_user_text': lambda x: x, '_as_pil_list': lambda x: [],
             'BatchFeature': lambda data, tensor_type: data})
        fn(obj, text='question', **kwargs)
        return seen

    def test_tokenizer_overrides_do_not_raise_duplicate_keywords(self):
        seen = self.call_processor(padding=True, truncation=False, max_length=37)
        self.assertTrue(seen['padding']); self.assertFalse(seen['truncation'])
        self.assertEqual(seen['max_length'], 37)

    def test_default_token_limit_preserved(self):
        self.assertEqual(self.call_processor()['max_length'], 8192)


class LocalTools(unittest.TestCase):
    def test_folder_filters_and_sorts(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['b.PNG', 'a.jpg', 'note.txt']:
                (Path(folder) / name).touch()
            self.assertEqual([p.name for p in cli.collect_images(folder=folder)], ['a.jpg', 'b.PNG'])

    def test_empty_folder_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'No supported images'):
                cli.collect_images(folder=folder)

    def test_html_escapes_model_text_and_names(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'report.html'
            cli.render_report([{'image': '<img onerror=evil>', 'answer': '<script>evil</script>'}], target)
            text = target.read_text()
            self.assertNotIn('<script>', text); self.assertNotIn('<img onerror', text)
            self.assertIn('&lt;script&gt;', text)

    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'photo.jpg'; path.write_bytes(b'keep original')
            with self.assertRaises(SystemExit) as exc:
                cli.main(['--image', str(path), '--output', str(path)])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(path.read_bytes(), b'keep original')

    def test_cli_help_needs_no_model_dependencies(self):
        import subprocess, sys
        result = subprocess.run([sys.executable, '-S', str(ROOT / 'scripts/ask_image.py'), '--help'], capture_output=True)
        self.assertEqual(result.returncode, 0)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch needed for numerical core regression')
class EmbeddingRegression(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.fn = source_method('hf-inference/models/vision_language_model.py', 'VisionLanguageModel',
                                '_replace_img_tokens_with_embd', {})
        self.obj = types.SimpleNamespace(tokenizer=types.SimpleNamespace(image_token_id=9))

    def test_extra_visual_embeddings_are_rejected(self):
        t = self.torch
        with self.assertRaisesRegex(ValueError, 'token count'):
            self.fn(self.obj, t.tensor([[9, 1]]), t.zeros(1, 2, 2), t.ones(1, 2, 2))

    def test_missing_visual_embeddings_are_rejected(self):
        t = self.torch
        with self.assertRaisesRegex(ValueError, 'token count'):
            self.fn(self.obj, t.tensor([[9, 9]]), t.zeros(1, 2, 2), t.ones(1, 1, 2))

    def test_matching_embeddings_preserve_text_positions(self):
        t = self.torch
        result = self.fn(self.obj, t.tensor([[9, 1]]), t.zeros(1, 2, 2), t.tensor([[[3., 4.]]]))
        self.assertEqual(result.tolist(), [[[3., 4.], [0., 0.]]])


if __name__ == '__main__':
    unittest.main()
