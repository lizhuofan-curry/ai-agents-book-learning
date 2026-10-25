import importlib.util
from pathlib import Path
import tempfile
import unittest

P = Path(__file__).resolve().parent/'我的代码/01_生成并阅读测试图.py'
spec = importlib.util.spec_from_file_location('chart_step', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class ChartChecks(unittest.TestCase):
    def test_source_and_fixed_reference(self):
        module, manifest = m.prepare()
        self.assertEqual(module.REVENUE, [120, 150, 95, 180])
        self.assertEqual(len(manifest['files']), 3)

    def test_rendered_png_and_reference_separation(self):
        with tempfile.TemporaryDirectory() as temp:
            chart, details, reference, _ = m.generate(temp)
            self.assertTrue(chart.is_file())
            self.assertEqual((details['width'], details['height']), (900, 600))
            self.assertEqual(reference['expected_quarter'], 'Q4')
            self.assertEqual(reference['expected_value'], 180)
            self.assertEqual(list(Path(temp).glob('*.json')), [])

    def test_no_image_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/'sample_chart.png'
            p.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                m.generate(temp)
            self.assertEqual(p.read_bytes(), b'preserve')


if __name__ == '__main__':
    unittest.main()
