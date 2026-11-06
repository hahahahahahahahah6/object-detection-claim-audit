import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from object_detection_claim_audit.audit import audit, evidence, main, matches, OBJECTDETECTION, NOOBJECTDETECTION, GENERIC_NAMES

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(list(OBJECTDETECTION), ['object detection', 'object detections', 'object detector', 'object detectors', 'object detection model', 'object detection models', 'object detection system', 'object detection pipeline', 'object detection feature', 'object detection capability', 'object detection support', 'supports object detection', 'provides object detection', 'performs object detection', 'object detection tool', 'object detection engine', 'object detection api', 'object detection step', 'object detection process', 'automatic object detection', 'auto object detection', 'real-time object detection', 'realtime object detection', 'detects objects', 'detect objects', 'detecting objects', 'detects an object', 'detect an object', 'detecting an object', 'object detection service', 'object detection library', 'object detection framework', 'object detection algorithm', 'object detection algorithms', 'object detection network', 'object detection networks', 'object detection task', 'object detection tasks', 'bounding box detection', 'bounding-box detection', 'object localization', 'object localisation', 'detects bounding boxes', 'detect bounding boxes', 'object detection results', 'object detection output', 'object detection module', 'object detection component', 'object detection function', 'object detection functionality'])
        self.assertEqual(list(NOOBJECTDETECTION), ['does not detect objects', 'do not detect objects', "doesn't detect objects", "don't detect objects", 'no object detection', 'without object detection', 'never detects objects', 'does not perform object detection', 'never performs object detection', 'object detection not supported', 'object detection unsupported', 'no object detection support', 'lacks object detection', 'missing object detection', 'object detection disabled', 'no object detector', 'without object detection support', 'does not provide object detection', 'does not support object detection', 'no automatic object detection', 'object detection not available', 'no object detection capability', 'does not detect an object', 'without an object detector'])

    def test_generic_oracle(self):
        self.assertEqual(list(GENERIC_NAMES), ['detect_objects', 'detect_object', 'find_objects', 'find_object', 'detect_boxes', 'locate_objects', 'locate_object', 'get_objects', 'detect_bounding_boxes', 'find_bounding_boxes', 'object_detection', 'object_detect', 'run_object_detection', 'perform_object_detection', 'predict_objects', 'predict_object', 'extract_objects', 'detect_all_objects', 'find_all_objects', 'get_detected_objects', 'get_bounding_boxes', 'detect_object_boxes', 'localize_objects', 'detect_regions'])

    def fixture(self, readme='', code='', extra=None, flags=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'README.md').write_text(readme)
            (root/'app.py').write_text(code)
            for name, text in (extra or {}).items():
                p = root/name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                status = main(['audit', str(root)] + list(flags))
            return status, out.getvalue()

    def test_pairs(self):
        for code in ['import ultralytics as ul; m=ul.YOLO(); m.predict(image)', 'from ultralytics import YOLO as Y; Y().predict(image)', 'from ultralytics import YOLO; m=YOLO(); m(image)', 'import ultralytics; ultralytics.YOLO()(image)', 'from torchvision.models.detection import fasterrcnn_resnet50_fpn as F; m=F(); m.forward(image)', 'import torchvision.models.detection as d; d.fcos_resnet50_fpn().forward(image)', 'from torchvision.models.detection import ssd300_vgg16; ssd300_vgg16()(image)', 'from detectron2.engine import DefaultPredictor as D; p=D(); p(image)', 'import detectron2.engine as e; e.DefaultPredictor().predict(image)', 'from mmdet.apis import init_detector as init, inference_detector as run; m=init(); run(m, image)', 'import mmdet.apis as a; a.inference_detector(a.init_detector(), image)', 'from mmdet.apis import init_detector; init_detector().inference_detector(image)', 'f=x.detect_fn; g=f; g(image)', 'x.detect_fn(image)', 'from transformers import pipeline as p; f=p("OBJECT-DETECTION"); f(image)', 'import transformers as t; t.pipeline("object-detection")(image)', 'from transformers import Owlv2ForObjectDetection as O; m=O(); m.predict(image)', 'import transformers as t; t.GroundingDinoForObjectDetection()(image)']:
            with self.subTest(code=code): self.assertTrue(evidence(code))

    def test_all_factory_variants(self):
        for factory in ('fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
                        'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2',
                        'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
                        'ssd300_vgg16', 'fcos_resnet50_fpn'):
            for run in ('m.forward(image)', 'm(image)'):
                code = 'from torchvision.models.detection import ' + factory + ' as F; m=F(); '
                with self.subTest(factory=factory, run=run):
                    self.assertTrue(evidence(code + run))
                    self.assertFalse(evidence(code))
                    self.assertFalse(evidence(code + 'other.forward(image)'))

    def test_other_pair_shadowing(self):
        cases = [
            'from torchvision.models.detection import fcos_resnet50_fpn as F\ndef F(): pass\nF()(image)',
            'def detect_fn(): pass\nx.detect_fn(image)',
            'from mmdet.apis import init_detector, inference_detector\ndef inference_detector(x): pass\ninference_detector(init_detector())',
            'from ultralytics import YOLO\ndef ultralytics(): pass\nimport ultralytics\nultralytics.YOLO()(image)',
        ]
        for code in cases:
            with self.subTest(code=code): self.assertFalse(evidence(code))

    def test_inventory_matching(self):
        for phrase in OBJECTDETECTION:
            with self.subTest(phrase=phrase):
                self.assertEqual(matches(phrase, OBJECTDETECTION)[0], [phrase])
        for phrase in NOOBJECTDETECTION:
            with self.subTest(phrase=phrase):
                denials, rest = matches(phrase, NOOBJECTDETECTION)
                self.assertEqual(denials, [phrase])
                self.assertEqual(matches(rest, OBJECTDETECTION)[0], [])
        self.assertEqual(matches('object detections', OBJECTDETECTION)[0], ['object detections'])

    def test_generics(self):
        for name in GENERIC_NAMES:
            for code in ('x.'+name+'()', 'from unknown import '+name+' as run; run()'):
                with self.subTest(code=code): self.assertTrue(evidence(code))

    def test_rejections(self):
        for code in ['import ultralytics; from transformers import pipeline', 'from torchvision.models import resnet50; m=resnet50(); m.forward(image)', 'from torchvision.models import efficientnet_b0; efficientnet_b0()(image)', 'import timm; timm.create_model("x")(image)', 'from keras.applications import ResNet50; ResNet50()(image)', 'from transformers import pipeline; pipeline("image-classification")(image)', 'from torchvision.models.segmentation import fcn_resnet50; fcn_resnet50()(image)', 'from facenet import FaceDetector; FaceDetector().predict(image)', 'ocr.readtext(image)', 'pose.estimate(image)', 'keypoint.predict(image)', 'from ultralytics import YOLO; YOLO().track(image)', 'Tracker().update(image)', 'generator.generate(image)', 'from ultralytics import YOLO; m=YOLO()', 'from torchvision.models.detection import fcos_resnet50_fpn; m=fcos_resnet50_fpn()', 'from detectron2.engine import DefaultPredictor; m=DefaultPredictor()', 'from mmdet.apis import init_detector; m=init_detector()', 'f=x.detect_fn', 'from transformers import pipeline; f=pipeline("object-detection")', 'from transformers import Owlv2ForObjectDetection; m=Owlv2ForObjectDetection()', '"object detection detect objects" # object detection', '"""object detection"""', 'from x import detect_objects; detect_objects', 'detect_objects()', 'import subprocess; subprocess.run(["object-detection", "image"])', 'from ultralytics import YOLO; m=YOLO(); other.predict(image)', 'this is invalid python!', 'from mmdet.apis import init_detector, inference_detector; m=init_detector(); inference_detector(other, image)']:
            with self.subTest(code=code): self.assertFalse(evidence(code))

    def test_shadowing(self):
        for code in ['from x import detect_objects\ndef detect_objects(): pass\ndetect_objects()', 'from ultralytics import YOLO\nclass YOLO: pass\nYOLO().predict(image)', 'from transformers import pipeline\ndef pipeline(x): pass\npipeline("object-detection")(image)', 'from detectron2.engine import DefaultPredictor\nclass DefaultPredictor: pass\nDefaultPredictor()(image)', 'from mmdet.apis import init_detector, inference_detector\ndef init_detector(): pass\ninference_detector(init_detector(),image)', 'from transformers import Owlv2ForObjectDetection\nclass Owlv2ForObjectDetection: pass\nOwlv2ForObjectDetection()(image)']:
            with self.subTest(code=code): self.assertFalse(evidence(code))

    def test_split_modules(self):
        self.assertEqual(self.fixture("object detection", "from ultralytics import YOLO; m=YOLO()", {"other.py": "m.predict(image)"})[0], 2)

    def test_matching(self):
        self.assertEqual(matches('object detection', OBJECTDETECTION)[0], ['object detection'])
        self.assertEqual(self.fixture('does not detect objects')[0], 0)
        self.assertEqual(self.fixture('OBJECT DETECTION')[0], 2)
        self.assertEqual(self.fixture('```text\nobject detection\n```')[0], 0)
        self.assertEqual(self.fixture('`object detection`')[0], 2)
        self.assertEqual(matches('object-detection', OBJECTDETECTION)[0], [])

    def test_cli_good(self):
        self.assertEqual(self.fixture('object detection', 'x.detect_objects()'), (0, 'PASS\n'))

    def test_cli_bad(self):
        status, out = self.fixture('object detection')
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-object-detection-claim-false', out)

    def test_cli_test_only(self):
        extra = {'tests/sample.md': 'object detection'}
        status, out = self.fixture(extra=extra)
        self.assertEqual(status, 0)
        self.assertIn('MEDIUM', out)
        self.assertEqual(self.fixture(extra=extra, flags=['--fail-on', 'medium'])[0], 2)

    def test_cli_denial(self):
        status, out = self.fixture('no object detection', 'x.detect_objects()')
        self.assertEqual(status, 2)
        self.assertIn('HIGH documented-no-object-detection-claim-false', out)

    def test_cli_no_claim(self):
        self.assertEqual(self.fixture('A useful tool.'), (0, 'PASS\n'))

    def test_cli_json(self):
        status, out = self.fixture('object detection', flags=['--json'])
        f = json.loads(out)[0]
        self.assertEqual(status, 2)
        self.assertEqual(f['verdict'], 'FAIL')
        self.assertTrue(f['message'])

    def test_cli_ignore(self):
        self.assertEqual(self.fixture(extra={'docs/claim.md': 'object detection'}, flags=['--ignore', 'docs/'])[0], 0)

    def test_cli_version(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as cm:
            main(['--version'])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), '0.1.0')

    def test_readme(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(self.fixture((root/'README.md').read_text()), (0, 'PASS\n'))

    def test_io_and_usage(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['audit', '/nonexistent-r142-directory']), 1)
            with self.assertRaises(SystemExit) as cm: main(['audit', '--unknown'])
        self.assertEqual(cm.exception.code, 1)

    def test_virtualenv_excluded(self):
        status, out = self.fixture('object detection', extra={
            'custom-env/pyvenv.cfg': 'home = /usr/bin',
            'custom-env/lib/model.py': 'x.detect_objects()',
        })
        self.assertEqual(status, 2)

    def test_verbose(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.fixture(flags=['--verbose'])[0], 0)
        self.assertIn('scope:', err.getvalue())

if __name__ == '__main__':
    unittest.main()
