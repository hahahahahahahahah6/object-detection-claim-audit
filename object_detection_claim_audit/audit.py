"""Deterministic documentation and Python source inspection."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import OBJECTDETECTION, NOOBJECTDETECTION, GENERIC_NAMES

SKIP = {'.git', '__pycache__', '.venv', 'venv', 'env', '.env', 'dist', 'build'}
POSITIVE = 'documented-object-detection-claim-false'
NEGATIVE = 'documented-no-object-detection-claim-false'


def matches(text, phrases):
    text = text.lower()
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        start = text.find(phrase)
        while start >= 0:
            found.append(phrase)
            text = text[:start] + ' ' * len(phrase) + text[start + len(phrase):]
            start = text.find(phrase)
    return found, text


def prose(text):
    return re.sub(r'^\s*```[^\n]*\n.*?^\s*```[^\n]*(?:\n|$)', '', text,
                  flags=re.M | re.S).replace('`', '')


def evidence(source):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    nodes = list(ast.walk(tree))
    shadow = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    imports = {}
    for n in nodes:
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split('.')[0]] = a.name if a.asname else a.name.split('.')[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imports[a.asname or a.name] = (n.module or '') + '.' + a.name

    def resolve(n):
        if isinstance(n, ast.Name):
            return '' if n.id in shadow else imports.get(n.id, '')
        if isinstance(n, ast.Attribute):
            base = resolve(n.value)
            return base + '.' + n.attr if base and n.attr not in shadow else ''
        return ''

    factories = {
        'ultralytics.YOLO': 'predict',
        'detectron2.engine.DefaultPredictor': 'predict',
        'mmdet.apis.init_detector': 'inference_detector',
    }
    for factory in ('fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
                    'maskrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn_v2',
                    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
                    'ssd300_vgg16', 'fcos_resnet50_fpn'):
        factories['torchvision.models.detection.' + factory] = 'forward'

    instances = {}

    def value_kind(n):
        if isinstance(n, ast.Name):
            return '' if n.id in shadow else instances.get(n.id, '')
        if isinstance(n, ast.Attribute) and n.attr == 'detect_fn':
            if n.attr not in shadow and not (isinstance(n.value, ast.Name) and n.value.id in shadow):
                return 'direct'
        if not isinstance(n, ast.Call):
            return ''
        name = resolve(n.func)
        if name in factories:
            return factories[name]
        if name.endswith('ForObjectDetection'):
            return 'predict'
        if (name == 'transformers.pipeline' and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)
                and n.args[0].value.lower() == 'object-detection'):
            return 'direct'
        return ''

    # Fixed-point propagation supports simple assigned aliases in one module.
    for _ in range(len(nodes) + 1):
        before = dict(instances)
        for n in nodes:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                kind = value_kind(n.value)
                if kind:
                    targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                    for target in targets:
                        if isinstance(target, ast.Name):
                            instances[target.id] = kind
        if before == instances:
            break
    for n in nodes:
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = resolve(f)
        if isinstance(f, ast.Attribute):
            if f.attr in GENERIC_NAMES and f.attr not in shadow and not (isinstance(f.value, ast.Name) and f.value.id in shadow):
                return True
            kind = value_kind(f.value)
            if kind and f.attr == kind and f.attr not in shadow:
                return True
        elif isinstance(f, ast.Name) and name.rsplit('.', 1)[-1] in GENERIC_NAMES:
            return True
        kind = value_kind(f)
        if kind and kind != 'inference_detector':
            return True
        if (name == 'mmdet.apis.inference_detector' and n.args
                and value_kind(n.args[0]) == 'inference_detector'):
            return True
    return False


def audit(repo, ignore=(), verbose=False):
    root = Path(repo)
    if not root.is_dir():
        raise OSError('Repository is not a directory: ' + str(root))
    paths = []
    for p in sorted(root.rglob('*')):
        rel = p.relative_to(root).as_posix()
        if any(part in SKIP for part in p.relative_to(root).parts) or any(rel.startswith(prefix) for prefix in ignore):
            continue
        if any((parent / 'pyvenv.cfg').is_file() for parent in p.parents if parent != root and root in parent.parents):
            continue
        if p.is_file():
            paths.append((p, rel))
    has_evidence = False
    docs = []
    for p, rel in paths:
        if p.suffix == '.py':
            found = evidence(p.read_text(encoding='utf-8'))
            has_evidence |= found
            if verbose:
                print('source ' + rel + ': ' + ('evidence' if found else 'no evidence'), file=sys.stderr)
        parts = Path(rel).parts
        if rel == 'README.md' or (parts[0] == 'docs' and p.suffix == '.md') or 'tests' in parts:
            text = prose(p.read_text(encoding='utf-8'))
            denials, remaining = matches(text, NOOBJECTDETECTION)
            claims, _ = matches(remaining, OBJECTDETECTION)
            docs.append((rel, claims, denials, 'tests' in parts))
            if verbose:
                print('documentation ' + rel + ': ' + str(len(claims)) + ' claims, ' + str(len(denials)) + ' denials', file=sys.stderr)
    findings = []
    for rel, claims, denials, test in docs:
        for phrases, contradicted, identifier in ((claims, not has_evidence, POSITIVE), (denials, has_evidence, NEGATIVE)):
            if contradicted and phrases:
                severity = 'MEDIUM' if test else 'HIGH'
                findings.append(dict(id=identifier, severity=severity, path=rel, verdict='FAIL',
                                     message='Documentation and detected implementation disagree.', phrases=phrases))
    return findings


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, 'error: ' + message + '\n')


def main(argv=None):
    parser = Parser(description='Audit documentation claims against Python source.')
    parser.add_argument('--version', action='version', version=__version__)
    sub = parser.add_subparsers(dest='command', required=True, parser_class=Parser)
    cmd = sub.add_parser('audit')
    cmd.add_argument('repo', nargs='?', default='.')
    cmd.add_argument('--json', action='store_true')
    cmd.add_argument('--verbose', action='store_true')
    cmd.add_argument('--ignore', action='append', default=[])
    cmd.add_argument('--fail-on', choices=['high', 'medium'], default='high')
    args = parser.parse_args(argv)
    try:
        if args.verbose:
            print('scope: ' + str(Path(args.repo).resolve()), file=sys.stderr)
        findings = audit(args.repo, args.ignore, args.verbose)
    except (OSError, UnicodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(findings))
    elif findings:
        for f in findings:
            print('{severity} {id} {path}: {message}'.format(**f))
    else:
        print('PASS')
    return 2 if any(f['severity'] == 'HIGH' or args.fail_on == 'medium' for f in findings) else 0
