# Documentation claim auditor

An offline, deterministic, standard-library-only auditor for claims about box
finding. Requires Python 3.9 or later. No network, telemetry, model execution,
or file writes are performed by the auditor.

## Usage

```sh
python3 -m pip install .
object-detection-claim-audit audit .
python3 -m object_detection_claim_audit audit . --json
object-detection-claim-audit audit . --verbose --ignore vendor/ --ignore examples/
object-detection-claim-audit audit . --fail-on medium
object-detection-claim-audit --version
```

The repository defaults to the current directory. Exit status is 0 when no
finding reaches the threshold, 2 when one does, and 1 for usage or IO errors.
The default threshold is high. JSON output is a list of findings, each with
id, severity, path, verdict, message, and matched phrases. Verbose scan scope
and per-file decisions go to standard error. Ignore prefixes are repeatable
and match repository-relative paths.

## Scope and rules

Scans the root README and Markdown below docs. Claims in fixtures under a
tests directory receive medium severity. Python evidence is searched throughout
the repository, excluding version control, bytecode caches, virtual environments,
and build output. Each route must be complete within one module. Imported
aliases, assigned aliases and chained calls are recognized. Local function and
class definitions shadow imported route names. Unparseable modules supply no evidence.

Fenced code is excluded; inline code counts as prose. Matching uses exact,
case-insensitive substrings without stemming or hyphen normalization. Denial
spans are removed first. Longest-first matching blanks matched spans so each
occurrence matches once under its longest phrase. A claim without evidence is
high severity; a denial contradicted by evidence is high severity. A denial
without evidence passes. The finding identifiers are:

```text
documented-object-detection-claim-false
documented-no-object-detection-claim-false
```

## Differentiation

The following neighbors check whole-picture labelling, per-pixel labelling,
following boxes across frames, and faces only, respectively:

```text
image-classification-claim-audit
semantic-segmentation-claim-audit
tracking-claim-audit
face-detection-claim-audit
```

This auditor checks box finding for general objects in one image. Those other
capabilities alone do not supply evidence for the checked feature.

## Evidence and limitations

Static patterns are not semantic judgement. Exact phrases miss restatements.
Generic methods on unrelated objects count because the name is shared:

```python
unrelated.detect_objects()
```

The task literal and class-name suffix below are trusted on name alone,
without downloading or inspecting weights:

```text
transformers.pipeline("object-detection")
*ForObjectDetection
```

Both require construction plus a call on the returned value; the class route
also accepts a prediction method. The following pairs count on construction
(or attribute access for the last route) plus a run call in one module:

```text
ultralytics.YOLO: predict or direct call
 torchvision.models.detection: forward or direct call
 detectron2.engine.DefaultPredictor: predict or direct call
 mmdet.apis.init_detector: inference_detector on the returned model
 TF: detect_fn attribute access followed by a call on that value
```

These pairs show use of the checked capability, not that the found boxes are
correct. Control flow and runtime types are not evaluated. Bare imports,
references, comments, strings, subprocess commands, and setup alone supply no
evidence. Evidence split between modules does not count. This tool cannot
assess model quality or guarantee correctness for every input.

## Development

```sh
python3 -m unittest discover
python3 -m build
```

Distributed under the MIT license.
