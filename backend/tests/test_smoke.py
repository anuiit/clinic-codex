# type: ignore
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUndefinedVariable=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownParameterType=false, reportGeneralTypeIssues=false, reportAttributeAccessIssue=false
# The test uses dynamic module stubs to avoid importing heavy ML deps in CI; keep
# diagnostics suppressed for this file.
import importlib
import sys
import types


def _insert_stubs():
    """Insert lightweight stub modules to avoid importing heavy ML deps.

    This creates a minimal `codex_model` module with a CodexClassifier stub and
    a `codex_pipeline.segmentation` module with MobileSAMSegmenter so that
    importing examples.flask_api is safe in CI where torch/mobile-sam aren't
    installed.
    """
    # codex_model stub
    mod = types.ModuleType("codex_model")

    class CodexClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify(self, *args, **kwargs):
            return {"class_name": "stub", "confidence": 1.0, "rejected": False, "top_k": []}

        def classify_batch(self, *args, **kwargs):
            return []

    setattr(mod, "CodexClassifier", CodexClassifier)
    sys.modules["codex_model"] = mod

    # codex_pipeline.segmentation stub
    pkg = types.ModuleType("codex_pipeline")
    seg_mod = types.ModuleType("codex_pipeline.segmentation")

    class MobileSAMSegmenter:
        def __init__(self, *args, **kwargs):
            pass

        def segment_page(self, img):
            return []

        def extract_crops(self, img, proposals):
            return []

    setattr(seg_mod, "MobileSAMSegmenter", MobileSAMSegmenter)
    sys.modules["codex_pipeline"] = pkg
    sys.modules["codex_pipeline.segmentation"] = seg_mod

    # PIL.Image stub
    try:
        import PIL  # type: ignore
    except Exception:
        pil_mod = types.ModuleType("PIL")
        setattr(pil_mod, "Image", types.SimpleNamespace(open=lambda *a, **k: None))
        sys.modules["PIL"] = pil_mod

    # Flask stub if flask isn't installed in the test environment
    try:
        import flask  # type: ignore
    except Exception:
        flask_stub = types.ModuleType("flask")

        class DummyResponse:
            def __init__(self, status_code=200, data=b"", headers=None):
                self.status_code = status_code
                self.data = data
                self.headers = headers or {}

        class Flask:
            def __init__(self, *args, **kwargs):
                self._routes = {}

            def route(self, path, methods=None):
                def decorator(fn):
                    self._routes[path] = fn
                    return fn

                return decorator

            def after_request(self, fn):
                return fn

            def test_client(self):
                parent = self

                class Client:
                    def get(self, path):
                        fn = parent._routes.get(path)
                        if fn is None:
                            return DummyResponse(status_code=404)
                        # Call route function; if it returns tuple, handle status
                        rv = fn()
                        if isinstance(rv, tuple):
                            body, status = rv
                            return DummyResponse(status_code=status, data=str(body).encode())
                        return DummyResponse(status_code=200, data=str(rv).encode())

                return Client()

        setattr(flask_stub, "Flask", Flask)
        setattr(flask_stub, "jsonify", lambda x: x)
        # Minimal request and send_file placeholders
        setattr(flask_stub, "request", types.SimpleNamespace(headers={}, args={}, files={}))
        setattr(flask_stub, "send_file", lambda path: path)
        sys.modules["flask"] = flask_stub


def test_flask_app_health():
    """Smoke test: import the Flask app with safe stubs and hit a lightweight endpoint.

    We stub heavy ML modules so importing examples.flask_api won't require torch or
    other large deps. Then we prefer hitting /classes which reads a JSON file and
    is safe.
    """
    _insert_stubs()
    # Import by file path package structure: add repo backend dir to sys.path so
    # the 'examples' package inside backend can be imported as a top-level module.
    import os
    repo_backend = os.path.dirname(os.path.dirname(__file__))
    if repo_backend not in sys.path:
        sys.path.insert(0, repo_backend)
    flask_mod = importlib.import_module("examples.flask_api")
    app = getattr(flask_mod, "app", None)
    assert app is not None, "Flask app not found in module"

    # Use the pytest-flask client fixture to query a safe endpoint
    # Use Flask's test client directly to avoid requiring pytest-flask fixture
    test_client = app.test_client()
    resp = test_client.get("/classes")
    assert resp.status_code == 200
