"""Kimodo inference service. The Blender client has no dependency on this package."""
__version__ = '0.3.1'


def check_bundled_core():
    """Fail clearly if an incomplete distribution falls back to an external core."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    expected = root / 'kimodo' / '__init__.py'
    spec = importlib.util.find_spec('kimodo')
    if not expected.is_file() or spec is None or Path(spec.origin).resolve() != expected:
        raise RuntimeError('Kimodo 内置核心缺失或被外部包覆盖，请使用完整服务端文件夹并从该目录启动。')
