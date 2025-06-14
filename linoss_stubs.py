"""
Stub file for LinOSS imports to help with static analysis
"""
# Type stubs for LinOSS
try:
    import sys
    sys.path.insert(0, '/home/rustuser/pyth/linoss_kos')
    from models.LinOSS import LinOSSLayer  # type: ignore
except ImportError:
    # Fallback stub
    class LinOSSLayer:
        def __init__(self, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            pass
