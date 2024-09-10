# my_script.py
from hawk_pack_py import MyRustLibrary

lib = MyRustLibrary()
lib.set_state(42)
print(lib.get_state()) # Should output 42
