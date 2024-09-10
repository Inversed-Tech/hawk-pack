
```bash
# Setup
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build the Rust-Python package
maturin develop --release
python test.py
```
