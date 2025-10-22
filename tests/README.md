This folder contains quick smoke tests for local development.

Run the smoke test after starting the API locally (default expects http://127.0.0.1:8000):

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD/src
./tests/smoke_test.sh
```

You can pass a base URL as the first argument, e.g. `./tests/smoke_test.sh https://my-host`.
