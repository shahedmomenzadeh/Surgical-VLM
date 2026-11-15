# `main.py` (original project) — External Entry Point

The original project's thin wrapper that runs the (now independent)
`video_pipeline` package against **this project's own** `config.yaml`:

```python
import os
from video_pipeline.cli import main

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")

if __name__ == "__main__":
    main(config=CONFIG_PATH)
```

## Usage

```bash
python main.py                  # run all 15 steps using the old project's config
python main.py --step download  # run one step
python main.py --config other.yaml   # any config override also works
```

## Notes

- The package itself is fully independent (bundled `config.yaml`, `.env`,
  `requirements.txt`); this file only pins the config for the old project's
  data layout.
- Running the package standalone: `cd video_pipeline && python -m video_pipeline`.

See [cli.md](cli.md) for the step list and dispatch logic.
