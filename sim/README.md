## Setup
- `mamba env create -f environment.yml`
- `git submodule init && cd sim/sdftoolbox && pip install -e`

## Generate data
- derive scenes from `generate/config/template.yaml` by adapting `to_pos` and `to_quat` (grasp pose), as well as `close_d` (final opening width)
- simulate them via `generate.py`, creating `log.pkl` with particle-based information (and `visualization.gif` if `render=True` in config)
- process simulated scenes in parallel via `process.py`, creating `data.h5` with additional mesh-based information
