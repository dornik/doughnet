## DoughNet :doughnut: A Visual Predictive Model for Topological Manipulation of Deformable Objects

[Dominik Bauer](https://dominikbauer.io)<sup>1</sup>, [Zhenjia Xu](https://www.zhenjiaxu.com/)<sup>1,2</sup>, [Shuran Song](https://shurans.github.io/)<sup>1,2</sup><br>
<sup>1</sup> Columbia University, <sup>2</sup> Stanford University 

[Project website](https://dough-net.github.io/) &nbsp;&nbsp;&bull;&nbsp;&nbsp; [Paper](https://arxiv.org/pdf/2404.12524)

[![Presentation](https://img.youtube.com/vi/aMV8Tz1RRX4/0.jpg)](https://www.youtube.com/watch?v=aMV8Tz1RRX4)


## Dependencies

### Create the conda environment
- `conda env create -f environment.yml`

### Install additional submodules
- `git submodule init && git submodule update`
- `conda activate doughnet`
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast): `cd net/nvdiffrast && pip install -e . && cd ../..`
- [sdftoolbox](https://github.com/cheind/sdftoolbox/): `cd sim/sdftoolbox && pip install -e . && cd ../..`



## Downloads
- All required files are provided in [this Google Drive folder](https://drive.google.com/drive/folders/102fXzrjDNjHYeNK2stfIACiOel5jh9yn?usp=sharing).
- Download our dataset for training, testing on synthetic and real manipulation trajectories and place it in `data/dataset.h5`.
- If you are only interested in the dataset, you can check the provided `README` for a simple data loader and a brief description of the structure.
- Download our pretrained weights to reproduce the results in the paper and place them in `weights/{ae,dyn}.pth`.



## Evaluation
Using the provided weights, the evaluation reproduces the main results from the paper. Note that due to dataset preprocessing and weights trained from scratch using this public code base, the results may vary slightly. Alternatively, train the model from scratch or create a new dataset as described below. Make sure to adapt the paths in the config files accordingly.
- `python net/prediction.py --config-name dyn "settings.test_only=True"`


## Training
Using the provided dataset, the autoencoder and the dynamics prediction are trained in two stages, as shown below. Alternatively, generate a custom dataset as described below.

Note that for multi-GPU training, e.g., using 2 GPUs, the `settings.ddp` flag needs to be set in the config. Run the scripts below with `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 {path_to_script}` instead of `python {path_to_script}`.

### Geometry-topology Autoencoder
- `python net/prediction.py --config-name ae`
- This saves the weights in the corresponding hydra output directory (i.e., `outputs/{date}/{time when run was started}/best.pth`). Either copy them to the default path (`weights/ae.pth`), or adapt the `settings.resume_path` in `net/config/dyn.yaml` accordingly before starting the next stage.

### Dynamics Prediction
- `python net/prediction.py --config-name dyn`
- Again, the weights are saved in the corresponding hydra output directory. Follow the directions above to make sure that `settings.test_path` points to the desired weights when running subsequent evaluations.



## Generation
Our simulation with topology annotation may be used to generate additional scenes or completely new datasets. 

To this end, first, derive novel scene definitions from `template.yaml`, e.g., by adapting `to_pos` and `to_quat` (grasp pose), or `close_d` (final opening width).

### Simulation
- `python sim/generate.py`
- This will create a `log.pkl` with particle-based information (and `visualization.gif` if `render=True` in config) in the scene directory.

### Processing
- `python sim/process.py`
- This will process the simulated scenes in parallel and create `data.h5` with additional mesh-based information.


## Citation
```
@article{bauer2024doughnet,
  title={DoughNet: A Visual Predictive Model for Topological Manipulation of Deformable Objects},
  author={Bauer, Dominik and Xu, Zhenjia and Song, Shuran},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
