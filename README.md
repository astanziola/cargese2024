# Differentiable acoustic simulations using `jwave`

The code provided in this repository has been developed for teaching purposes at the [European Summer School on Physical Acoustics and its Applications](https://physacoustics.sciencesconf.org/#:~:text=and%20its%20applications-,The%20European%20Summer%20School%20on%20Physical%20Acoustics%20and%20its%20applications,16th%2D20th%202024.) (2024).


## Getting started


#### 1. Install required packages

```bash
pip install -r requiremnts.txt
```

If you are on a linux machin with an NVIDIA gpu, install the GPU version of jax as well

```bash
pip install jax[cuda12]
```


Then, start the jupyter server with

```bash
jupyter lab
```

and open the `Tutorial` notebook.

You can directly run this tutorial on Google Colab by following the link https://githubtocolab.com/astanziola/cargese2024/blob/main/Tutorial.ipynb 

## Literature

- [JAX basics](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html)
- [Automatic differentiation cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Differentiable modelling review](https://www.nature.com/articles/s43017-023-00450-9). Even if focused on geosciences, it gives a good overview of the possibilities given by automatic differentiation in scientific modelling
- [jwave paper](https://www.softxjournal.com/article/S2352-7110(23)00034-1/fulltext)
- [Linear uncertainty quantification](https://arxiv.org/abs/1610.08716) and its [applications in ultrasound simulations](https://pubmed.ncbi.nlm.nih.gov/37166991/)