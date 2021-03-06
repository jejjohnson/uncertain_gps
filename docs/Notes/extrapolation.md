# Extrapolation

One interesting problem that is related to uncertainty is how well this extrapolates for unseen regions (whether it is spatially or temporally).

---

## Smart Combination of Kernels
Related to what @Gus and I were talking about earlier. I think a GP would be better at extrapolating if we use kernel that is better defined for extrapolation.


For example, we can just use a combination of well defined kernels with actual thought into the trends we expect like they did for the classic Mauna dataset: ([sklearn demo](https://scikit-learn.org/stable/modules/gaussian_process.html#gpr-on-mauna-loa-co2-data)) , ([tensorflow demo](https://peterroelants.github.io/posts/gaussian-process-kernel-fitting/#Mauna-Loa-CO%E2%82%82-data)). They ended up using a combination of `RBF` + `RBF * ExpSinSquared` + `RQ` + `RBF` + `White` with arbitrary scalers in front of all of the terms. I have no clue how in the world they came up with that…

---

## Fourier Basis Functions

In general any GP that is approximated with Fourier Basis functions will be good at finding periodic trends. For example the Sparse Spectrum GP (SSGP) (as you mentioned in your [paper](https://www.uv.es/lapeva/papers/2016_IEEE_GRSM.pdf) Gustau, pg68) (original [paper](http://www.jmlr.org/papers/v11/lazaro-gredilla10a.html) for SSGP) is related to the Fourier features method but GP-ed.

??? details "Resources"
    * [Gustau Paper](https://www.uv.es/lapeva/papers/2016_IEEE_GRSM.pdf)
    > Gustaus paper summarizing GPs in the context of Earth science. Briefly mentions SSGPs.
    * [Presentation](https://www.hiit.fi/wp-content/uploads/2018/04/Spectral-Kernels-S12.pdf)
    > A lab that does a lot of work on spectral kernels did a nice presentation with a decent summary of the literature.
    * [SSGP Paper](http://www.jmlr.org/papers/v11/lazaro-gredilla10a.html)
    > original paper on SSGP by Titsias.
    * SSGP w. Uncertain Inputs [Paper](http://proceedings.mlr.press/v70/pan17a.html)
    > Paper detailing how one can do the Taylor expansion to propagate the errors.
    * [VSSGP Paper](https://arxiv.org/pdf/1503.02424.pdf) | [Poster](https://www.cs.ox.ac.uk/people/yarin.gal/website/PDFs/ICML_2015_Improving_poster.pdf)  | [ICML Presentation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwi97Kvdl6TpAhUlSxUIHeU5CooQFjAAegQIBRAB&url=http%3A%2F%2Fmlg.eng.cam.ac.uk%2Fyarin%2FPDFs%2FICML_Improving_presentation.pdf&usg=AOvVaw1_Chb_QJzTNmz8NpuDNtAk) 
    > Original paper by Yarin Gal and presentation about the variational approach to the SSGP method. The original author did the code in Theano so I don't recommend you use it nor try to understand it. It's ugly...
    * Variational Fourier Features for GPs - [Paper](http://www.jmlr.org/papers/volume18/16-579/16-579.pdf)
    > Original paper by James Hensman et al. using variational fourier features for GPs. To be honest, I'm still not 100% sure what the difference is between this method and the method by Yarin Gal... According to the paper they say:
    "Gal and Turner proposed variation inference in a sparse spectrum model that is derived form a GP model. Our work aims to directly approximate the posterior of the try models using a variational representation." I still don't get it.

---

## Spectral Mixture Kernel

Or a kernel that was designed to include all the parameters necessary to find patterns like the spectral mixture kernel ([paper](https://arxiv.org/pdf/1302.4245.pdf), pg 2, eq. 12).
Lastly, just use a neural network and slap a GP layer at the end and let the data tell you the pattern.

??? details "Resources"
    * Original [Paper](https://arxiv.org/pdf/1302.4245.pdf)
    > original paper with the derivation of the spectral mixture kernel.
    * [Paper](https://arxiv.org/abs/1808.01132)
    > some extenions to Multi-Task / Multi-Output / Multi-Fidelity problems
    * [Thesis](https://lib.ugent.be/fulltxt/RUG01/002/367/115/RUG01-002367115_2017_0001_AC.pdf) of Vincent Dutordoi (chapter 6, pg. 67)
    > Does the derivation for the spectral mixture kernel extension to incorporate uncertain inputs. Uses exact moment matching (which is an expensive operation...) but in theory, it should better propagate the uncertain inputs. It's also a **really good** thesis that explains how to propagate uncertain inputs through Gaussian processes (chapter 4, pg. 42). **Warning**: the equations are nasty...
    * [GPyTorch Demo](https://gpytorch.readthedocs.io/en/latest/examples/01_Exact_GPs/Spectral_Mixture_GP_Regression.html)
    > The fastest implementation you'll find on the internet.

---

## Large Scale

All of the above methods can be trained using sparse and variational methods to make them scale to really large amounts of data. Although I think they kind of fail when you have a lot of dimensions. No more than 3-5 is what I’ve seen in the literature. And typically they only do toy examples with 1D data.

---

## Error Propagation

Now to couple them with the Taylor expansion would be interesting. As long as the methods have a kernel that is differentiable (wrt inputs), then it should be possible.
