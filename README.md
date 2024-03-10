## FaceTalk: Audio-Driven Motion Diffusion for Neural Parametric Head Models<br><sub>Official PyTorch implementation of the paper</sub>

https://github.com/shivangi-aneja/FaceTalk/assets/37518691/c1d61981-dd30-450c-9d72-d67649ba2203



**FaceTalk: Audio-Driven Motion Diffusion for Neural Parametric Head Models**<br>
Shivangi Aneja, Justus Thies, Angela Dai, Matthias Niessner<br>
https://shivangi-aneja.github.io/projects/facetalk <br>

Abstract: *We introduce FaceTalk, a novel generative approach designed for synthesizing high-fidelity 3D motion sequences of talking human heads from input audio signal. To capture the expressive, detailed nature of human heads, including hair, ears, and finer-scale eye movements, we propose to couple speech signal with the latent space of neural parametric head models to create high-fidelity, temporally coherent motion sequences. We propose a new latent diffusion model for this task, operating in the expression space of neural parametric head models, to synthesize audio-driven realistic head sequences. In the absence of a dataset with corresponding NPHM expressions to audio, we optimize for these correspondences to produce a dataset of temporally-optimized NPHM expressions fit to audio-video recordings of people talking. To the best of our knowledge, this is the first work to propose a generative approach for realistic and high-quality motion synthesis of volumetric human heads, representing a significant advancement in the field of audio-driven 3D animation. Notably, our approach stands out in its ability to generate plausible motion sequences that can produce high-fidelity head animation coupled with the NPHM shape space. Our experimental results substantiate the effectiveness of FaceTalk, consistently achieving superior and visually natural motion, encompassing diverse facial expressions and styles, outperforming existing methods by 75% in perceptual user study evaluation.*

<br>

### <a id="section0">0. TODOS</a>
- [ ] Add Source code
- [ ] Add the link to the NPHM model
- [ ] Add license files
- [ ] Add the pretrained checkpoints
- [ ] Provide the form for getting access to the dataset and Identity Latent Codes
- [ ] Provide the link to missing Identity Latent Codes from the NPHM model
- [ ] Provide script for generating fixed length segments (used in training)



### <a id="section1">1. Getting started</a>

#### Pre-requisites
- Linux
- NVIDIA GPU + CUDA 11.8 
- Python 3.9

#### Installation
It is recommended to use `conda` for installing dependencies, please install it before creating the environment. The script for creating the environment is provided as `environment.sh`. Note that the dependency `pyg-lib` for `pytorch_geometric` package is build using CUDA from the machine. If your machine has different CUDA version other than 11.8, you need to install the matching version of `pyg-lib` from the [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Create the environment using the following command:
```.bash
./environment.sh
```



### <a id="section2">2. Pre-trained Models required for training FaceTalk</a>
Please download these models, as they will be required for experiments.

| Path                                             | Description
|:-------------------------------------------------| :----------
| [NPHM](https://simongiebenhain.github.io/NPHM//) | We use NPHM neural 3DMM in our experiments. NPHM takes as input Identity and Expression blendshapes and predicts SDF, from which meshes are extracted using MarchingCubes. We used the **NPHM Backward Deformation model** for our experiments, different from the one released with original NPHM paper. Using any other version of NPHM model than used in this project might lead to wrong expression predictions. The model & required indices can be downloaded from [here](). The latent codes for the identities will be provided along with the dataset once the signed user agreement is validated. Extract and copy these into `ckpts/nphm_geometric/` in the project directory. NPHM related assets can be downloaded from [here](), copy these into `assets/nphm/` in the project directory. 
| [Wave2Vec 2.0 (Pretrained)](https://arxiv.org/abs/2006.11477)                    | We used the Wave2Vec 2.0 as audio encoder, pretrained from Faceformer on VOCA dataset. This can be downloaded from [here](https://drive.google.com/file/d/1FMdc8PbEvQ5jkm_fJQird4ngPZWuvQ8S). Copy the model as `ckpts/voca_pretrained_audio_encoder.pt` in the project directory. This can be skipped if you don't want to train the model. 

### <a id="section3">3. Dataset</a>

To get access to the dataset and Identity Latent Codes, please fill out the user agreement form [here](). If you only need to run inference you can also do so with zero Identity code. If you have already filled the form, please wait for the confirmation email. Once the user agreement is validated, the dataset and Identity Latent Codes will be shared with you. We also provide FLAME tracking for comparing to Flame based baseline methods. The dataset should be placed in the `data/` directory. The dataset should be organized as follows:
  ```
  data/
  ├── <IDName>_<SequenceName>/
  │   │   ├── audio/
  │   │   │   ├── cleaned_audio.wav
  │   │   │   ├── wave2vec_audio_feats.pkl
  │   │   ├── nphm_tracking/
  │   │   │   ├── nphm_expressions/
  │   │   │   │   ├── frame_<frame_num>.npz
  │   │   │   ├── nphm_meshes/
  │   │   │   │   ├── frame_<frame_num>.ply
  │   │   ├── flame_2023_tracking/
  │   │   │   ├── flame_2023_template.ply
  │   │   │   ├── flame_meshes/
  │   │   │   │   ├── frame_<frame_num>.ply  
  ```


### <a id="section4">4. Training</a>
The code is well-documented and should be easy to follow.
* **Source Code:**   `$ git clone` this repo and install the dependencies from `environment.yml`. The source code is implemented in PyTorch Lightning and Wandb for logging so familiarity with these is expected.
* **Config:** All the config parameters are defined in `configs/nphm.yaml` file. The config files contains the paths to the dataset, results, and default hyperparameters used for training. Modify the paths to your dataset/asset paths, and the hyperparams (if needed).
* **Training**: Run the corresponding scripts to train for NPHM sequence generation  . The scripts for training are available in `trainer/` directory. By default, we export the results after 10 epochs, with meshes rendered at resolution of 64 for faster training. If you want to export the results at higher resolution, change the `resolution` parameter in the script file. 
  - **NPHM Sequence:** To train, run the following command:
  ```.bash
  python -m trainer.train_nphm_sequence_generation
  ```
  - **Flame Sequence:** To support comparing to existing baselines, we provide support to train our model on Flame trackings of our dataset (note that this is not what the paper proposes). To train, run the following command:
  ```.bash
  python -m trainer.train_flame_sequence_generation
  ```

### <a id="section5">5. Inference</a>

* **Inference on trained model:**  Once training is complete, then to evaluate, specify the path to the pretrained checkpoint in the script files and evaluate. We provide the scripts for evaluation in `tests/` directory.
  - **On Unseen Audios:** To render the results on the hybrid test set (reported in the paper) or in-the-wild audio clips, specify the path to audio wav file(s) in the script file and run the following command:
  ```.bash
  python -m tests.test_unseen_audio_sequence_generation
  ```
  - **With Dataloader:** To render results on the audio clips via the dataloader, run the following command:
  ```.bash
  python -m tests.test_nphm_sequence_generation
  ```

  To render the results on different identities, change the variable `identity_idx` in the script file(s) to the desired identity.
</br>

### More
Special thanks to authors of [NPHM](https://simongiebenhain.github.io/NPHM/) for proving their neural 3DMM and [NerSemble](https://tobias-kirschstein.github.io/nersemble/) for help with dataset construction. Finally, we would like to thank the authors of [Faceformer](https://evelynfan.github.io/audio2face/) for providing their pretrained model.  


### Citation

If you find our dataset or paper useful for your research , please include the following citation:

```
@misc{aneja2023facetalk,
      title={FaceTalk: Audio-Driven Motion Diffusion for Neural Parametric Head Models}, 
      author={Shivangi Aneja and Justus Thies and Angela Dai and Matthias Nießner},
      year={2023},
      eprint={2312.08459},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

</br>

### Contact Us

If you have questions regarding the dataset or code, please email us at shivangi.aneja@tum.de. We will get back to you as soon as possible.
