# ECSS

## Introduction
This is an implementation of the following paper.
[《Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling》](https://arxiv.org/pdf/2312.11947.pdf)
 (Accepted by AAAI2024)

[Rui Liu *](https://ttslr.github.io/), Yifan Hu, [Yi Ren](https://rayeren.github.io/), Xiang Yin, [Haizhou Li](https://colips.org/~eleliha/).

## Demo Page
[Speech Demo](https://walker-hyf.github.io/ECSS/)

## Dependencies
* For details about the operating environment dependency, see [FCTalker](https://github.com/walker-hyf/FCTalker/).
* You also need to install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/stable/index.html)（Used to support heterogeneous graph neural networks）

## Dataset
* You can download [dataset](https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL) from DailyTalk.
* You can get the emotion category and emotion intensity annotation information in the ./preprocessed_data/DailyTalk/ folder.

`1_1_d30|1|{Y EH1 S AY1 N OW1}|yes, i know.|none|1` The format of each piece of data is representing `sentence ID|speaker|phoneme sequence|original content|emotion|emotion intensity`

## Preprocessing

Run 
  ```
  python3 prepare_align.py --dataset DailyTalk
  ```
  for some preparations.

  For the forced alignment, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
  Pre-extracted alignments for the datasets are provided [here](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing). 
  You have to unzip the files in `preprocessed_data/DailyTalk/TextGrid/`. Alternately, you can [run the aligner by yourself](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/index.html). Please note that our pretrained models are not trained with supervised duration modeling (they are trained with `learn_alignment: True`).

  After that, run the preprocessing script by
  ```
  python3 preprocess.py --dataset DailyTalk
  ```

## Training

Train your model with
```
python3 train.py --dataset DailyTalk
```

## Inference

Only the batch inference is supported as the generation of a turn may need contextual history of the conversation. Try

```
python3 synthesize.py --source preprocessed_data/DailyTalk/val_*.txt --restore_step RESTORE_STEP --mode batch --dataset DailyTalk
```
to synthesize all utterances in `preprocessed_data/DailyTalk/val_*.txt`.

## Citing
To cite this repository:
```bibtex
@inproceedings{liu2024emotion,
  title={Emotion rendering for conversational speech synthesis with heterogeneous graph-based context modeling},
  author={Liu, Rui and Hu, Yifan and Ren, Yi and Yin, Xiang and Li, Haizhou},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={17},
  pages={18698--18706},
  year={2024}
}

```

## Author

E-mail：hyfwalker@163.com
