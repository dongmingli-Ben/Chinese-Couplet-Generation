# Chinese Couplet Generation

## Overview

This repository contains codes to get raw text Chinese couplets, codes to train four models, and codes to generate a couplet response using saved models.

## Requirement

pytorch 1.2

## File Structure

Crawling raw text couplets and preprocessing raw texts are in [utils](./utils) folder.

The models are in ```model.py```. The training configuration of each model is in ```.py``` files with heading ```train```. ```transformer.py``` contains the implementation of the submodule of models and large part of codes come from [annotated-transformer](https://github.com/harvardnlp/annotated-transformer).

The codes where saved models are used to generate couplet responses is in ```test.py```.

Codes in [measure](./measure) folder are used to calculate BLEU score (from [BLEU4Python](https://github.com/zhyack/BLEU4Python)).

*Note: The models use pre-trained word vectors as part of the embedding. Models use this [embedding](https://github.com/Embedding/Chinese-Word-Vectors). The specific embedding used in this repository can be downloaded from [here](https://pan.baidu.com/s/1vPSeUsSiWYXEWAuokLR0qQ). **To run the codes, the word vectors have to be downloaded first.***

## Model Description


| Model Name | Description |
| - | - |
| Vanilla Transformer | The standard transformer model |
| Oneway Model | The model is basically a transformer encoder |
| Twoway Model | The model is basically a transformer decoder <br />where the source is sent to decoder as input <br />while partially decoded couplet response is <br />fed to decoder as context. |
| Memory Model | The model is basically a transformer followed <br />by a memory network. The memory network is <br />different from common ones. The memory <br />network is basically a transformer decoder <br />without self-attention. Similarly, the partially <br />decoded response is used as the memory. |

## Sample

These two samples are generated using Twoway Model (run ```test.py```).

```
请输入上联（按q退出）:欢天喜地度佳节
上联：欢天喜地度佳节
下联：张灯结彩迎新春
```

```
请输入上联（按q退出）:知足心常乐
上联：知足心常乐
下联：诚身欲能直
```
