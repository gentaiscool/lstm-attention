# LSTM with Attention by using Vector Context for Classification task

This code is used in <a href="https://arxiv.org/abs/1805.12307">Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision</a> paper. The idea is to consider the importance of every words in the inputs and use it during the classification.
The word importance then is normalized through the softmax layer and weighted sum to perform the classification.

If you are using the code in your work, please cite the following (Will appear in ICASSP 2018 Proceeding)
```
@article{winata2018attention,
  title={Attention-Based LSTM for Psychological Stress Detection from Spoken Language Using Distant Supervision},
  author={Winata, Genta Indra and Kampman, Onno Pepijn and Fung, Pascale},
  journal={arXiv preprint arXiv:1805.12307},
  year={2018}
}
```

You can easily get the attention weights from the model and visualized them

<img src="img/stressed.jpg" width=500>
