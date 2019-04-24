# Keras implementation for Vocal separation UNet

Based on [SINGING VOICE SEPARATION WITH DEEP U-NET CONVOLUTIONAL NETWORKS](https://ismir2017.smcnus.org/wp-content/uploads/2017/10/171_Paper.pdf) by A. Jansson, et al.

Inspired by [Xiao-Ming/UNet-VocalSeparation-Chainer](https://github.com/Xiao-Ming/UNet-VocalSeparation-Chainer).

# Pretrained Models

CCmixter [model](https://users.dcc.uchile.cl/~voyanede/more/trained_models/vocal_20.h5) - loss: 0.0032 - val_loss: 
0.0032
CCMixter + DSD100 [model]( https://users.dcc.uchile.cl/~voyanede/more/trained_models/vocal_100_2.h5) - loss: 0.0032 - val_loss: 0.0067

# References

@inproceedings{   SiSEC16,   Title = {The 2016 Signal Separation Evaluation Campaign},   Address = {Cham},   Author = {Liutkus, Antoine and St{\"o}ter, Fabian-Robert 
and Rafii, Zafar and Kitamura, Daichi and Rivet, Bertrand and Ito, Nobutaka and Ono, Nobutaka and Fontecave, Julie},   Editor = {Tichavsk{\'y}, Petr and 
Babaie-Zadeh, Massoud and Michel, Olivier J.J. and Thirion-Moreau, Nad{\`e}ge},   Pages = {323--332},   Publisher = {Springer International Publishing},   Year = 
{2017},   booktitle = {Latent Variable Analysis and Signal Separation - 12th International Conference, {LVA/ICA} 2015, Liberec, Czech Republic, August 25-28, 2015, 
Proceedings}, }
