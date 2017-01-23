# SSD-text detection: Text Detector

This is a modified SSD model for text detection.

Compared to faster R-CNN, SSD is much faster. In my expriment, SSD only needs about 0.05s for each image.

### Disclaimer
This is a re-implementation of mxnet SSD. The official
repository is available [here](https://github.com/dmlc/mxnet/tree/master/example/ssd).
The arXiv paper is available [here](http://arxiv.org/abs/1512.02325).

### Getting started
* Build MXNet: Make sure the extra operators for this example is enabled, and please following the the official instructions [here](https://github.com/dmlc/mxnet/tree/master/example/ssd).

### Train the model
I modify the original SSD on SynthText and ICDAR. Other datasets should
be easily supported by adding subclass derived from class `Imdb` in `dataset/imdb.py`.
See example of `dataset/pascal_voc.py` for details.
* Download the converted pretrained `vgg16_reduced` model [here](https://dl.dropboxusercontent.com/u/39265872/vgg16_reduced.zip), unzip `.param` and `.json` files
into `model/` directory by default.

To gain a good performance, we should train our model on SynthText which is a quite big dataset (about 40G) firstly, and then fine tune this model on ICDAR. If you want to apply this model for other applications, you can fine tune it on any dataset.

* Download the SynthText dataset [here](http://www.robots.ox.ac.uk/~vgg/data/scenetext/), and extract it into `data`.

Because SSD requires every image's size but SythText is too big, it will take too much time if we have to use opencv to read the images' size each time when we star training. So I use 'read_size.py' (`data/synthtext_img_size`) to creat a h5py file 'size.h5' to store the sizes of all images. You can copy this file to the extracted folder 'SynthText'.


* Start training:
```
python train_synthtext.py
```
### Fine tune the model
* Download the ICDAR challenge 2 dataset [here](http://rrc.cvc.uab.es/?ch=2&com=introduction), and extract it into `data`.

* Start training:
```
python train_icdar.py --finetune N
```
Please replace 'N' into an integer number which depends on the save model you train on SynthText.

### Try the demo
* After training, you can try your model on test images. I give two demos here (`demo.py` and `demo_savefig.py`). `demo.py` can visualize the detection result, while `demo_savefig.py` can save the detection result as images.

When running `demo_savefig.py`, please give the test images path.
* Run `demo.py`
```
# play with examples:
python demo.py --epoch 0 --images ./data/demo/test.jpg --thresh 0.5
```
* Check `python demo.py --help` for more options.

When running `demo_savefig.py`, please give the test images folder path.
* Run `demo_savefig.py`
```
# play with examples:
python demo_savefig.py --epoch 0 --images ./data/demo/test --thresh 0.5
```



