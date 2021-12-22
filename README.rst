deepsignal
==========


Documentation
-------------
v0.1.10
-------
make sure results of each read be written together in call_mods' output

v0.1.9
------
enable multi-class (>=3) training/predicting,

fix bug of extrating contig name from fast5s

v0.1.8
------
modify denoise module
fix success_file bug
update README

v0.1.7
------
Prevent Queue.qsize() from raising NotImplementedError on Mac OS X (github: vterron/lemon@9ca6b4b)
covert raw signals to pA values before normalization in extract_feature module
add denoise module
add module-chosen options of model (rnn/base/cnn), re-train human model

v0.1.6
------
add option --positions in extract_features module,
add option/function of binary_format feature file to speed up training

v0.1.5
------
normalize probs before output

v0.1.4
------
change the loss function to weighted_cross_entropy_with_logits,
allow training using unbalanced samples.

v0.1.3
------
fix the deadlock issue in multiprocessing

v0.1.2
------
add MANIFEST.in file

v0.1.1
------
3 modules (extract, call_mods, train) supported