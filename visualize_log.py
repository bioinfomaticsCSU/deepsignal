import matplotlib.pyplot as plt
import argparse


def draw_log(logdir):
    '''
    draw the figures of loss accuracy recall precision from train.txt and test.txt
    lines:
        epoch:0, iterid:100, loss:3.545, accuracy:0.501, recall:0.378, precision:0.511
    '''
    f_train = open(logdir+'/'+'train.txt', 'r')
    f_test = open(logdir+'/'+'test.txt', 'r')

    train_loss = []
    train_accuracy = []
    train_recall = []
    train_precision = []

    test_loss = []
    test_accuracy = []
    test_recall = []
    test_precision = []

    for line in f_train:
        epoch, iterid, loss, accuracy, recall, precision = line.strip().split()
        loss = float(loss.split(":")[-1].strip(','))
        accuracy = float(accuracy.split(":")[-1].strip(','))
        recall = float(recall.split(":")[-1].strip(','))
        precision = float(precision.split(":")[-1].strip(','))

        train_loss.append(loss)
        train_accuracy.append(accuracy)
        train_recall.append(recall)
        train_precision.append(precision)

    for line in f_test:
        epoch, iterid, loss, accuracy, recall, precision = line.strip().split()
        loss = float(loss.split(":")[-1].strip(','))
        accuracy = float(accuracy.split(":")[-1].strip(','))
        recall = float(recall.split(":")[-1].strip(','))
        precision = float(precision.split(":")[-1].strip(','))

        test_loss.append(loss)
        test_accuracy.append(accuracy)
        test_recall.append(recall)
        test_precision.append(precision)

    f_train.close()
    f_test.close()

    fig = plt.figure()
    ax_loss = fig.add_subplot(2, 2, 1)
    ax_accuracy = fig.add_subplot(2, 2, 2)
    ax_recall = fig.add_subplot(2, 2, 3)
    ax_precision = fig.add_subplot(2, 2, 4)

    assert(len(train_loss) == len(test_loss))
    ax_loss.plot(range(len(train_loss)), train_loss, 'orange')
    ax_loss.plot(range(len(test_loss)), test_loss, 'blue')
    ax_loss.legend(['train', 'valid'])
    ax_loss.set_xlabel('time')
    ax_loss.set_title("loss")

    assert(len(train_accuracy) == len(test_accuracy))
    ax_accuracy.plot(range(len(train_accuracy)), train_accuracy, 'orange')
    ax_accuracy.plot(range(len(test_accuracy)), test_accuracy, 'blue')
    ax_accuracy.legend(['train', 'valid'])
    ax_accuracy.set_xlabel('time')
    ax_accuracy.set_title("accuracy")

    assert(len(train_recall) == len(test_recall))
    ax_recall.plot(range(len(train_recall)), train_recall, 'orange')
    ax_recall.plot(range(len(test_recall)), test_recall, 'blue')
    ax_recall.legend(['train', 'valid'])
    ax_recall.set_xlabel('time')
    ax_recall.set_title("recall")

    assert(len(train_precision) == len(test_precision))
    ax_precision.plot(range(len(train_precision)), train_precision, 'orange')
    ax_precision.plot(range(len(test_precision)), test_precision, 'blue')
    ax_precision.legend(['train', 'valid'])
    ax_precision.set_xlabel('time')
    ax_precision.set_title("precision")

    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


if __name__ == "__main__":
    argv = argparse.ArgumentParser()
    argv.add_argument('-i', '--logdir', required=True)
    argv = argv.parse_args()
    draw_log(argv.logdir)
