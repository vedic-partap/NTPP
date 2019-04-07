import matplotlib.pyplot as plt


def plot(args, train_loss_list, train_mape_t_list, train_mape_n_list, dev_loss_list, dev_mape_t_list, dev_mape_n_list):
    f1 = plt.figure(1)
    plt.plot(train_loss_list)
    plt.plot(dev_loss_list)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.savefig(args['save_dir'] + 'loss.jpg')
    plt.close(f1)

    f2 = plt.figure(2)
    plt.plot(train_mape_t_list)
    plt.plot(dev_mape_t_list)
    plt.title('MAPE_t')
    plt.ylabel('MAPE_t')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.savefig(args['save_dir'] + 'MAPE_t.jpg')
    plt.close(f2)

    f3 = plt.figure(3)
    plt.plot(train_mape_n_list)
    plt.plot(dev_mape_n_list)
    plt.title('MAPE_n')
    plt.ylabel('MAPE_n')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Dev'], loc='upper left')
    plt.savefig(args['save_dir'] + 'MAPE_n.jpg')
    plt.close(f3)
