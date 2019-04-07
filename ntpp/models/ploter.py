import matplotlib.pyplot as plt


def plot(args, loss_list, mape_t_list, mape_n_list):
    f1 = plt.figure(1)
    plt.plot(loss_list)
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.savefig(args['save_dir'] + 'loss.jpg')
    plt.close(f1)

    f2 = plt.figure(2)
    plt.plot(mape_t_list)
    plt.title('MAPE_t')
    plt.ylabel('MAPE_t')
    plt.xlabel('epoch')
    plt.savefig(args['save_dir'] + 'MAPE_t.jpg')
    plt.close(f2)

    f3 = plt.figure(3)
    plt.plot(mape_n_list)
    plt.title('MAPE_n')
    plt.ylabel('MAPE_n')
    plt.xlabel('epoch')
    plt.savefig(args['save_dir'] + 'MAPE_n.jpg')
    plt.close(f3)
