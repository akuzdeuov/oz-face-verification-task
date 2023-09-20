import matplotlib.pyplot as plt
import numpy as np


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(25, 4, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

    
def plot_losses(train_losses, val_losses, epochs, figure_name="train_val_losses.png"):
    fig = plt.figure()
    plt.plot(range(epochs), train_losses, color='red', label='train_loss')
    plt.plot(range(epochs), val_losses, color='blue', label='val_loss')
    #plt.ylim([0.0, max(train_losses)+0.1])
    #plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and val loss plot')
    plt.legend(loc='lower right')
    plt.show() 
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()
    
    
def plot_accs(train_accs, val_accs, epochs, figure_name="train_val_acc.png"):
    fig = plt.figure()
    plt.plot(range(epochs), train_accs, color='red', label='train_acc')
    plt.plot(range(epochs), val_accs, color='blue', label='val_acc')
    #plt.ylim([0.0, 1])
    #plt.xlim([0, epochs + 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and val acc plot')
    plt.legend(loc='lower right')
    plt.show() 
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()