import matplotlib.pyplot as plt

def plotAUC(tpr, fpr, auc_value, eer):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='black', lw=lw, 
            label='System AUC = %0.4f, System EER= %0.4f' % (auc_value, eer))
    plt.plot([0, 1], [0, 1], color='darkorange', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()
    return
