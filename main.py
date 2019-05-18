import click

from util.process import load, train_evaluate, evaluate
from util.plots import plotAUC

# SAME DAY:

# CROSS DAY:

# TODO: add support for identification/verification
# TODO: add support for LSTM
# TODO: document code
# TODO: t-SNE plots

@click.command()
@click.option('--raw/--feat', default=True,
              help='Use raw data')
@click.option('--feat-ext', type=click.Choice(['none', 'dense', 'lstm']),
              default='none', help='Type of autoencoder used for feature extraction')              
@click.option('--same-day/--cross-day', default=True,
              help='Evaluate with data from session 2')
@click.option('--steps', type=click.IntRange(1, 10), default=5,
              help='Number of consecutive cycles used for evaluation')
@click.option('--identification/--verification', default=False,
              help='Measure identification/verification')
@click.option('--loopsteps', default=False,
              help='Evaluate with all cycles')
def main(raw, feat_ext, same_day, steps, identification, loopsteps):
    if feat_ext == 'none':
        feat_ext = None
    params = (raw, feat_ext, same_day, steps, identification)
    
    print()
    print('Running evaluation with:')
    print(' - raw data:             {}'.format(params[0]))
    print(' - feature extraction:   {}'.format(params[1]))
    print(' - same day:             {}'.format(params[2]))
    if identification:
        print(' - steps:                {}'.format(1))
    elif loopsteps:
        print(' - loop steps:           {}'.format(loopsteps))
    else:
        print(' - steps:                {}'.format(params[3]))
    print(' - identification:       {}'.format(params[4]))        
    print()

    if identification:
        params = (raw, feat_ext, same_day, 1, identification)
        print(train_evaluate(params))
    else:
        if loopsteps:
            for i in range(1, 11):
                print(' - steps: {}'.format(i))
                params = (raw, feat_ext, same_day, i, identification)
                system_scores = train_evaluate(params)
                print()
                tpr, fpr, auc, eer = evaluate(system_scores)
        else:
            system_scores = train_evaluate(params)
            tpr, fpr, auc, eer = evaluate(system_scores)
            plotAUC(tpr, fpr, auc, eer)

if __name__ == '__main__':
    main()