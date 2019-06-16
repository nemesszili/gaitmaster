import click

from util.process import load, train_evaluate, evaluate
from util.plots import plotAUC

@click.command()
@click.option('--feat-ext', type=click.Choice(['none', 'dense', 'lstm', '59']),
              default='none', help='Type of autoencoder used for feature extraction')              
@click.option('--same-day/--cross-day', default=True,
              help='Evaluate with data from session 2')
@click.option('--steps', type=click.IntRange(1, 10), default=5,
              help='Number of consecutive cycles used for evaluation')
@click.option('--identification/--verification', default=False,
              help='Measure identification/verification')
@click.option('--unreg/--reg', default=True,
              help='Use known or unknown impostors')
@click.option('--loopsteps/--regular', default=False,
              help='Evaluate with all cycles')
@click.option('--epochs', type=click.IntRange(1, 100), default=10,
              help='Number of epochs used for autoencoder training')
def main(feat_ext, same_day, steps, identification, unreg, loopsteps, epochs):
    if feat_ext == 'none':
        feat_ext = None
    params = (feat_ext, same_day, unreg, steps, identification, epochs)
    
    print()
    print('Running evaluation with:')
    print(' - feature extraction:   {}'.format(params[0]))
    print(' - same day:             {}'.format(params[1]))
    print(' - unknown impostors:    {}'.format(params[2]))
    if identification:
        print(' - steps:                {}'.format(1))
    elif loopsteps:
        print(' - loop steps:           {}'.format(loopsteps))
    else:
        print(' - steps:                {}'.format(params[3]))
    print(' - identification:       {}'.format(params[4]))
    if feat_ext in ['dense', 'lstm']:
        print(' - epochs:               {}'.format(params[5]))        
    print()

    if identification:
        params = (feat_ext, same_day, unreg, 1, identification, epochs)
        print(train_evaluate(params))
    else:
        if loopsteps:
            for i in range(1, 11):
                params = (feat_ext, same_day, unreg, i, identification, epochs)
                system_scores = train_evaluate(params)
                tpr, fpr, auc, eer = evaluate(system_scores)
        else:
            system_scores = train_evaluate(params)
            tpr, fpr, auc, eer = evaluate(system_scores)
            plotAUC(tpr, fpr, auc, eer)

if __name__ == '__main__':
    main()