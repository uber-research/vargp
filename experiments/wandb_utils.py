import wandb
from datetime import datetime


def generate_vargp_sweep(project='continual_gp', submit=False,
                         dataset=None, epochs=500, M=60, lr=3e-3, beta=10.0):
  assert dataset in ['s_mnist']

  md = datetime.now().strftime('%h%d')
  sweep_config = {
    'name': f'[{md}] {dataset}-{M}-{lr}-{beta}',
    'method': 'random',
    'parameters': {
      'epochs': {
        'value': epochs
      },
      'M': {
        'value': M
      },
      'lr': {
        'value': lr
      },
      'beta': {
        'value': beta
      }
    },
    'program': 'experiments/vargp.py',
    'command': [
      '${env}',
      '${interpreter}',
      '${program}',
      dataset,
      '${args}',
    ]
  }
  
  if submit:
    sweep_id = wandb.sweep(sweep_config, project=project)
    return sweep_id
  
  return sweep_config


if __name__ == "__main__":
  import fire
  fire.Fire(dict(
    gvs=generate_vargp_sweep,
  ))
