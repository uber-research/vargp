import os
import glob
import subprocess
import time
import logging


def wb_sweep(project, sweep_id):
  wb_envs = ' '.join([
    f'WANDB_API_KEY={os.getenv("WANDB_API_KEY")}',
    f'WANDB_PROJECT={project}',
    f'WANDB_USERNAME=sanyam',
    f'WANDB_DIR=/root',
    f'WANDB_DISABLE_CODE=true',
  ])

  label = f'{project}:{sweep_id.split("/")[-1]}'

  args = ['ma train tf docker --zone phx4-prod02 --respool /UberAI/Default'] + \
         [f'--label {label} --name-suffix [{label}] --num-cpus 8 --memory-size-mb 65536 --num-gpus 1'] + \
         [f'--custom-docker={os.getenv("CGP_DOCKER_IMAGE")} --command-line \'{wb_envs} wandb agent --count=1 {sweep_id}\'']

  arg_str = ' '.join(args)

  logging.debug(arg_str)

  proc = subprocess.Popen(arg_str, shell=True)
  proc.communicate()

  assert proc.returncode == 0, 'non-zero exit code {} found!'.format(proc.returncode)


def main(id, n=1, delay=1):
  for _ in range(n):
    wb_sweep('vargp', id)
    if delay:
      logging.debug(f'Delaying {delay}s')
      time.sleep(delay)


if __name__ == "__main__":
  logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s',
                      datefmt='%Y%m%d-%I:%M:%S%p',
                      level=getattr(logging, os.getenv('LOGLEVEL', 'WARNING')))

  import fire
  fire.Fire(main)
