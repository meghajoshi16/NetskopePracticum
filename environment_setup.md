# Environment Setup

1. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Navigate to the `/environment` folder
    - Use the command: `conda env create -f env_netskope_<OS>.yml`
    - Make sure to modify the command based on your OS (`linux`, `mac`, or `win`).
    - This should create an environment named `env_netskope`. 
1. Activate the conda environment:
    - Windows command: `activate env_netskope` 
    - MacOS / Linux command: `conda activate env_netskope`

