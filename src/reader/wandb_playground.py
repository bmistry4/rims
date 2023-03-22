import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("user/rims")
summary_list = []
config_list = []
name_list = []
for run in runs:
    # run.summary are the output key/values like accuracy.
    # We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # run.config is the input metrics.
    # We remove special values that start with _.
    config = {k: v for k, v in run.config.items() if not k.startswith('_')}
    config_list.append(config)

    # run.name is the name of the run.
    name_list.append(run.name)

import pandas as pd

summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})
all_df = pd.concat([name_df, config_df, summary_df], axis=1)

# all_df.to_csv("project.csv")

# only get the newest runs (i.e. those over id 18)
df_postprocess = all_df[all_df["args/id"] >= 29]
df_postprocess = df_postprocess.filter(
    ['name', 'args/id', 'args/cell', 'args/name_postfix', 'metric/train/loss/last-10',
     'metric/test/loss/last-10/dataloader_idx_1', 'metric/valid/loss/last-10/dataloader_idx_0', 'epoch'])
df_postprocess = df_postprocess.groupby(['args/id', 'args/name_postfix'])
df_stats = df_postprocess.describe()
df_stats_summary = df_stats.filter(['args/name_postfix', ('metric/train/loss/last-10', 'std'),
                                    # ('metric/valid/loss/last-10/dataloader_idx_0', 'std'),
                                    ('metric/test/loss/last-10/dataloader_idx_1', 'std'),
                                    ('metric/train/loss/last-10', 'mean'),
                                    # ('metric/valid/loss/last-10/dataloader_idx_0', 'mean'),
                                    ('metric/test/loss/last-10/dataloader_idx_1', 'mean'),
                                    ('epoch', 'mean')]
                                   )
# df = df.std()
