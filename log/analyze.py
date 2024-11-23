from pathlib import Path
from typing import *
import pandas as pd
import json
import matplotlib.pyplot as plt

RESULTS_DIR = Path('.').resolve()


def get_full_run_dirs_from_experiment(experiment_dir: Path) -> List[Path]:
    run_folders = [folder for folder in experiment_dir.iterdir() if folder.stem.isnumeric()]
    return [folder for folder in run_folders if (folder / "test").exists()]


def get_stats_from_run(run_dir: Path, run_type: str) -> pd.DataFrame:
    with open(run_dir / run_type / "stats.json", 'r') as f:
        stats_content = f.read().splitlines()
    
    stats_list = [json.loads(line) for line in stats_content]
    df = pd.DataFrame(data=stats_list)
    return df.set_index('epoch')


def get_best_validation_epoch(run_dir: Path) -> int:
    stats = get_stats_from_run(run_dir, 'val')
    return stats['accuracy'].argmax()


def print_mean_std(values: List[Any], title: str):
    values = pd.Series(values)
    mean = values.mean()
    std = values.std()

    print(f"{title}: {mean:.2f} ± {std:.2f}")


def print_mean_epoch_time(experiment_dir: Path, run_type: str):
    runs = get_full_run_dirs_from_experiment(experiment_dir)

    mean_epoch_times = []
    best_test_accruacies = []
    for run in runs:
        test_stats = get_stats_from_run(run, run_type)
        mean_epoch_times.append(test_stats['time_epoch'].mean())
        best_epoch = get_best_validation_epoch(run)
        best_test_accruacies.append(test_stats.loc[best_epoch]['accuracy'])

    print_mean_std(mean_epoch_times, f"mean epoch time ({run_type})")


def print_final_test_metric(experiment_dir: Path, metric: str, metric_transform : Callable[[Any], Any] = None):
    runs = get_full_run_dirs_from_experiment(experiment_dir)
    best_test_metrics = []
    for run in runs:
        test_stats = get_stats_from_run(run, "test")
        best_epoch = get_best_validation_epoch(run)
        metric_value = test_stats.loc[best_epoch][metric]
        if metric_transform is not None:
            metric_value = metric_transform(metric_value)
        best_test_metrics.append(metric_value)

    print_mean_std(best_test_metrics, f"final test {metric}")


def analyze_experiment(experiment_dir: Path):
    print(f"=== Analyzing {experiment_dir.name} ===")

    print_mean_epoch_time(experiment_dir, "train")
    print_mean_epoch_time(experiment_dir, "val")
    print_mean_epoch_time(experiment_dir, "test")
    print_final_test_metric(experiment_dir, 'accuracy', lambda acc: acc * 100)


if __name__ == '__main__':
    # for experiment_dir in sorted([x for x in RESULTS_DIR.iterdir() if x.is_dir()], key=lambda x: x.name[::-1]):
    #     analyze_experiment(experiment_dir)
    analyze_experiment(RESULTS_DIR / "malnettiny-GPS-ESLapPE")
    analyze_experiment(RESULTS_DIR / "composition-malnettiny-GPS-ESLapPE")