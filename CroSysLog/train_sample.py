import torch
import numpy as np
import random
from MetaDataset import MetaDataset
from Meta import MAML
from ray import tune
import ray
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import time

def set_seed(seed):
    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    from transformers import set_seed
    set_seed(seed)
def train_lstm(config):

    device = ray.get(device_ref)

    x_spt_tensor = ray.get(x_spt_tensor_ref)
    y_spt_tensor = ray.get(y_spt_tensor_ref)
    x_qry_tensor = ray.get(x_qry_tensor_ref)
    y_qry_tensor = ray.get(y_qry_tensor_ref)


    test_x_spt_tensor = ray.get(test_x_spt_tensor_ref)
    test_y_spt_tensor = ray.get(test_y_spt_tensor_ref)
    test_x_qry_tensor = ray.get(test_x_qry_tensor_ref)
    test_y_qry_tensor = ray.get(test_y_qry_tensor_ref)

    learner_config = [
        # ('rnn', [args.input_dim, args.hidden_dim, args.n_layers, args.n_way])
        ('LSTM', [config['input_size'], config['hidden_dim'], config['n_layers'], 2, config['dropout']])

    ]

    maml = MAML(config, learner_config).to(device)

    for epoch in range(config['epoch']):
        #print("epoch:", epoch)
        loss = maml(x_spt_tensor, y_spt_tensor, x_qry_tensor, y_qry_tensor)
        #print("Training loss:", loss)

        if epoch % 1 == 0:
            cache_f1 = [0 for _ in range(config['target_n_task'])]
            cache_precision = [0 for _ in range(config['target_n_task'])]
            cache_recall = [0 for _ in range(config['target_n_task'])]

            # Step 2: Capture the start time
            testing_start_time = time.time()
            for cache in range(config['target_n_task']):
                print("cache:", cache)
                cache_x_spt_tensor = test_x_spt_tensor[cache, :, :]
                cache_x_qry_tensor = test_x_qry_tensor[cache, :, :]
                cache_y_spt_tensor = test_y_spt_tensor[cache, :]
                cache_y_qry_tensor = test_y_qry_tensor[cache, :]

                tem_f1, tem_precision, tem_recall = maml.finetunning(cache_x_spt_tensor, cache_y_spt_tensor, cache_x_qry_tensor,
                                                   cache_y_qry_tensor)
                highest_cache_f1 = np.max(tem_f1)
                index_of_highest = np.argmax(tem_f1)

                tem_cache_precision = tem_precision[index_of_highest]
                tem_cache_recall = tem_recall[index_of_highest]

                cache_f1[cache] = highest_cache_f1
                cache_precision[cache] = tem_cache_precision
                cache_recall[cache] = tem_cache_recall

                #cache_ave_precision.append(precision)
                #cache_ave_recall.append(recall)

            testing_end_time = time.time()
            # Step 5: Compute the time difference
            testing_time = testing_end_time - testing_start_time
            ave_test_fi = np.mean(cache_f1)
            ave_test_precision = np.mean(cache_precision)
            ave_test_recall = np.mean(cache_recall)

            metrics_to_report = {f'ave_test_recall': ave_test_recall}
            metrics_to_report.update({'ave_test_precision': ave_test_precision})
            metrics_to_report.update({'ave_test_fi': ave_test_fi})
            metrics_to_report.update({'testing_time': testing_time})

            tune.report(**metrics_to_report)

if __name__ == '__main__':

    config = {
        "source_system": ["LIBERTY", "BGL"],
        "target_system": "SPIRIT",

        # Meta config
        "epoch": 150,
        "task_num": 1,
        "target_n_task": 20,
        "k_spt": 10000,
        "k_qry": 10000,
        "update_lr": tune.loguniform(2e-3, 1e-2),
        "meta_lr": tune.loguniform(9e-6, 9e-4),

        "window_size": 100,
        "update_step": 5,
        "update_step_test": 15,

        # Learner config
        "input_size": 768,
        "hidden_dim": tune.choice([32, 64, 128]),
        "n_layers": 1,
        "dropout": tune.uniform(0.1, 0.5),

    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = MetaDataset(n_task=config['task_num'], target_n_task=config['target_n_task'],
                     k_spt=config['k_spt'], k_query=config['k_qry'],
                     source_system=config['source_system'], target_system=config['target_system'],
                     window_size=config['window_size'])

    set_seed(42)

    all_x_spt_tensors = []
    all_y_spt_tensors = []
    all_x_qry_tensors = []
    all_y_qry_tensors = []

    for system_name in config["source_system"]:
        print("system:", system_name, flush=True)
        x_spt_tensor, y_spt_tensor, x_qry_tensor, y_qry_tensor = db.load_data_cache_source(system_name)

        all_x_spt_tensors.append(x_spt_tensor)
        all_y_spt_tensors.append(y_spt_tensor)
        all_x_qry_tensors.append(x_qry_tensor)
        all_y_qry_tensors.append(y_qry_tensor)

    x_spt_tensor= torch.cat(all_x_spt_tensors, dim=0)
    y_spt_tensor = torch.cat(all_y_spt_tensors, dim=0)
    x_qry_tensor = torch.cat(all_x_qry_tensors, dim=0)
    y_qry_tensor= torch.cat(all_y_qry_tensors, dim=0)
    y_spt_tensor = y_spt_tensor.clone().detach().long()
    y_qry_tensor = y_qry_tensor.clone().detach().long()

    test_x_spt_tensor, test_y_spt_tensor, test_x_qry_tensor, test_y_qry_tensor = db.load_data_cache_target()
    test_y_spt_tensor = test_y_spt_tensor.clone().detach().long()
    test_y_qry_tensor = test_y_qry_tensor.clone().detach().long()
    test_x_spt_tensor, test_y_spt_tensor, test_x_qry_tensor, test_y_qry_tensor = test_x_spt_tensor.to(
        device), \
        test_y_spt_tensor.to(device), test_x_qry_tensor.to(device), test_y_qry_tensor.to(device)


    x_spt_tensor, y_spt_tensor, x_qry_tensor, y_qry_tensor = x_spt_tensor.to(device), \
        y_spt_tensor.to(device), x_qry_tensor.to(device), y_qry_tensor.to(device)

    device_ref = ray.put(device)

    x_spt_tensor_ref = ray.put(x_spt_tensor)
    y_spt_tensor_ref = ray.put(y_spt_tensor)
    x_qry_tensor_ref = ray.put(x_qry_tensor)
    y_qry_tensor_ref = ray.put(y_qry_tensor)

    test_x_spt_tensor_ref = ray.put(test_x_spt_tensor)
    test_y_spt_tensor_ref = ray.put(test_y_spt_tensor)
    test_x_qry_tensor_ref = ray.put(test_x_qry_tensor)
    test_y_qry_tensor_ref = ray.put(test_y_qry_tensor)


    scheduler = ASHAScheduler(
        metric="ave_test_fi",
        mode="max",
        max_t=60,
        grace_period=10,
        reduction_factor=5
    )

    reporter = CLIReporter(
        metric_columns=["ave_test_recall", "ave_test_precision", "ave_test_fi", "testing_time", "training_iteration"],
        max_progress_rows=2
    )

    analysis = tune.run(
        train_lstm,
        resources_per_trial={"cpu": 32, "gpu": 1},
        config=config,
        num_samples=120,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='./baselineOut',
        keep_checkpoints_num=10,  # Keep only the best checkpoint.
        checkpoint_score_attr="ave_test_aucs"
    )

    best_trial = analysis.get_best_trial("ave_test_fi", "max", "last")
    print("Best trial config: ", analysis.get_best_config(metric="ave_test_fi", mode="max"))
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final training accuracy: {}".format(
        best_trial.last_result["ave_test_fi"]))
