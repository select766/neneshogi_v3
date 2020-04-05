import argparse
import os
from datetime import datetime, timedelta

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from neneshogi import models
from neneshogi.packed_sfen_dataset import PackedSfenDataset
from neneshogi.train_manager import TrainManager
from neneshogi.util import yaml_load


def setup_trainer(model_config, train_config, device):
    model_class = getattr(models, model_config["model"])
    model = model_class(board_shape=(119, 9, 9), move_dim=27 * 9 * 9, **model_config.get("kwargs", {}))
    model.to(device)
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), **train_config["optimizer"].get("kwargs", {}))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **train_config["lr_scheduler"].get("kwargs", {}))
    return model, criterion_policy, criterion_value, optimizer, scheduler


def setup_data_loader(train_config):
    train_set = PackedSfenDataset(**train_config["dataset"]["train"]["data"])
    train_loader = torch.utils.data.DataLoader(train_set,
                                               shuffle=False, num_workers=0,
                                               **train_config["dataset"]["train"]["loader"])

    val_set = PackedSfenDataset(**train_config["dataset"]["val"]["data"])
    val_loader = torch.utils.data.DataLoader(val_set,
                                             shuffle=False, num_workers=0,
                                             **train_config["dataset"]["val"]["loader"])
    return train_loader, val_loader


def forever_iterator(iterator):
    while True:
        for item in iterator:
            yield item


def dump_status(train_dir, train_manager, model, optimizer, lr_scheduler):
    save_dir = os.path.join(train_dir, "checkpoints", f"train_{train_manager.trained_samples:010d}")
    os.makedirs(save_dir)
    # モデルの保存
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(save_dir, "model.pt"))
    # 再開用状態保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "train_manager": train_manager,
    }, os.path.join(save_dir, "resume.bin"))


def resume_status(train_dir, resume_dir, device, model, optimizer, lr_scheduler):
    statuses = torch.load(os.path.join(resume_dir, "resume.bin"), map_location=str(device))
    model.load_state_dict(statuses["model_state_dict"])
    optimizer.load_state_dict(statuses["optimizer_state_dict"])
    lr_scheduler.load_state_dict(statuses["lr_scheduler_state_dict"])
    return {"train_manager": statuses["train_manager"]}


def create_stop_file(train_dir):
    with open(os.path.join(train_dir, "deletetostop.tmp"), "w") as f:
        pass


def check_stop_file(train_dir):
    return os.path.exists(os.path.join(train_dir, "deletetostop.tmp"))


def train_loop(train_manager: TrainManager, train_config, device, model, criterion_policy, criterion_value, optimizer,
               lr_scheduler,
               train_loader, val_loader, summary_writer, train_dir):
    train_forever_iterator = iter(forever_iterator(train_loader))
    next_dump_time = datetime.now() + timedelta(hours=1)
    while check_stop_file(train_dir):
        next_action_info = train_manager.get_next_action()
        if next_action_info["action"] == "train":
            model.train()
            data = train_forever_iterator.__next__()
            inputs = data['board'].to(device)
            move_index = data['move_index'].to(device)
            game_result = data['game_result'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output_policy, output_value = model(inputs)
            loss_policy = criterion_policy(output_policy, move_index)
            loss_value = criterion_value(output_value, game_result)
            loss = loss_policy * train_config["loss"]["policy"] + loss_value * train_config["loss"]["value"]
            loss.backward()
            optimizer.step()
            # print statistics
            summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], train_manager.trained_samples)
            summary_writer.add_scalar("train/loss", loss.item(), train_manager.trained_samples)
            summary_writer.add_scalar("train/loss_policy", loss_policy.item(), train_manager.trained_samples)
            summary_writer.add_scalar("train/loss_value", loss_value.item(), train_manager.trained_samples)
            train_manager.put_train_result()

            if datetime.now() >= next_dump_time:
                # dump_status(train_dir=train_dir, train_manager=train_manager, model=model, optimizer=optimizer,
                #             lr_scheduler=lr_scheduler)
                next_dump_time += timedelta(hours=1)

        elif next_action_info["action"] == "val":
            model.eval()

            running_loss_times = 0
            running_loss = 0.0
            running_loss_policy = 0.0
            running_loss_value = 0.0
            policy_correct_count = 0
            value_correct_count = 0
            n_val_items = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs = data['board'].to(device)
                    move_index = data['move_index'].to(device)
                    game_result = data['game_result'].to(device)
                    output_policy, output_value = model(inputs)
                    loss_policy = criterion_policy(output_policy, move_index)
                    loss_value = criterion_value(output_value, game_result)
                    loss = loss_policy * train_config["loss"]["policy"] + loss_value * train_config["loss"]["value"]
                    policy_pred_label = torch.argmax(output_policy, 1)
                    policy_correct_count += torch.sum(torch.eq(move_index, policy_pred_label)).item()
                    value_pred_label = torch.argmax(output_value, 1)
                    value_correct_count += torch.sum(torch.eq(game_result, value_pred_label)).item()
                    n_val_items += policy_pred_label.shape[0]
                    running_loss += loss.item()
                    running_loss_policy += loss_policy.item()
                    running_loss_value += loss_value.item()
                    running_loss_times += 1
            avg_loss = running_loss / running_loss_times
            avg_loss_policy = running_loss_policy / running_loss_times
            avg_loss_value = running_loss_value / running_loss_times
            avg_top1_accuracy_policy = policy_correct_count / n_val_items
            avg_top1_accuracy_value = value_correct_count / n_val_items
            summary_writer.add_scalar("val/avg_loss", avg_loss, train_manager.trained_samples)
            summary_writer.add_scalar("val/avg_loss_policy", avg_loss_policy, train_manager.trained_samples)
            summary_writer.add_scalar("val/avg_loss_value", avg_loss_value, train_manager.trained_samples)
            summary_writer.add_scalar("val/avg_top1_accuracy_policy", avg_top1_accuracy_policy,
                                      train_manager.trained_samples)
            summary_writer.add_scalar("val/avg_top1_accuracy_value", avg_top1_accuracy_value,
                                      train_manager.trained_samples)
            lr_scheduler.step(avg_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            train_manager.put_val_result(new_lr=new_lr)
            dump_status(train_dir=train_dir, train_manager=train_manager, model=model, optimizer=optimizer,
                        lr_scheduler=lr_scheduler)
        else:
            break
    print("quit: ", train_manager.quit_reason)
    dump_status(train_dir=train_dir, train_manager=train_manager, model=model, optimizer=optimizer,
                lr_scheduler=lr_scheduler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train_dir = args.train_dir
    model_config = yaml_load(os.path.join(train_dir, "model.yaml"))
    train_config = yaml_load(os.path.join(train_dir, "train.yaml"))
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    model, criterion_policy, criterion_value, optimizer, lr_scheduler = setup_trainer(model_config, train_config,
                                                                                      device)
    train_loader, val_loader = setup_data_loader(train_config)
    summary_writer = SummaryWriter(os.path.join(train_dir, "log"))
    if args.resume:
        resumed_status = resume_status(train_dir, args.resume, device=device, model=model, optimizer=optimizer,
                                       lr_scheduler=lr_scheduler)
        train_manager = resumed_status["train_manager"]
    else:
        train_manager = TrainManager(**train_config["manager"])
    create_stop_file(train_dir)
    train_loop(train_manager=train_manager, train_config=train_config, device=device, model=model,
               criterion_policy=criterion_policy,
               criterion_value=criterion_value, optimizer=optimizer, lr_scheduler=lr_scheduler,
               train_loader=train_loader,
               val_loader=val_loader, summary_writer=summary_writer, train_dir=train_dir)


if __name__ == '__main__':
    main()
