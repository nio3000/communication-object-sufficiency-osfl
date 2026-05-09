import random
import torch
import math
import time

def reptile_meta_init(model_fn, clients, device, meta_steps=1000, meta_lr=1e-3,
                      inner_steps=5, inner_lr=1e-3, verbose=True):
    start_time = time.time()

    # 初始化元模型和临时模型
    meta_model = model_fn().to(device)
    temp_model = model_fn().to(device)

    # 关闭 BN 追踪
    for m in [meta_model, temp_model]:
        for sub_m in m.modules():
            if isinstance(sub_m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                sub_m.track_running_stats = False

    meta_state = {k: v.clone() for k, v in meta_model.state_dict().items()}
    client_indices = list(range(len(clients)))

    # 缓存每个 client 的迭代器
    client_iters = {i: iter(clients[i].train_loader) for i in range(len(clients))}

    for step in range(meta_steps):
        # 余弦退火学习率
        scheduled_lr = 0.5 * meta_lr * (1 + math.cos(math.pi * step / meta_steps))

        # 随机选择一个客户端
        c_idx = random.choice(client_indices)

        # 加载当前元状态到临时模型
        temp_model.load_state_dict(meta_state)
        temp_model.train()

        # 内部优化器 (SGD with momentum)
        optimizer = torch.optim.SGD(temp_model.parameters(), lr=inner_lr, momentum=0.9)

        # 内部更新
        for _ in range(inner_steps):
            try:
                x, y = next(client_iters[c_idx])
            except StopIteration:
                client_iters[c_idx] = iter(clients[c_idx].train_loader)
                x, y = next(client_iters[c_idx])

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # 确保标签为长整型且 shape 正确
            if y.dim() > 1:
                y = y.squeeze(1).long()
            else:
                y = y.long()

            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(temp_model(x), y)
            loss.backward()
            optimizer.step()

        # Reptile 更新（直接在 meta_state 上操作）
        trained_state = temp_model.state_dict()
        with torch.no_grad():
            for k in meta_state.keys():
                if meta_state[k].dtype.is_floating_point:
                    meta_state[k].add_(trained_state[k] - meta_state[k], alpha=scheduled_lr)

        if verbose and (step % 200 == 0 or step == meta_steps - 1):
            elapsed = time.time() - start_time
            print(f"   [{step+1:4d}/{meta_steps}] Meta-LR: {scheduled_lr:.6f} | 已耗时: {elapsed:.1f}s")

    # 清理缓存
    del temp_model, client_iters
    torch.cuda.empty_cache()
    return meta_state