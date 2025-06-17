import torch
import gc
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from util import freeze_layers, AverageMeter
from model import EventDetector_clstm_lr
from dataloader import GolfDB, Normalize, ToTensor



def main():
    # GPU 최적화
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 하이퍼파라미터
    n_cpu = 0
    seq_length = 64
    k = 10
    bs = 6
    val_bs = 12
    accum_steps = 1
    epochs = 30
    lr = 0.001
    min_lr = 1e-6
    weight_decay = 0.01
    warmup_epochs = 3
    warmup_factor = 0.1
    patience = 7
    early_stop_delta = 1e-4
    val_frequency = 2
    log_interval = 80

    best_val_loss = float('inf')
    best_accuracy = 0.0

    writer = SummaryWriter('runs/clstm_fast_accurate')

    # 모델 초기화
    print("모델 초기화 중...")
    model = EventDetector_clstm_lr(n_conv=seq_length, num_classes=9).cuda()
    freeze_layers(k, model)

    # 손실함수, 옵티마이저, 스케일러, 스케줄러
    weights = torch.FloatTensor([1/8] * 8 + [1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=min_lr
    )

    # LR warm-up 함수
    def adjust_lr_warmup(opt, epoch):
        if epoch < warmup_epochs:
            scale = warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
            lr_ = lr * scale
            for g in opt.param_groups:
                g['lr'] = lr_
            return lr_
        return opt.param_groups[0]['lr']

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=7, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.delta = delta

        def __call__(self, val_loss):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} / {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=early_stop_delta)

    # 데이터 로더
    transform = transforms.Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("데이터 로더 준비 중...")
    train_dataset = GolfDB(
        data_file='data/train_split.pkl',
        vid_dir='data/videos_160',
        seq_length=seq_length,
        transform=transform,
        train=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True, drop_last=True,
        num_workers=n_cpu, pin_memory=True, persistent_workers=False
    )

    # 검증도 train=True 모드로 고정 길이 반환
    val_dataset = GolfDB(
        data_file='data/val_split.pkl',
        vid_dir='data/videos_160',
        seq_length=seq_length,
        transform=transform,
        train=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=val_bs, shuffle=False, drop_last=False,
        num_workers=n_cpu, pin_memory=True, persistent_workers=False
    )

    train_losses, val_losses, val_accuracies, val_precisions, val_recalls = [], [], [], [], []

    print(f"\n========== 학습 시작 ==========")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        # LR 조정
        if epoch <= warmup_epochs:
            current_lr = adjust_lr_warmup(optimizer, epoch - 1)
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # --- Train ---
        model.train()
        train_meter = AverageMeter()
        for step, sample in enumerate(train_loader, 1):
            imgs = sample['images'].cuda(non_blocking=True)
            labs = sample['labels'].view(-1).cuda(non_blocking=True)

            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labs) / accum_steps
            scaler.scale(loss).backward()

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_meter.update(loss.item() * accum_steps, imgs.size(0))
            if step % log_interval == 0:
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{len(train_loader)}] "
                      f"Loss: {train_meter.val:.4f} (Avg: {train_meter.avg:.4f}) LR: {current_lr:.6f}")

        train_losses.append(train_meter.avg)

        # --- Validation ---
        if epoch % val_frequency == 0 or epoch in {1, epochs}:
            model.eval()
            val_meter = AverageMeter()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for sample in val_loader:
                    imgs = sample['images'].cuda(non_blocking=True)
                    labs = sample['labels'].view(-1).cuda(non_blocking=True)

                    with autocast():
                        logits = model(imgs)
                        v_loss = criterion(logits, labs)
                    val_meter.update(v_loss.item(), imgs.size(0))
                    all_preds.extend(logits.argmax(1).cpu().numpy())
                    all_labels.extend(labs.cpu().numpy())

            # 지표 계산
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            val_losses.append(val_meter.avg)
            val_accuracies.append(acc)
            val_precisions.append(prec)
            val_recalls.append(rec)

            # TensorBoard 로깅
            writer.add_scalars('Loss', {'train': train_meter.avg, 'val': val_meter.avg}, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            writer.add_scalar('Accuracy', acc, epoch)
            writer.add_scalar('Precision', prec, epoch)
            writer.add_scalar('Recall', rec, epoch)
            writer.add_scalar('F1_macro', f1, epoch)

            elapsed = time.time() - epoch_start
            print(f"\nEpoch {epoch} | Time: {elapsed:.1f}s")
            print(f"Train Loss: {train_meter.avg:.4f} | Val Loss: {val_meter.avg:.4f}")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1_macro: {f1:.4f}\n")

            if val_meter.avg < best_val_loss:
                best_val_loss = val_meter.avg
                best_accuracy = acc
                torch.save(model.state_dict(), 'models/best_conv_lr.pth')
                print("✓ Best model saved!")

            early_stopping(val_meter.avg)
            if early_stopping.early_stop:
                print("조기 종료!")
                break

        else:
            writer.add_scalar('Loss/train', train_meter.avg, epoch)
            writer.add_scalar('LR', current_lr, epoch)
            print(f"Epoch {epoch} | Train Loss: {train_meter.avg:.4f} | LR: {current_lr:.6f}")

        if epoch > warmup_epochs:
            scheduler.step()

    # ========================================== 최종 저장 ==========================================
    writer.close()
    torch.save(model.state_dict(), 'models/final_conv_lr.pth')
    print("\n모든 작업이 완료되었습니다!")
    if torch.cuda.is_available():
        print(f"GPU 메모리 사용량: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

if __name__ == "__main__":
    main()
