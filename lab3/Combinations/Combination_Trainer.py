from Combinations.CaptionModel import CaptionModel
from dataset import get_loaders
from torch import nn
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
import json

def train_model(
        model: CaptionModel,
        model_name: str,
        num_epochs: int,
        data_dir = "data",
        output_dir = "SavedModels"
    ):

    device = next(iter(model.parameters())).device

    save_dir = os.path.join(output_dir, model_name)

    (train_loader, val_loader, test_loader), dataset = get_loaders(data_dir)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(f"runs/{model_name}")
    
    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valloss=float('inf')
    best_epoch = 0
    train_accs=[]
    val_accs=[]
    val_losses=[]
    train_losses=[]
    start=time.time()

    for epoch in tqdm(range(num_epochs)):
        train_loss=0
        train_correct=0
        train_total=0

        model.train()

        for (data,label) in tqdm(train_loader, leave=False):
            data=data.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            
            
            #We use the feature array from the CNN as input for the RNN


            # teacher forcing
            inputs = label[:, :-1]
            targets = label[:, 1:]

            output = model(data, inputs)
            output = output[:, 1:, :]
            
            loss=criterion(output.reshape(-1,vocab_size),targets.reshape(-1))
            
            loss.backward()
            optimizer.step()
            pred=output.argmax(dim=2)

            train_correct+=(pred==targets).sum().item()
            train_total+=label.size(0)
            train_loss += loss.item()
        
        train_loss/=len(train_loader)
        train_acc=train_correct/train_total
        train_accs.append(train_acc)
        train_losses.append(train_loss)

        valloss = 0
        val_correct = 0
        val_total = 0
        # Caption sample: we capture one per epoch (from the first image of the first val batch)
        sample_gen_caption = None
        sample_ref_caption = None
        first_val_batch = True

        model.eval()
        with torch.no_grad():
            for (data, label) in tqdm(val_loader, leave=False):
                data = data.to(device)
                label = label.to(device)
                inputs = label[:, :-1]
                targets = label[:, 1:]
                output = model(data, inputs)
                output = output[:, 1:, :]

                loss = criterion(output.reshape(-1, vocab_size), targets.reshape(-1))
                pred = output.argmax(dim=2)

                val_correct += (pred == targets).sum().item()
                val_total += label.size(0)
                valloss += loss.item()

                # --- Group member's approach: generate caption inside the val loop ---
                # We only do this for the FIRST image of the FIRST batch (once per epoch)
                if first_val_batch:
                    first_val_batch = False
                    # Generate an actual caption for the first validation image
                    sample_gen_caption = model.generate_caption(data[0:1], dataset.vocab)
                    # Get the reference caption from the label
                    ref_words = []
                    for idx in label[0]:
                        word = dataset.vocab.itos[idx.item()]
                        if word == "<EOS>": break
                        if word not in ["<PAD>", "<SOS>"]: ref_words.append(word)
                    sample_ref_caption = " ".join(ref_words)

        val_acc = val_correct / val_total
        val_accs.append(val_acc)
        valloss /= len(val_loader)
        val_losses.append(valloss)

        if valloss < best_valloss:
            best_valloss = valloss
            best_epoch = epoch + 1
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, "best_comb.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, "last_comb.pth"))

        print(f"[{epoch+1} / {num_epochs}] Train loss: {train_loss:.4f} | Val loss: {valloss:.4f}")
        if sample_gen_caption:
            print(f"  Sample caption → GEN: {sample_gen_caption}")
            print(f"  Sample caption → REF: {sample_ref_caption}")

        # --- TensorBoard Scalars ---
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", valloss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # --- TensorBoard Text: caption evolution per epoch ---
        if sample_gen_caption:
            writer.add_text(
                "Captions/Evolution",
                f"**Epoch {epoch+1}**\n\n**Generated:** {sample_gen_caption}\n\n**Reference:** {sample_ref_caption}",
                epoch
            )

    traintime=time.time()-start
    print("traintime: ", traintime)
   
    if True:
        data_meta={"traintime":traintime,
                    "best_epoch": best_epoch,
                    "best_valloss":best_valloss,
                    "epochs":num_epochs,
                    "lr":lr,
                    "val_losses":val_losses,
                    "train_losses":train_losses,
                    "val_acc":val_accs,
                    "train_accs":train_accs}
        path_meta=os.path.join(save_dir,"Metadata.json")
        with open(path_meta, "w") as f:
            json.dump(data_meta, f)
            print("Meta-data saved")
            
    writer.close()