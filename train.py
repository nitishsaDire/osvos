from Dataset import denormalize
from metrics import eval_iou, get_contour_accuracy
import torch
import matplotlib.pyplot as plt
import time
import gc

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def balance_CELoss(output, labels):
    # print(output.max(), output.min())
    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    sig_o = torch.sigmoid(output)

    vall = sig_o + 2 * (labels - 1) * sig_o + 1 - labels + 1e-7

    # print(vall.max(), vall.min())
    loss_matrix = -torch.log(vall)
    pos_loss = torch.sum(labels * loss_matrix)
    neg_loss = torch.sum((1.0 - labels) * loss_matrix)
    net_loss = num_labels_neg / num_total * pos_loss + num_labels_pos / num_total * neg_loss
    # print(net_loss,net_loss/output.shape[0])

    return net_loss / output.shape[0]


def train_model_online(unet, optimizer, dataloader, dataset_sizes, device, loadModel=False, num_epochs=200):
    since = time.time()
    epoch_losses = {}
    epoch_jaccs = {}
    epoch_cas = {}

    epoch_accuracies = {}
    cnn = unet.to(device)
    for k in ['train', 'val']:
        epoch_losses[k] = []
        epoch_accuracies[k] = []
        epoch_jaccs[k] = []
        epoch_cas[k] = []

    best_acc = 0.0
    # OLD_PATH = '/content/drive/MyDrive/OSVOS_davis_5-m2'
    # PATH = '/content/drive/MyDrive/OSVOS_davis_5-m2'
    OLD_PATH = '/content/drive/MyDrive/OSVOS_davis-m3'
    if loadModel == True:
        checkpoint = torch.load(OLD_PATH)
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cnn = cnn.to(device)
        for g in optimizer.param_groups:
            g['lr'] = 0.00005

    for epoch in range(num_epochs):
        epoch_b = time.time()

        # print(device)
        # print(torch.cuda.memory_summary(device=device, abbreviated=False)
        torch.cuda.empty_cache()
        gc.collect()

        unet = unet.to(device)
        unet.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        running_jacc = 0.0
        running_ca = 0.0

        # Iterate over data.
        count = 0
        it_begin = time.time()
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = cnn(inputs)
                jacc = eval_iou(labels, outputs[-1])
                ca = get_contour_accuracy(outputs[-1], labels)

                loss = balance_CELoss(outputs[-1], labels)

                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_jacc += jacc * inputs.size(0)
            running_ca += ca * inputs.size(0)

            running_corrects += 10
            if count % 20 == 0:
                time_elapsed = time.time() - it_begin

            count += 1

            # print(count)
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            epoch_jacc = running_jacc / dataset_sizes
            epoch_ca = running_ca / dataset_sizes
            if epoch % 50 == 0:
                print('Loss: {:.4f} ca: {:.4f} jaccard: {:.4f}'.format(
                    epoch_loss, epoch_ca, epoch_jacc))

        time_elapsed = time.time() - epoch_b

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return unet


def train_model(unet, optimizer, dataloader, dataset_sizes, device, loadModel=False, num_epochs=200):
    since = time.time()
    epoch_losses = {}
    epoch_jaccs = {}
    epoch_cas = {}

    epoch_accuracies = {}
    cnn = unet.to(device)
    for k in ['train', 'val']:
        epoch_losses[k] = []
        epoch_accuracies[k] = []
        epoch_jaccs[k] = []
        epoch_cas[k] = []

    best_acc = 0.0
    epoch = 0

    OLD_PATH = '/content/drive/MyDrive/OSVOS_davis-m3_2'
    PATH = '/content/drive/MyDrive/OSVOS_davis-m3_2'
    if loadModel == True:
        checkpoint = torch.load(OLD_PATH)
        cnn.load_state_dict(checkpoint['cnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        cnn = cnn.to(device)
        epoch_losses = checkpoint['epoch_losses']
        epoch_jaccs = checkpoint['epoch_jaccs']
        epoch_cas = checkpoint['epoch_cas']
        for g in optimizer.param_groups:
            g['lr'] = 0.0001

    for epoch in range(epoch, num_epochs):
        epoch_b = time.time()

        print(device)
        # print(torch.cuda.memory_summary(device=device, abbreviated=False)
        torch.cuda.empty_cache()
        gc.collect()

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            unet = unet.to(device)
            if phase == 'train':
                unet.train()  # Set model to training mode
            else:
                unet.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_jacc = 0.0
            running_ca = 0.0

            # Iterate over data.
            count = 0
            it_begin = time.time()
            for inputs, labels in dataloader[phase]:
                # print(labels.shape)
                # labels = labels.squeeze(1)
                inputs, labels = inputs.to(device), labels.to(device)

                if count % 40 == 0:
                    with torch.no_grad():
                        print(phase)
                        random_index = torch.randint(0, inputs.shape[0], (1,))[0]
                        print(random_index)
                        _, _ = plt.subplots(figsize=(6, 6))
                        im = inputs[random_index]
                        print(im.shape)
                        plt.imshow(denormalize(im.cpu().permute(1, 2, 0)))
                        plt.show()
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = cnn(inputs)
                    jacc = eval_iou(labels, outputs[-1])
                    ca = get_contour_accuracy(outputs[-1], labels)

                    loss = 0.0
                    for i in outputs:
                        loss += balance_CELoss(i, labels)
                    loss /= 6.0

                    if count % 40 == 0:
                        with torch.no_grad():
                            print(phase)
                            print("gt")
                            mask = labels[random_index]
                            _, _ = plt.subplots(figsize=(6, 6))
                            plt.imshow(denormalize(im.cpu().permute(1, 2, 0)))
                            plt.imshow(mask.squeeze().cpu().detach().numpy(), alpha=0.5)
                            plt.show()

                            print("pred")
                            if phase != 'val':
                                unet.eval()
                                outputs = cnn(inputs)
                            mask = outputs[-1][random_index]
                            mask_sig = torch.sigmoid(mask)
                            # o = cnn(torch.randn((2,3,480, 854)).cuda())


                            # _, _ = plt.subplots(figsize=(6, 6))
                            # plt.imshow(torch.ge(mask_sig, 0.5).float().squeeze().cpu().detach().numpy())
                            # plt.show()


                            _, _ = plt.subplots(figsize=(6, 6))
                            plt.imshow(denormalize(im.cpu().permute(1, 2, 0)))
                            plt.imshow(torch.ge(mask_sig, 0.5).float().squeeze().cpu().detach().numpy(), alpha=0.5)
                            plt.show()

                            if phase != 'val':
                                unet.train()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_jacc += jacc * inputs.size(0)
                running_ca += ca * inputs.size(0)

                running_corrects += 10
                if count % 20 == 0:
                    time_elapsed = time.time() - it_begin
                    print("IIterated over ", count, "LR=", get_lr(optimizer),
                          'Iteration Completed in {:.0f}m {:.0f}s'.format(
                              time_elapsed // 60, time_elapsed % 60), "loss", loss.item())

                count += 1

            print(count)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_jacc = running_jacc / dataset_sizes[phase]
            epoch_ca = running_ca / dataset_sizes[phase]

            epoch_losses[phase].append(epoch_loss)
            epoch_accuracies[phase].append(epoch_acc)
            epoch_jaccs[phase].append(epoch_jacc)
            epoch_cas[phase].append(epoch_ca)

            print('{} Loss: {:.4f} Acc: {:.4f} jaccard: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_jacc))

        torch.save({
            'epoch': epoch,
            'cnn_state_dict': cnn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'epoch_losses': epoch_losses,
            'epoch_jaccs': epoch_jaccs,
            'epoch_cas': epoch_cas
        }, PATH)

        time_elapsed = time.time() - epoch_b
        print('epoch completed in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        print()
        print(epoch_losses)
        print(epoch_jaccs)
        print(epoch_cas)
        print('-' * 30)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return unet
