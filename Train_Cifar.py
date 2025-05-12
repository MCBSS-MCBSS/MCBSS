import torch.backends.cudnn as cudnn
import os
import argparse
import random
import torch
import torchvision.transforms as transforms
from model.ResNet import resnet18
import numpy as np
import torch.nn.functional as F
from opt import k_barycenter, label_reg, label_propagation_analysis_class_change
import torch.optim as optim
from scipy.spatial.distance import cdist
from data.data_augment import CIFAR10Policy
from torch.cuda.amp import autocast #, GradScaler
import time
import torch.optim.lr_scheduler as lr_scheduler
from data.imbalance_cifar import *
from utils import util


def parse_args():
    parser = argparse.ArgumentParser(description='Imbalance Noisy CIFAR Training')
    parser.add_argument('--batch_size', default=128, type=int) 
    parser.add_argument('--lr', '--learning_rate', default=0.02, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--seed', default=2025)
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--lambd', default=10, type=float)
    parser.add_argument('--barycenter_number', default=1, type=int)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--Nb', type=int, default=128, help='number of tracked bathches')
    parser.add_argument('--alpha', type=float, default=1, help='class invariance coefficient')
    parser.add_argument('--last', action='store_true', help='choose model from {semi_supervised_train,semi_supervised_train^last}')
    parser.add_argument('--hard_label', type=bool, default=False)
    parser.add_argument('--T', type=float, default=1)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--output_file', default='test_accuracy.txt', type=str)
    parser.add_argument('--closeset_ratio', type=float, default=0.2)
    parser.add_argument('--noise_type', type=str, default='unif')
    parser.add_argument('--imbalance', type=bool, default=True)
    parser.add_argument('--imb_factor', type=float, default=0.05)
    parser.add_argument('--resample_weighting', default=0.0, type=float,help='weighted for sampling probability (q(1,k))')
    parser.add_argument('--label_weighting', default=1.0, type=float, help='weighted for Loss')
    parser.add_argument('--contrast_weight', default=4,type=int,help='Mixture Consistency  Weights')   
    return parser.parse_args()

def get_transform(args):
    if args.dataset=='cifar10':       
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_strong_10 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
            
        labeled_transform = [transform_train, transform_train] 
        unlabeled_transform = [transform_train, transform_strong_10] 


    elif args.dataset=='cifar100':
       
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)),
        ])
        
        transform_strong_100 = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        labeled_transform = [transform_train, transform_train] 
        unlabeled_transform = [transform_train, transform_strong_100] 

    return transform_test, labeled_transform, unlabeled_transform

def get_dataset(args,transform_train,transform_test):
    if args.dataset=='cifar10':
        num_class = 10
        warmup =  50
        CIFAR_noisy_train = CIFAR10_im(root="./data/cifar10", mode='train', meta=False, num_meta=0,
                                    corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=transform_train, selected=[],
                                    download=True,  imblance=args.imbalance,
                                    imb_factor=args.imb_factor)

        CIFAR_test = CIFAR10_im(root="./data/cifar10", mode='test', meta=False, num_meta=0,
                                corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=transform_test, selected=[],
                                download=True,  imblance=args.imbalance,
                                imb_factor=args.imb_factor)

    elif args.dataset=='cifar100':
        num_class = 100
        warmup =  50
        CIFAR_noisy_train = CIFAR100_im(root="./data/cifar100", mode='train', meta=False, num_meta=0,
                                    corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=transform_train, selected=[],
                                    download=True,  imblance=args.imbalance,
                                    imb_factor=args.imb_factor)

        CIFAR_test = CIFAR100_im(root="./data/cifar100", mode='test', meta=False, num_meta=0,
                                corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=transform_test, selected=[],
                                download=True,  imblance=args.imbalance,
                                imb_factor=args.imb_factor)

    return num_class, warmup, CIFAR_noisy_train, CIFAR_test

def create_model(num_class):
    model = resnet18(num_classes = num_class, pretrained=False)  
    model = model.cuda()
    return model

def adjust_learning_rate(args,optimizer, epoch, scheduler, warmup):
    if epoch < warmup:
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr  
    else:
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate updated by scheduler at epoch {epoch}: {current_lr}")        

def SimSiamLoss(p, z, version='simplified'):  
    z = z.detach()  

    if version == 'original':
        p = F.normalize(p, dim=1)  
        z = F.normalize(z, dim=1)  
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified': 
        return - F.cosine_similarity(p, z, dim=-1).mean()
    else:
        raise Exception

def rebalanced_warmup(args,warmup,num_class,train_cls_num_list, train_loader, weighted_train_loader, model, optimizer,epoch):
    alpha = 1 - (float(epoch) / float(warmup)) ** 2
    weighted_train_loader = iter(weighted_train_loader)
    for batch_idx, (inputs1, inputs2, targets, sample_id) in enumerate(train_loader):
        model.train()
        inputs1, inputs2, targets = inputs1.cuda(), inputs2.cuda(), targets.cuda()        
        optimizer.zero_grad()

        per_cls_weights = 1.0 / (np.array(train_cls_num_list) ** args.label_weighting)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * train_cls_num_list
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()        

        input_org_1 = inputs1.cpu()
        input_org_2 = inputs2.cpu()
        target_org = targets.cpu()

        try:
            input_invs_1, input_invs_2, target_invs, sample_invs_id = next(weighted_train_loader)
        except:
            weighted_train_loader = iter(weighted_train_loader)
            input_invs_1, input_invs_2, target_invs, sample_invs_id = next(weighted_train_loader)


        one_hot_org = torch.zeros(target_org.size(0), num_class).scatter_(1, target_org.view(-1, 1), 1)
        one_hot_org_w = per_cls_weights.cpu() * one_hot_org
        one_hot_invs = torch.zeros(target_invs.size(0), num_class).scatter_(1, target_invs.view(-1, 1), 1)
        one_hot_invs = one_hot_invs[:one_hot_org.size()[0]]
        one_hot_invs_w = per_cls_weights.cpu() * one_hot_invs

        input_org_1 = input_org_1.cuda()
        input_org_2 = input_org_2.cuda()
        input_invs_1 = input_invs_1.cuda()
        input_invs_2 = input_invs_2.cuda()

        one_hot_org = one_hot_org.cuda()
        one_hot_org_w = one_hot_org_w.cuda()
        one_hot_invs = one_hot_invs.cuda()
        one_hot_invs_w = one_hot_invs_w.cuda()


        mix_x, cut_x, mixup_y, mixcut_y, mixup_y_w, cutmix_y_w = util.GLMC_mixed(org1=input_org_1, org2=input_org_2,
                                                                                invs1=input_invs_1,
                                                                                invs2=input_invs_2,
                                                                                label_org=one_hot_org,
                                                                                label_invs=one_hot_invs,
                                                                                label_org_w=one_hot_org_w,
                                                                                label_invs_w=one_hot_invs_w)


        output_1, output_cb_1, z1, p1 = model(mix_x, type='three')
        output_2, output_cb_2, z2, p2 = model(cut_x, type='three')
        contrastive_loss = SimSiamLoss(p1, z2) + SimSiamLoss(p2, z1)

        loss_mix = -torch.mean(torch.sum(F.log_softmax(output_1, dim=1) * mixup_y, dim=1)) 
        loss_cut = -torch.mean(torch.sum(F.log_softmax(output_2, dim=1) * mixcut_y, dim=1)) 
        loss_mix_w = -torch.mean(torch.sum(F.log_softmax(output_cb_1, dim=1) * mixup_y_w, dim=1)) 
        loss_cut_w = -torch.mean(torch.sum(F.log_softmax(output_cb_2, dim=1) * cutmix_y_w, dim=1)) 

        balance_loss = loss_mix + loss_cut
        rebalance_loss = loss_mix_w + loss_cut_w

        loss_imb = alpha * balance_loss + (1 - alpha) * rebalance_loss + args.contrast_weight * contrastive_loss

        loss_imb.backward()
        optimizer.step()

def test(test_loader, model, epoch, output_file='test_accuracy.txt'):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, type='one')
            loss = F.cross_entropy(outputs, targets)           
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    print("test accuracy:",acc)
    
    with open(output_file, 'a') as f:
        f.write(f'Epoch {epoch}: Test Accuracy: {acc:.2f}%\n')

    return acc

def train(args, train_dataset, train_loader, test_loader, model, optimizer, epoch, num_class, labeled_transform, unlabeled_transform, class_transition_matrix, label_bank, distri, label_count):
    model.eval()
    train_feature = []
    train_label = np.array([])
    train_sample_id = np.array([])
    
    for batch_idx, (inputs1, inputs2, targets, sample_id) in enumerate(train_loader):
        label = targets.numpy()
        sample_id = sample_id.numpy()
        inputs1, targets = inputs1.cuda(), targets.cuda()
        feature = model(inputs1, type='two')
        feature = feature.detach().cpu().numpy()       
        feature = feature.tolist()        
        train_feature+=feature
        train_label = np.concatenate((train_label, label),axis = 0)
        train_sample_id = np.concatenate((train_sample_id, sample_id),axis = 0)
        # 清理内存
        del inputs1, inputs2, targets, feature
    train_feature = np.array(train_feature)
    
    barycenter = []
    barycenter_class = []
    for class_id in range(num_class):
        sample = np.where(train_label==class_id)
        sample_feature = train_feature[sample]
        center = k_barycenter(sample_feature.transpose(),args.barycenter_number, args.lambd)
        barycenter_class+=[class_id for i in range(args.barycenter_number)]
        barycenter+=center.transpose().tolist()
    
    barycenter = np.array(barycenter)
    barycenter_weight = np.ones(args.barycenter_number*num_class)/(args.barycenter_number*num_class)
    train_sample_weight = np.ones(len(train_feature))/len(train_feature)
    
    cost_matrix = cdist(barycenter, train_feature, metric='euclidean')       
    opt_result = label_reg(barycenter_weight, train_sample_weight, cost_matrix, barycenter_class, num_class, args.lambd)
    probability, label_propagation = label_propagation_analysis_class_change(opt_result, barycenter_class, num_class, train_label)
    EM_feature = train_feature[np.where(train_label==label_propagation)]
    EM_label = train_label[np.where(train_label==label_propagation)]
    EM_labeled_sample = train_sample_id[np.where(train_label==label_propagation)]
    train_dataset.calculate_precision_recall(EM_labeled_sample, epoch, args.output_file)  # 高精度低召回率
    EM_barycenter = []
    barycenter_class = []
    for class_id in range(num_class):
        sample = np.where(EM_label==class_id)
        sample_feature = EM_feature[sample]
        center = k_barycenter(sample_feature.transpose(),args.barycenter_number, args.lambd)
        barycenter_class+=[class_id for i in range(args.barycenter_number)]
        EM_barycenter+=center.transpose().tolist()
    
    EM_barycenter = np.array(EM_barycenter)
    
    cost_matrix = cdist(EM_barycenter, train_feature, metric='euclidean')       
    opt_result = label_reg(barycenter_weight, train_sample_weight, cost_matrix, barycenter_class, num_class, args.lambd)
    probability, label_propagation = label_propagation_analysis_class_change(opt_result, barycenter_class, num_class, train_label)

    # get clean barycenter
    labeled_sample = train_sample_id[np.where(train_label==label_propagation)]
    unlabeled_sample = train_sample_id[np.where(train_label!=label_propagation)]    
    # train_dataset.calculate_precision_recall(labeled_sample, epoch, args.output_file) 
    if args.dataset=='cifar10':
        labeled_dataset = CIFAR10_im(root="./data/cifar10", mode='labeled', meta=False, num_meta=0,
                                    corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=labeled_transform, selected=labeled_sample,
                                    download=True, imblance=args.imbalance,
                                    imb_factor=args.imb_factor)

        unlabeled_dataset = CIFAR10_im(root="./data/cifar10", mode='unlabeled', meta=False, num_meta=0,
                                corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=unlabeled_transform, selected=unlabeled_sample,
                                download=True, imblance=args.imbalance,
                                imb_factor=args.imb_factor)  
    elif args.dataset=='cifar100':
        labeled_dataset = CIFAR100_im(root="./data/cifar100", mode='labeled', meta=False, num_meta=0,
                                    corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=labeled_transform, selected=labeled_sample,
                                    download=True, imblance=args.imbalance,
                                    imb_factor=args.imb_factor)

        unlabeled_dataset = CIFAR100_im(root="./data/cifar100", mode='unlabeled', meta=False, num_meta=0,
                                corruption_prob=args.closeset_ratio, corruption_type=args.noise_type, transform=unlabeled_transform, selected=unlabeled_sample,
                                download=True, imblance=args.imbalance,
                                imb_factor=args.imb_factor)  
    
    labeled_trainloader = torch.utils.data.DataLoader(
    labeled_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True)  
    
    unlabeled_trainloader = torch.utils.data.DataLoader(
    unlabeled_dataset, args.batch_size, shuffle=True, num_workers=0, drop_last=True) 

    del train_feature, train_label, train_sample_id, barycenter, barycenter_weight, train_sample_weight, cost_matrix
    torch.cuda.empty_cache()

    semi_supervised_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, class_transition_matrix, label_bank, distri, label_count)

def semi_supervised_train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, epoch, class_transition_matrix, label_bank, distri, label_count):       
    num_classes = class_transition_matrix.size(0)
    train_model = model
    lambda_u = args.ulb_loss_ratio
    use_hard_label = args.hard_label
    
    train_model.train()              
    N = args.Nb  
    alpha = args.alpha  # 1
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    
    num_iter1 = (len(labeled_trainloader.dataset)//args.batch_size)    
    num_iter2 = (len(unlabeled_trainloader.dataset)//args.batch_size)    
    num_iter = max(num_iter1, num_iter2)

    for batch_idx in range(num_iter):    
        try:
            inputs_x, inputs_x2, labels_x, ground_truth_lb = next(labeled_train_iter)  
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, inputs_x2, labels_x, ground_truth_lb = next(labeled_train_iter)                       

        try:
            inputs_u, inputs_u2, index_u, labels_u, ground_truth_ulb = next(unlabeled_train_iter) 
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2, index_u, labels_u, ground_truth_ulb = next(unlabeled_train_iter)            

        labels_x = labels_x.long()              
        num_lb = inputs_x.shape[0]  
        num_ulb = inputs_u.shape[0]  
        assert num_ulb == inputs_u2.shape[0]


        inputs_x,  inputs_u,  inputs_u2 = inputs_x.cuda(), inputs_u.cuda(), inputs_u2.cuda()
        labels_x = labels_x.cuda()
        
        inputs = torch.cat((inputs_x, inputs_u, inputs_u2))  
        with torch.cuda.amp.autocast(enabled=False):  
            logits = train_model(inputs, type='one')            
            logits_inputs_x = logits[:num_lb] 
            logits_inputs_u, logits_inputs_u2 = logits[num_lb:].chunk(2) 

            pseudo_label = torch.softmax(logits_inputs_u, dim=-1) 
            max_probs, max_idx = torch.max(pseudo_label, dim=-1) 
            
            if not args.last:  
                for i in range(len(index_u)): 
                    if not index_u[i].cpu().item() in label_bank.keys():
                        label_bank[index_u[i].cpu().item()] = max_idx[i].cpu().item()
                    else:
                        if label_bank[index_u[i].cpu().item()] != max_idx[i].cpu().item():
                            class_transition_matrix[label_bank[index_u[i].cpu().item()], max_idx[i].cpu().item(), label_count] += 1
                            label_bank[index_u[i].cpu().item()] = max_idx[i].cpu().item()
                
                distri[label_count] = pseudo_label.detach().mean(0)
                label_count = (label_count + 1) % N
                class_transition_matrix[:, :, label_count] = torch.zeros(num_classes, num_classes)

            T = args.T
            p_cutoff = args.p_cutoff
            
            del logits
            
            sup_loss = ce_loss(logits_inputs_x, labels_x, reduction='mean') 

            CTT_tmp = torch.mean(class_transition_matrix, dim=2)  

            diag = torch.diag(torch.ones(num_classes, num_classes) * (alpha / (num_classes - 1))) 
            a_diag = torch.diag_embed(diag) 

            H = CTT_tmp / (CTT_tmp.abs().sum(1, keepdim=True) + 1e-16)
            H = H + a_diag  

            H_prime = H.cuda() / torch.mean(distri, dim=0)  
            
            unsup_loss, mask = consistency_loss_prg(
                logits_inputs_u,  
                logits_inputs_u2,  
                H_prime,  
                label_bank, 
                index_u, 
                args.last,                                                                  
                'ce', T, p_cutoff,
                use_hard_labels=args.hard_label
            )

            total_loss = sup_loss + lambda_u * unsup_loss


            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    if use_hard_labels:
        smoothing = 0.2
        return F.cross_entropy(logits, targets, reduction=reduction, label_smoothing=smoothing)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets*log_pred, dim=1)
        return nll_loss

def consistency_loss_prg(logits_w, logits_s, H_prime, label_bank, idx, last=False, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w/T, dim=-1) 
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float() 
        
     
        if last:
            for i in range(pseudo_label.size(0)):
                if idx[i].cpu().item() in label_bank.keys():
                    pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * H_prime[label_bank[idx[i].cpu().item()],:])  
        else:
            for i in range(pseudo_label.size(0)):
                pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * H_prime[max_idx[i],:])


     
        if use_hard_labels: 
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else: 
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def normalize_d(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']  
    loss = checkpoint['loss']  

    print(f"Checkpoint loaded from epoch {start_epoch}.")
    return model, optimizer, scheduler, start_epoch

def main():
    start_time = time.time()
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) 
    transform_test, labeled_transform, unlabeled_transform = get_transform(args)
    num_class, warmup, CIFAR_noisy_train, CIFAR_test = get_dataset(args,labeled_transform, transform_test)
    train_loader = torch.utils.data.DataLoader(CIFAR_noisy_train, args.batch_size, shuffle=True, num_workers=2)
    cls_num_list = [0] * num_class
    for label in CIFAR_noisy_train.train_labels:
        cls_num_list[label] += 1 
    train_cls_num_list = np.array(cls_num_list)
    cls_weight = 1.0 / (np.array(cls_num_list) ** args.resample_weighting)
    cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)  
    assert all(0 <= label < num_class for label in CIFAR_noisy_train.train_labels), "Label out of range!"
    samples_weight = np.array([cls_weight[t] for t in CIFAR_noisy_train.train_labels])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
    weighted_train_loader = torch.utils.data.DataLoader(CIFAR_noisy_train, batch_size=args.batch_size,num_workers=2, persistent_workers=True,pin_memory=True,sampler=weighted_sampler)
    test_loader = torch.utils.data.DataLoader(CIFAR_test, args.batch_size, shuffle=False, num_workers=2)
    net = create_model(num_class)
    cudnn.benchmark = True
    class_transition_matrix =  torch.zeros(num_class,num_class,args.Nb) 
    label_bank = {} 
    distri = torch.zeros((args.Nb,num_class),dtype=torch.float32).cuda() 
    label_count = 0  
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6) 
    checkpoint_path = "nnn.pth"  
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        net, optimizer, scheduler, start_epoch = load_checkpoint(net, optimizer, scheduler, checkpoint_path)
    else:
        print("No checkpoint found. Starting from scratch.")

    for epoch in range(start_epoch, args.num_epochs):
        print("epoch",epoch)
        adjust_learning_rate(args, optimizer, epoch, scheduler, 50)
        if epoch<warmup:
            rebalanced_warmup(args,warmup,num_class,train_cls_num_list, train_loader, weighted_train_loader, net, optimizer,epoch)
            test(test_loader, net, epoch, args.output_file)      
        else:
            train(args, CIFAR_noisy_train, train_loader, test_loader, net, optimizer, epoch, num_class, labeled_transform, unlabeled_transform, class_transition_matrix, label_bank, distri, label_count)  
            test(test_loader, net, epoch, args.output_file)

    end_time = time.time()
    elapsed_time = end_time - start_time    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training completed in {hours} hours {minutes} minutes {seconds} seconds.")

if __name__ == '__main__':
    main()
    