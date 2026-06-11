import torch
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random

from .AL import AL


class PCB(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device 
        self.pred = []
        self.statistics = statistics

    def select(self, n_query, **kwargs):
        
        '''
        def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, **kwargs):
        self.unlabeled_dst = unlabeled_dst     # 1. 原始的、完整的未标注数据集
        self.U_index = U_index                  # 2. 部分被选出来的索引
        self.unlabeled_set = torch.utils.data.Subset(unlabeled_dst, U_index)    # 3. 根据全局索引，创建一个 PyTorch 的 Subset 对象，方便后续处理
        self.n_unlabeled = len(self.unlabeled_set) # 4. 未标注样本的总数
        self.n_class = n_class         # 5. 数据集的总类别数
        self.model = model              # 6. 当前的模型
        self.index = []
        self.cfg = cfg          
        '''
        self.pred = [] # 每次都清空上一轮的预测记录
        self.model.eval()
        # embDim = self.model.image_encoder.attnpool.c_proj.out_features
        num_unlabeled = len(self.U_index)                                 #num_unlabeled 和 u_index 是一样的，只是u_index属于self，但是这是u_index 是从全部样本取出来的几个样本里面的索引，并非全部的索引
        assert len(self.unlabeled_set) == num_unlabeled, f"{len(self.unlabeled_dst)} != {num_unlabeled}"
      
        # Store features for t-SNE visualization
        all_unlabeled_features = []
        
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,#0
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,#16
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )
            
            # generate entire unlabeled features set
            for i, batch in enumerate(unlabeled_loader):
                inputs = batch["img"].to(self.device)

                #只对其进行预测，我认为是获取伪标签
                out, features = self.model(image=inputs, get_feature=True)
                
                #将模型的原始输出(logits)通过 softmax 转换为概率分布
                batchProbs = torch.nn.functional.softmax(out, dim=1).data
                
                #softmax和linear层的选择分类

                #找出每个样本概率最高的类别作为模型的"猜测"
                maxInds = torch.argmax(batchProbs, 1)

                # _, preds = torch.max(out.data, 1)
                
                self.pred.append(maxInds.detach().cpu()) #pred只是伪标签，通过softmax分类
                all_unlabeled_features.append(features.detach().cpu())


        self.pred = torch.cat(self.pred) # 将所有批次的"猜测"结果拼接成一个完整的张量
        
        # Store unlabeled features for visualization
        self.unlabeled_features = torch.cat(all_unlabeled_features) if all_unlabeled_features else None
        
        Q_index = [] #这个是每次都清空
        
        ###########################
        while len(Q_index) < n_query:
            min_cls = int(torch.argmin(self.statistics))

            # 找出属于该类的未标注样本索引（相对位置）
            sub_pred = (self.pred == min_cls).nonzero().squeeze(dim=1).tolist()

            # 如果该类中没有候选，则随机选一个备用
            if len(sub_pred) == 0:
                num = random.randint(0, num_unlabeled - 1)
                while num in Q_index:
                    num = random.randint(0, num_unlabeled - 1)
                Q_index.append(num)
                continue

        # --------------------------------------------------------
        #  按照 MEH_Selector 输出的顺序（U_index 顺序）
        #    也就是不确定度从高到低的顺序进行选择
        # --------------------------------------------------------
            sub_pred_sorted = sorted(
                sub_pred, 
                key=lambda i: self.U_index.index(self.U_index[i])
            )

        # 从该类中选第一个未被选过的样本
            selected = False
            for idx in sub_pred_sorted:
                if idx not in Q_index:
                    Q_index.append(idx)
                    self.statistics[min_cls] += 1
                    selected = True
                    break

            # 如果没选到（全部重复），则随机补一个
            if not selected:
                num = random.randint(0, num_unlabeled - 1)
                while num in Q_index:
                    num = random.randint(0, num_unlabeled - 1)
                Q_index.append(num)








        ####################################################
        # while len(Q_index) < n_query: #n_query指的是每一批中去标注的数据，前面self定义为num classes

        #     min_cls = int(torch.argmin(self.statistics)) #statistics是数据集中每个类别数量，这里是找出数量最少的类别
        #     #这里面有个问题，种类有102个，但是第一轮也只有102，意味着一开始有很多东西是0，而他会先从小的选取，这样子有什么问题，如何改进。#（错——然后一直随机的话很难随机的到0

        #     sub_pred = (self.pred == min_cls).nonzero().squeeze(dim=1).tolist()##我认为是找出种类数量最少的每一个个体

        #     #sub_pred 是个体，Q_index又是什么
        #     if len(sub_pred) == 0:       #退化，又变成随机
        #         num = random.randint(0, num_unlabeled-1)
        #         while num in Q_index:
        #             num = random.randint(0, num_unlabeled-1)
        #         Q_index.append(num)

        #     else:   
        #         random.shuffle(sub_pred)
        #         #输出一个位置列表，相对位置 比如说
        #         #pred是[2,3,2,1,3,1,2,3] 1 2 3代表种类 ,1 是最少的
        #         #subpred 返回的是[3,5]
        #         for idx in sub_pred:
        #             if idx not in Q_index:
        #                 Q_index.append(idx)
        #                 self.statistics[min_cls] += 1
        #                 break 
        #         else: 
        #             num = random.randint(0, num_unlabeled-1)
        #             while num in Q_index:
        #                 num = random.randint(0, num_unlabeled-1)
        #             Q_index.append(num)

            
        Q_index = [self.U_index[idx] for idx in Q_index]
        
        # Store query indices for visualization (relative to U_index)
        self.query_indices = Q_index
        
        return Q_index
    
    def get_features_for_visualization(self):
        """
        Extract features for labeled and unlabeled data for t-SNE visualization.
        Returns: (unlabeled_features, query_features)
        """
        if not hasattr(self, 'unlabeled_features') or self.unlabeled_features is None:
            return None, None
        
        return self.unlabeled_features, None
