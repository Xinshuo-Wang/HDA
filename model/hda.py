import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ViT
from  .Resnet import Backbone
from models.image_extractor import get_image_extractor
from .basic_layers import MLP, BasicConv, DyCls
from .word_embedding_utils import initialize_wordembedding_matrix, load_word_embeddings
device = 'cuda:0'

def cross_entropy_loss(input, target, size_average=True):
    input = F.log_softmax(input, dim=1)
    loss = -torch.sum(input * target)
    if size_average:
        return loss / input.size(0)
    else:
        return loss


class CrossEntropyLoss(object):
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):
        return cross_entropy_loss(input, target, self.size_average)

class HDA(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset, cfg):
        super(SCOOT, self).__init__()
        self.cfg = cfg
        self.dset = dset
        self.num_attrs = len(dset.attrs)
        self.num_objs = len(dset.objs)
        self.pair2idx = dset.pair2idx
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs).cuda()

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.train_pairs)
        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()
        self.tot_attrs = torch.LongTensor(list(range(self.num_attrs))).cuda()
        self.tot_objs = torch.LongTensor(list(range(self.num_objs))).cuda()
        self.emb_dim = cfg.MODEL.emb_dim

        # Setup layers for word embedding composer.
        self._setup_word_composer(dset, cfg)

        if not cfg.TRAIN.use_precomputed_features and not cfg.TRAIN.comb_features:
            self.feat_extractor = ViT('B_16', pretrained=True)


        feat_dim = cfg.MODEL.img_emb_dim

        obj_emb_modules = [
            nn.Linear(feat_dim, self.emb_dim)
        ]
        attr_emb_modules = [
            nn.Linear(feat_dim, self.emb_dim),
        ]
        pair_emb_modules = [
            nn.Linear(feat_dim,self.emb_dim)
        ]#vit

        if cfg.MODEL.img_emb_drop > 0:
            attr_emb_modules += [nn.Dropout2d(cfg.MODEL.img_emb_drop)]
            obj_emb_modules += [nn.Dropout2d(cfg.MODEL.img_emb_drop)]
            pair_emb_modules +=[nn.Dropout2d(cfg.MODEL.img_emb_drop)]

        self.objc_embedder = nn.Sequential(*obj_emb_modules)
        self.attrc_embedder = nn.Sequential(*attr_emb_modules)
        self.pair_embedder = nn.Sequential(*pair_emb_modules)
        #
        self.img_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp) 
        self.attr_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.obj_classifier = CosineClassifier(temp=cfg.MODEL.cosine_cls_temp)
        self.cross_entropy = CrossEntropyLoss()


    def compose_visual(self, obj_feats, att_feats):
        inputs = torch.cat([obj_feats, att_feats], 1)
        output = self.projection_1(inputs)
        return output


    def _setup_word_composer(self, dset, cfg):
        attr_wordemb, self.word_dim = \
            load_word_embeddings(cfg.MODEL.wordembs, dset.attrs, cfg)# initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.attrs, cfg)
        obj_wordemb, _ = \
            load_word_embeddings(cfg.MODEL.wordembs, dset.objs, cfg) # initialize_wordembedding_matrix(cfg.MODEL.wordembs, dset.objs, cfg)

        self.attr_embedder = nn.Embedding(self.num_attrs, self.word_dim)
        self.obj_embedder = nn.Embedding(self.num_objs, self.word_dim)
        self.attr_embedder.weight.data.copy_(attr_wordemb)
        self.obj_embedder.weight.data.copy_(obj_wordemb)

        self.wordemb_compose = cfg.MODEL.wordemb_compose
        
        self.compose = nn.Sequential(
                nn.Linear(self.word_dim *2 , self.word_dim *3),
                nn.BatchNorm1d(self.word_dim*3),
                nn.ReLU(0.1),
                nn.Linear(self.word_dim *3, self.word_dim * 2),
                nn.BatchNorm1d(self.word_dim*2),
                nn.ReLU(0.1),
                nn.Linear(self.word_dim * 2, self.emb_dim)
            )



    def compose_word_embeddings(self, mode='train'):
        if mode == 'train':
            attr_emb = self.attr_embedder(self.train_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.train_objs) # # [n_pairs, word_dim].
        elif mode == 'all':
            attr_emb = self.attr_embedder(self.all_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.all_objs)
        elif mode == 'unseen':
            attr_emb = self.attr_embedder(self.unseen_pair_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.unseen_pair_objs)
        else:
            attr_emb = self.attr_embedder(self.val_attrs) # [n_pairs, word_dim].
            obj_emb = self.obj_embedder(self.val_objs) # # [n_pairs, word_dim].

        concept_emb = torch.cat((obj_emb, attr_emb), dim=-1)
        concept_emb = self.compose(concept_emb)

        return concept_emb



    def train_forward_augment(self, batch, lam=0.5):
        img1 = batch['img']
        img2_o = batch['img1_o'] # Image that shares the same object


        # Labels of 1st image.
        attr_labels_a = batch['attr']
        attr_labels_b = batch['attr1_o']
        obj_labels_a = batch['obj']
        obj_labels_b = batch['obj1_o']
        pair_labels_a = batch['pair'] #64
        pair_labels_b = batch['idx1_o']

        # concat image & label
        img1 = torch.cat((img1, img2_o)) #128 3 224 224
        pair_labels = torch.cat((pair_labels_a, pair_labels_b)) #128
        obj_labels = torch.cat((obj_labels_a, obj_labels_b))
        attr_labels = torch.cat((attr_labels_a, attr_labels_b))

        # generate word embedding
        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs) # 440 300

        bs = img1.shape[0] #128
        concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12,cls_token = self.feat_extractor(img1)#128 768

        img1_obj = self.objc_embedder(cls_token)#128 300
        obj_pred = self.classifier(img1_obj, obj_weight) #128 674
        
        img1_att = self.attrc_embedder(cls_token)
        attr_pred = self.classifier(img1_att, attr_weight)
        img1_pair = self.compose_visual(img1_obj, img1_att)

        # ea,eo,ec
        img1_pair2att = self.enc_att(img1_pair)
        img1_pair2obj = self.enc_obj(img1_pair)
        img1_ao2pair = self.dec(torch.cat((img1_att, img1_obj), dim=1))
        img1_ao2pairp = self.pair_cls(img1_ao2pair)  # duiqizhangliang

	    #p(a)' p(o)' p(c)'
        img1_pair2attp = self.pair2att_cls(img1_pair2att, img1_pair)
        img1_pair2objp = self.pair2obj_cls(img1_pair2obj, img1_pair)
        img1_pairp = self.compose_visual(img1_pair2objp, img1_pair2attp)
        img1_pair2att_pred = self.classifier(img1_pair2attp, attr_weight)
        img1_pair2obj_pred = self.classifier(img1_pair2objp, obj_weight)
        img1_pair_pred = self.classifier(img1_pairp, concept)


        # pure cot
        obj_loss = F.cross_entropy(obj_pred, obj_labels)  # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]
        attr_loss = F.cross_entropy(attr_pred, attr_labels)  # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]
        comp_pred = self.classifier(img1_pair, concept)
        comp_loss = F.cross_entropy(comp_pred, pair_labels)
        comp_pred = torch.max(comp_pred, dim=1)[1]

        img1_ao2pair_pred = self.classifier(img1_ao2pairp, concept)
       #loss-mod
        loss_att_pair = F.cross_entropy(img1_pair2att_pred, attr_labels)
        loss_obj_pair = F.cross_entropy(img1_pair2obj_pred, obj_labels)
        loss_pair = F.cross_entropy(img1_pair_pred, pair_labels)
        img1_pair2obj_pred = torch.max(img1_ao2pair_pred, dim=1)[1]
        img1_pair2att_pred = torch.max(img1_ao2pair_pred, dim=1)[1]
        img1_pair_pred = torch.max(img1_pair_pred, dim=1)[1]


        loss_s = loss_pair + loss_obj_pair + loss_att_pair

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (comp_pred == pair_labels)
        correct_attr = (attr_pred == attr_labels)

        loss = comp_loss + obj_loss + attr_loss  +  loss_s/2
              
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_comp.sum(),float(bs)) 
        }

        return out


    def train_forward(self, batch):
        img1 = batch['img']

        # Labels of 1st image.
        attr_labels = batch['attr']
        obj_labels = batch['obj']
        pair_labels = batch['pair']


        bs = img1.shape[0]

        concept = self.compose_word_embeddings(mode='train') # (n_pairs, emb_dim)

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img1)  # 128 768

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)


        img1_obj = self.objc_embedder(cls_token)
        obj_pred = self.classifier(img1_obj, obj_weight)
        
        img1_att = self.attrc_embedder(cls_token)
        attr_pred = self.classifier(img1_att, attr_weight)
        img1_pair = self.compose_visual(img1_obj, img1_att)

        # ea,eo,ec
        img1_pair2att = self.enc_att(img1_pair)
        img1_pair2obj = self.enc_obj(img1_pair)
        img1_ao2pair = self.dec(torch.cat((img1_att, img1_obj), dim=1))
        img1_ao2pairp = self.pair_cls(img1_ao2pair)  # duiqizhangliang

        #p(a)' p(o)' p(c)'
        img1_pair2attp = self.pair2att_cls(img1_pair2att, img1_pair)
        img1_pair2objp = self.pair2obj_cls(img1_pair2obj, img1_pair)
        img1_pairp = self.compose_visual(img1_pair2objp, img1_pair2attp)
        img1_pair2att_pred = self.classifier(img1_pair2attp, attr_weight)
        img1_pair2obj_pred = self.classifier(img1_pair2objp, obj_weight)
        img1_pair_pred = self.classifier(img1_pairp, concept)


        # pure cot
        obj_loss = F.cross_entropy(obj_pred, obj_labels)  # object classification
        obj_pred = torch.max(obj_pred, dim=1)[1]
        attr_loss = F.cross_entropy(attr_pred, attr_labels)  # object classification
        attr_pred = torch.max(attr_pred, dim=1)[1]
        comp_pred = self.classifier(img1_pair, concept)
        comp_loss = F.cross_entropy(comp_pred, pair_labels)
        comp_pred = torch.max(comp_pred, dim=1)[1]

        img1_ao2pair_pred = self.classifier(img1_ao2pairp, concept)
       #loss-mod
        loss_att_pair = F.cross_entropy(img1_pair2att_pred, attr_labels)
        loss_obj_pair = F.cross_entropy(img1_pair2obj_pred, obj_labels)
        loss_pair = F.cross_entropy(img1_pair_pred, pair_labels)
        img1_pair2obj_pred = torch.max(img1_ao2pair_pred, dim=1)[1]
        img1_pair2att_pred = torch.max(img1_ao2pair_pred, dim=1)[1]
        img1_pair_pred = torch.max(img1_pair_pred, dim=1)[1]


        loss_s = loss_pair + loss_obj_pair + loss_att_pair

        correct_obj = (obj_pred == obj_labels)
        correct_comp = (comp_pred == pair_labels)
        correct_attr = (attr_pred == attr_labels)


        loss = comp_loss + obj_loss + attr_loss + loss_s/2
              
        out = {
            'loss_total': loss,
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_comp.sum(),float(bs)) 
        }


        return out

    def val_forward(self, batch):
        img = batch['img']

        concept = self.compose_word_embeddings(mode='val') # [n_pairs, emb_dim].

        img_feat_l3, img_feat_l6, img_feat_l9, img_feat_l12, cls_token = self.feat_extractor(img)

        obj_weight = self.obj_embedder(self.tot_objs)
        attr_weight = self.attr_embedder(self.tot_attrs)

        vis_obj = self.objc_embedder(cls_token)
        attr_emb = self.attrc_embedder(cls_token)
        vis_comp = self.compose_visual(vis_obj, attr_emb)

        attr_pred = self.classifier(attr_emb, attr_weight, scale=False)
        attr_pred = attr_pred.index_select(1, self.pairs[:, 0])
        obj_pred = self.classifier(vis_obj, obj_weight, scale=False)
        obj_pred = obj_pred.index_select(1, self.pairs[:, 1])
        comp_pred = self.classifier(vis_comp, concept, scale=False)
        pred = comp_pred + attr_pred + obj_pred

        out = {}
        out['pred'] = pred

        out['scores'] = {}
        for _, pair in enumerate(self.val_pairs):
            out['scores'][pair] = pred[:,self.pair2idx[pair]]

        return out
    
    def forward(self, x, flag=False):
        if self.training:
            if flag:
                out = self.train_forward_augment(x)
                #out = self.train_forward(x)
            else:
                out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out



class CosineClassifier(nn.Module):
    def __init__(self, temp=0.05):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, img, concept, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        img_norm = F.normalize(img, dim=-1)
        concept_norm = F.normalize(concept, dim=-1)
        pred = torch.matmul(img_norm, concept_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        return pred
