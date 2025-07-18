from .quasi_dense_pdlan_seg_refine_all import EMQuasiDenseMaskRCNNRefineAll
import numpy as np
import random
import seaborn as sns
import cv2
import mmcv
from Cython.Compiler.Naming import self_cname
from mmdet.core import bbox2result
from mmdet.models import TwoStageDetector

from pdlan.core import track2result, segtrack2result
from ..builder import MODELS, build_tracker
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner import auto_fp16, force_fp32
from PIL import ImageColor

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .RLSA import RecurrentLinearSelfAttention

@MODELS.register_module()
class PDLANRLSA(TwoStageDetector):

    def __init__(self, tracker=None,fixed=False,*args, **kwargs): 
        self.prepare_cfg(kwargs) 
        super(PDLANRLSA,self).__init__(*args, **kwargs) 
        self.tracker_cfg = tracker 
        channels = 256
        proto_num = 35  
        stage_num = 10
        update_rate = 0.23
        self.channels = channels
        self.proto_num = proto_num
        self.stage_num = stage_num
        self.update_rate = update_rate
        self.memo_banks_key = []
        self.memo_banks_value = []
        self.device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.att0 = RecurrentLinearSelfAttention(channels)
        self.att1 = RecurrentLinearSelfAttention(channels)
        self.att2 = RecurrentLinearSelfAttention(channels)

        if fixed:
            self.fix_modules()

    def create_mu(self,batch_size):
        for i in range(5):
            protos = torch.Tensor(batch_size, self.channels, self.proto_num).to(self.device0)  
            protos.normal_(0, math.sqrt(2. / self.proto_num))  
            protos = self._l2norm(protos, dim=1) 
            self.register_buffer('mu%d' % i, protos)
            self.register_buffer('mu%d_key' % i, protos)  
            self.register_buffer('mu%d_value' % i, protos)


    def forward(self, img, img_metas, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def random_color(self,seed):  
        random.seed(seed)  
        colors = sns.color_palette(n_colors=64)  
        color = random.choice(colors) 
        return color  

    @force_fp32(apply_to=('inp',))
    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    @force_fp32(apply_to=('inp',))  
    def _l2norm(self, inp, dim):  
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

    def prepare_cfg(self, kwargs): 
        if kwargs.get('train_cfg', False): 
            kwargs['roi_head']['track_train_cfg'] = kwargs['train_cfg'].get(
                'embed', None) 

    def init_tracker(self): 
        self.tracker = build_tracker(self.tracker_cfg) 

    def fix_modules(self):  #
        fixed_modules = [
            self.backbone,
            self.neck,
            self.rpn_head,
            self.roi_head.bbox_roi_extractor,
            self.roi_head.bbox_head,
            self.roi_head.track_roi_extractor,
            self.roi_head.track_head,
            self.roi_head.mask_roi_extractor,  
            self.roi_head.mask_head]

        for module in fixed_modules:
            for name, param in module.named_parameters():
                param.requires_grad = False  #

    @torch.no_grad()
    def _em_iter(self, x, mu):

        R, C, H, W = x.size()
        x = x.view(R, C, H * W)  

        for _ in range(self.stage_num): 
            z = torch.einsum('rcn,rck->rnk', (x, mu))  # r * n * k

            z = F.softmax(20 * z, dim=2)  # r * n * k 
            z = self._l1norm(z, dim=1)  # r * n * k 
            mu = torch.einsum('rcn,rnk->rck', (x, z))  # r * c * k 
            mu = self._l2norm(mu, dim=1)  # r * c * k 

        return mu
    
    def _prop(self, feat, mu): 
        B, C, H, W = feat.size()
        x = feat.view(B, C, -1)  # B * C * N
        z = torch.einsum('bcn,bck->bnk', (x, mu))  # B * N * K
        z = F.softmax(z, dim=2)  # B * N * K
        return z

    def ADD_layernorm(self,RLSA,feature,eps=1e-5):
        B, C, H, W = feature.size()
        feature = feature.flatten(2).transpose(1, 2)
        if torch.cuda.is_available():
            feature = feature.to('cuda')  
        output = RLSA + feature  

        output_normalized = F.layer_norm(output, output.size()[1:],eps=eps) 
        output_normalized = output_normalized.transpose(1, 2).view(B, C, H, W) 

        return output_normalized

    def forward_train(self,  
                      img,   
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_match_indices,
                      ref_img, 
                      ref_img_metas,
                      ref_gt_bboxes,
                      ref_gt_labels,
                      ref_gt_match_indices,
                      gt_bboxes_ignore=None,
                      gt_masks=None, 
                      ref_gt_bboxes_ignore=None, 
                      ref_gt_masks=None, 
                      **kwargs): 
        if not isinstance(img_metas, list) or len(img_metas) == 0:
            raise ValueError("img_metas must be a non-empty list.")
        frame_id = img_metas[0].get('frame_id', -1)  
        self.x = self.extract_feat(img) 
        if frame_id == 0 or frame_id == -1: 
            self.init_tracker() 
            self.memo_banks_key = [self.x[0],self.x[1],self.x[2]] 

            RLSA2, _ = self.att2(self.x[2])
            self.memo_banks_value = [self.x[0],self.x[1],self.ADD_layernorm(RLSA2, self.x[2])]

            B, _,_,_ = self.x[0].size()
            self.create_mu(B)
            self.mus_key = [self.mu0_key,self.mu1_key,self.mu2_key]
            self.mus_value = [self.mu0_value,self.mu1_value,self.mu2_value]
        
        self.x = list(self.x)
        for i in range(3):

            B, C, H, W = self.memo_banks_key[i].size()
            protos_key = self._em_iter(self.memo_banks_key[i], self.mus_key[i])
            ref_z = self._prop(self.x[i], protos_key) 

            protos_value = self._em_iter(self.memo_banks_value[i], self.mus_value[i])

            ref_r = torch.einsum('bck,bnk->bcn', (protos_value, ref_z))
            ref_r = ref_r.view(B, C, H, W) 
            self.x[i] = self.x[i] * (1-self.update_rate) + ref_r * self.update_rate
            self.memo_banks_key[i] = self.x[i] * 0.75 + self.memo_banks_key[i] * 0.25
            self.memo_banks_value[i] = self.x[i] * 0.75 + self.memo_banks_value[i] * 0.25
            self.mus_key[i] = self.mus_key[i] * 0.5 + protos_key * 0.5
            self.mus_value[i] = self.mus_value[i] * 0.5 + protos_value * 0.5

        self.x = tuple(self.x)

        losses = dict() 

        # RPN forward and loss 

        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn) 
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            self.x,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=gt_bboxes_ignore,
            proposal_cfg=proposal_cfg) 
        losses.update(rpn_losses) 


        ref_x = self.extract_feat(ref_img) 
        ref_proposals = self.rpn_head.simple_test_rpn(ref_x, ref_img_metas) 

        roi_losses = self.roi_head.forward_train(
            self.x, img_metas, proposal_list, gt_bboxes, gt_labels,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes, ref_gt_labels, gt_bboxes_ignore, gt_masks,
            ref_gt_bboxes_ignore, ref_gt_masks, **kwargs)

        losses.update(roi_losses) 

        return losses 

    def simple_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.' 
        frame_id = img_metas[0].get('frame_id', -1) 
        if frame_id == 0: 
            self.init_tracker() 

        x = self.extract_feat(img) 
        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas) 
        det_bboxes, det_labels, track_feats = self.roi_head.simple_test(
            x, img_metas, proposal_list, rescale) 

        if track_feats is not None: 
            bboxes, labels, ids = self.tracker.match(
                bboxes=det_bboxes,
                labels=det_labels,
                track_feats=track_feats,
                frame_id=frame_id)

        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes) 

        if track_feats is not None:
            track_result = track2result(bboxes, labels,ids) 
        else: 
            from collections import defaultdict
            track_result = defaultdict(list) 
        return dict(bbox_result=bbox_result, track_result=track_result)

    def forward_test(self, img, img_metas, rescale=False):
        # TODO inherit from a base tracker
        assert self.roi_head.with_track, 'Track head must be implemented.' 
        img_metas = img_metas[0]
        frame_id = img_metas[0].get('frame_id', -1)  
        x = self.extract_feat(img[0]) 

        if frame_id == 0 : 
            self.init_tracker() 
            self.memo_banks_key = [x[0],x[1],x[2]] 
            RLSA0, _ = self.att0(x[2])
            RLSA1, _ = self.att1(x[2])
            RLSA2, _ = self.att2(x[2])
            self.memo_banks_value = [self.ADD_layernorm(RLSA2, x[0]),self.ADD_layernorm(RLSA2, x[1]),self.ADD_layernorm(RLSA2, x[2])]
            B, _,_,_ = x[0].size()
            self.create_mu(B)
            self.mus_key = [self.mu0_key,self.mu1_key,self.mu2_key]
            self.mus_value = [self.mu0_value,self.mu1_value,self.mu2_value]

        x = list(x)

        for i in range(3): 
            B, C, H, W = self.memo_banks_key[i].size()
            protos_key = self._em_iter(self.memo_banks_key[i], self.mus_key[i])
            ref_z = self._prop(x[i], protos_key) 
            protos_value = self._em_iter(self.memo_banks_value[i], self.mus_value[i])
            ref_r = torch.einsum('bck,bnk->bcn', (protos_value, ref_z))
            ref_r = ref_r.view(B, C, H, W) 
            x[i] = x[i] * (1-self.update_rate) + ref_r * self.update_rate
            self.memo_banks_key[i] = x[i] * 0.75 + self.memo_banks_key[i] * 0.25
            self.memo_banks_value[i] = x[i] * 0.75 + self.memo_banks_value[i] * 0.25
            self.mus_key[i] = self.mus_key[i] * 0.5 + protos_key * 0.5
            self.mus_value[i] = self.mus_value[i] * 0.5 + protos_value * 0.5


        x = tuple(x)



        proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        det_bboxes, det_labels, det_masks, track_feats = (
            self.roi_head.simple_test(x, img_metas, proposal_list, rescale))
        # print(det_labels)
        bbox_result = bbox2result(det_bboxes, det_labels,
                                  self.roi_head.bbox_head.num_classes)
        segm_result, ori_segms, labels_ori = self.roi_head.get_seg_masks(
            img_metas, det_bboxes, det_labels, det_masks, rescale=rescale)
        update_cls_segms = [[] for _ in range(self.roi_head.bbox_head.num_classes)]

        if track_feats is None:
            from collections import defaultdict
            track_result = defaultdict(list)
            refine_bbox_result = bbox_result
            update_cls_segms = segm_result
        else:
           
            bboxes, labels, masks, ids, embeds, ref_feats, ref_masks, inds, valids = (
                self.tracker.match(
                    bboxes=det_bboxes,
                    labels=det_labels,
                    masks=det_masks,
                    track_feats=track_feats,
                    frame_id=frame_id))


            mask_preds, mask_feats = masks['mask_pred'], masks['mask_feats']

            refine_segm_result, segms, refine_preds = self.roi_head.simple_test_refine(
                img_metas, mask_feats, mask_preds, bboxes, labels, ref_feats,
                ref_masks, rescale=rescale)
            # print("segms:",segms)

            ori_segms = np.array(ori_segms)
            ori_segms = ori_segms[list(inds.cpu().numpy()), :]

            labels_ori = labels_ori[inds]
            valids = list(valids.cpu().numpy())
            valids_new = [ind2 for ind2 in range(len(valids)) if valids[ind2] == True]

            ori_segms[valids_new, :] = segms
            ori_segms = list(ori_segms)

            for i1 in range(len(ori_segms)):
                update_cls_segms[labels_ori[i1]].append(ori_segms[i1])

            self.tracker.update_memo(ids, bboxes, mask_preds, mask_feats,
                                     refine_preds, embeds, labels, frame_id)

            track_result = segtrack2result(bboxes, labels, segms, ids)

        return dict(bbox_result=bbox_result, segm_result=update_cls_segms,
                    track_result=track_result)

    def show_result(self,
                    img,
                    result,
                    show=False,
                    out_file=None,
                    score_thr=0.3,
                    draw_track=True):
        track_result = result['track_result']
        img = mmcv.bgr2rgb(img)

        img = mmcv.imread(img)
        for id, item in track_result.items():
            bbox = item['bbox']
            if bbox[-1] <= score_thr:
                continue
            color = (np.array(self.random_color(id)) * 256).astype(np.uint8)
            mask = item['segm']
            img[mask] = img[mask] * 0.5 + color * 0.5
            label_text = '{}'.format(int(id))
            mask_bool = mask.astype(bool)
            area = np.sum(mask_bool)
            y, x = np.where(mask_bool)
            center_y = np.mean(y).astype(int)
            center_x = np.mean(x).astype(int)
            font_size = min(5,max(3, area / 500))
            text_x = center_x - font_size * 2
            text_y = center_y - font_size
            plt.text(text_x, text_y, label_text, fontsize=font_size, color='white')

        plt.imshow(img)
        plt.gca().set_axis_off()
        plt.autoscale(False)
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=None, wspace=None)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        if out_file is not None:
            mmcv.imwrite(img, out_file)
            plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.clf()
        return img
    
